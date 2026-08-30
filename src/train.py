import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["JAX_LOG_COMPILES"] = "1"
# os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"

import json
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import mlflow
import dagshub
from tqdm import tqdm

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from core.dataset import DataLoader
from core.losses import loss_fn
from core.engine import pmap_train_block, restore_step_count
from model.tensorf import TensoRF, upsample_tensoRF, update_alpha_mask, shrink_bbox
from visualization import evaluate_test_psnr


def device_put_sharded(shards, devices):
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(lambda *xs: jax.device_put(jnp.stack(xs), sharding), *shards)


def device_put_replicated(tree, devices):
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(
        lambda x: jax.device_put(
            jnp.broadcast_to(x, (len(devices),) + x.shape), sharding
        ),
        tree,
    )


def partition_model(model):
    params, rest = eqx.partition(model, eqx.is_inexact_array)
    static_arrays, static = eqx.partition(rest, eqx.is_array)
    return params, static_arrays, static


def build_schedule(n_iters):
    milestones = [2000, 3000, 4000, 5500, 7000]
    filtered = [m for m in milestones if m < n_iters]
    schedule = sorted(list(set(filtered + [n_iters])))
    return schedule


def main(args):
    backend = jax.default_backend()
    devices = jax.local_devices()
    n_devices = len(devices)

    print("\n" + "=" * 50)
    print(f"Hardware Detected: {backend.upper()} with {n_devices} core(s).")
    if backend == "cpu":
        print(
            "⚠️  WARNING: JAX is running on CPU! Double check your Kaggle environment and JAX installation."
        )
    print("=" * 50 + "\n")

    try:
        dagshub.init(repo_owner="silvergrace26", repo_name="TensoRF", mlflow=True)
        mlflow.set_experiment("TensoRF_JAX_Training")
    except Exception as e:
        print(f"Tracking initialization skipped/failed: {e}")

    with mlflow.start_run(
        run_name=f"TensoRF_grid{args.init_grid_dim}_iters{args.n_iters}"
    ):
        mlflow.log_params(vars(args))
        mlflow.log_param("backend", backend)
        mlflow.log_param("n_devices", n_devices)

        n_comp_den = [16, 16, 16]
        n_comp_app = [48, 48, 48]
        mlflow.log_param("n_comp_den", n_comp_den)
        mlflow.log_param("n_comp_app", n_comp_app)

        dataset = DataLoader(base_dir=args.data_dir, split="train", half_res=False)
        test_dataset = DataLoader(base_dir=args.data_dir, split="test", half_res=False)

        os.makedirs(args.ckpt_dir, exist_ok=True)
        ckpt_prefix = os.path.join(args.ckpt_dir, "tensorf_ckpt")

        BATCH_SIZE_PER_DEVICE = max(1, args.global_batch_size // max(1, n_devices))
        print(
            f"Global Batch Size: {args.global_batch_size} | Per Device Chunk: {BATCH_SIZE_PER_DEVICE}"
        )

        initial_grid_dim = args.init_grid_dim
        res_map = {2000: 150, 3000: 200, 4000: 300, 5500: 400, 7000: 512}
        schedule = build_schedule(args.n_iters)

        TV_START_WEIGHT = 0.5
        TV_END_WEIGHT = 0.01

        key = jax.random.PRNGKey(42)
        model_key, train_key = jax.random.split(key)

        current_step = 0

        warmup_steps = min(1000, max(1, args.n_iters // 2))
        schedule_decay_steps = max(args.n_iters, warmup_steps + 1)

        lr_schedule_grids = optax.warmup_cosine_decay_schedule(
            init_value=2e-3,
            peak_value=2e-2,
            warmup_steps=warmup_steps,
            decay_steps=schedule_decay_steps,
            end_value=2e-3,
        )
        lr_schedule_mlp = optax.warmup_cosine_decay_schedule(
            init_value=1e-4,
            peak_value=1e-3,
            warmup_steps=warmup_steps,
            decay_steps=schedule_decay_steps,
            end_value=1e-4,
        )

        optim_grids = optax.adam(lr_schedule_grids, b1=0.9, b2=0.99)
        optim_mlp = optax.adam(lr_schedule_mlp, b1=0.9, b2=0.99)

        def label_fn(tree):
            labels = jax.tree_util.tree_map(lambda _: "grid", tree)
            if hasattr(tree, "mlp_render") and tree.mlp_render is not None:
                labels = eqx.tree_at(
                    lambda m: m.mlp_render,
                    labels,
                    jax.tree_util.tree_map(lambda _: "mlp", tree.mlp_render),
                )
            if hasattr(tree, "basis_mat") and tree.basis_mat is not None:
                labels = eqx.tree_at(
                    lambda m: m.basis_mat,
                    labels,
                    jax.tree_util.tree_map(lambda _: "mlp", tree.basis_mat),
                )
            return labels

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.multi_transform({"grid": optim_grids, "mlp": optim_mlp}, label_fn),
        )

        if (
            os.path.exists(ckpt_prefix + "_meta.json")
            and os.path.exists(ckpt_prefix + "_model.eqx")
            and os.path.exists(ckpt_prefix + "_opt.eqx")
        ):
            print("Found existing checkpoint. Restoring weights and optimizer state...")
            with open(ckpt_prefix + "_meta.json", "r") as f:
                meta = json.load(f)

            current_step = meta["step"]
            initial_grid_dim = meta["grid_dim"]

            model = TensoRF(
                model_key,
                grid_dim=initial_grid_dim,
                n_comp_den=n_comp_den,
                n_comp_app=n_comp_app,
            )

            params, static_arrays, static = partition_model(model)
            model = eqx.combine(params, static_arrays, static)
            model = eqx.tree_deserialise_leaves(ckpt_prefix + "_model.eqx", model)
            params, static_arrays, static = partition_model(model)

            opt_state = optimizer.init(params)
            opt_state = eqx.tree_deserialise_leaves(ckpt_prefix + "_opt.eqx", opt_state)
            print(
                f"Resumed successfully at Step {current_step} | Grid Size {initial_grid_dim}."
            )
        else:
            print("No checkpoint found. Initializing fresh model...")
            model = TensoRF(
                model_key,
                grid_dim=initial_grid_dim,
                n_comp_den=n_comp_den,
                n_comp_app=n_comp_app,
            )
            params, static_arrays, static = partition_model(model)
            opt_state = optimizer.init(params)

        print("\nReplicating parameters across hardware devices...")
        H, W = dataset.H, dataset.W
        imgs_jax = jnp.array(dataset.imgs)
        rays_o_jax = jnp.array(dataset.rays_o)
        rays_d_jax = jnp.array(dataset.rays_d)

        params_rep = device_put_replicated(params, devices)
        opt_state_rep = device_put_replicated(opt_state, devices)

        print("Starting optimized block loop...")

        device_keys_list = list(jax.random.split(train_key, n_devices))
        device_keys = device_put_sharded(device_keys_list, devices)

        final_loss, final_mse, psnr = 0.0, 0.0, 0.0

        for next_upsample in schedule:
            steps_in_block = next_upsample - current_step

            if steps_in_block <= 0:
                continue

            alpha = current_step / max(1, args.n_iters)
            current_tv_weight = np.exp(
                (1 - alpha) * np.log(TV_START_WEIGHT) + alpha * np.log(TV_END_WEIGHT)
            )
            actual_tv_lambda = current_tv_weight * 1e-4

            print(f"\nTargeting Step {next_upsample} (Upsample Bound)...")
            start_time = time.time()

            chunk_size = 100
            with tqdm(total=steps_in_block, desc="Training") as pbar:
                while current_step < next_upsample:
                    run_steps = min(chunk_size, next_upsample - current_step)
                    if run_steps <= 0:
                        break

                    params_rep, opt_state_rep, device_keys, losses, mses = (
                        pmap_train_block(
                            params_rep,
                            opt_state_rep,
                            static_arrays,
                            static,
                            device_keys,
                            imgs_jax,
                            rays_o_jax,
                            rays_d_jax,
                            H,
                            W,
                            current_step,
                            run_steps,
                            actual_tv_lambda,
                            optimizer,
                            BATCH_SIZE_PER_DEVICE,
                            args.verbose,
                        )
                    )

                    # --- SYNCHRONIZATION BARRIER ---
                    losses.block_until_ready()
                    # -------------------------------

                    current_step += run_steps

                    if losses.shape[-1] > 0:
                        final_loss = float(jnp.mean(losses[:, -1]))
                        final_mse = float(jnp.mean(mses[:, -1]))
                        psnr = -10.0 * np.log10(max(final_mse, 1e-10))

                    pbar.set_postfix(
                        {"Loss": f"{final_loss:.4f}", "PSNR": f"{psnr:.2f} dB"}
                    )
                    pbar.update(run_steps)

                    pbar.set_postfix(
                        {"Loss": f"{final_loss:.4f}", "PSNR": f"{psnr:.2f} dB"}
                    )
                    pbar.update(run_steps)

            print(
                f"Reached Step {next_upsample} | Final Step Loss: {final_loss:.5f} | PSNR: {psnr:.2f} dB | Time: {time.time() - start_time:.1f}s"
            )

            mlflow.log_metrics(
                {
                    "train_loss": final_loss,
                    "train_psnr": psnr,
                    "grid_dim": initial_grid_dim,
                },
                step=next_upsample,
            )

            if current_step in res_map:
                new_dim = res_map[current_step]
                print(
                    f"[Upsampling Boundary] Iter {current_step}: {initial_grid_dim} -> {new_dim}"
                )

                params_single = jax.tree_util.tree_map(lambda x: x[0], params_rep)
                model = eqx.combine(params_single, static_arrays, static)

                print("Evaluating Occupancy Grid and Shrinking Bounding Box...")
                model = update_alpha_mask(model)
                model = shrink_bbox(model)

                model = eqx.tree_at(
                    lambda m: m.alpha_mask,
                    model,
                    jax.device_put(jnp.ones_like(model.alpha_mask)),
                )

                model = upsample_tensoRF(model, new_dim, train_key)
                params, static_arrays, static = partition_model(model)

                new_opt_state = optimizer.init(params)
                old_state_single = jax.tree_util.tree_map(lambda x: x[0], opt_state_rep)
                opt_state = restore_step_count(new_opt_state, old_state_single)

                params_rep = device_put_replicated(params, devices)
                opt_state_rep = device_put_replicated(opt_state, devices)
                initial_grid_dim = new_dim

            print(f"Saving checkpoint at step {current_step}...")
            params_single = jax.tree_util.tree_map(lambda x: x[0], params_rep)
            opt_state_single = jax.tree_util.tree_map(lambda x: x[0], opt_state_rep)

            model_to_save = eqx.combine(params_single, static_arrays, static)
            eqx.tree_serialise_leaves(ckpt_prefix + "_model.eqx", model_to_save)
            eqx.tree_serialise_leaves(ckpt_prefix + "_opt.eqx", opt_state_single)

            with open(ckpt_prefix + "_meta.json", "w") as f:
                json.dump(
                    {
                        "step": current_step,
                        "grid_dim": initial_grid_dim,
                        "n_comp_den": n_comp_den,
                        "n_comp_app": n_comp_app,
                    },
                    f,
                )
            print("Checkpoint saved successfully.")

        print("\n============================================================")
        print("Training Complete!")
        print(f"Final grid dimension: {initial_grid_dim}")
        print("============================================================")

        params_final = jax.tree_util.tree_map(lambda x: x[0], params_rep)
        test_psnr = evaluate_test_psnr(
            params_final, static_arrays, static, test_dataset
        )

        mlflow.log_metric("test_psnr", test_psnr, step=args.n_iters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train JAX/Equinox TensoRF on GPU or TPU"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to NeRF synthetic dataset"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints",
        help="Directory to save/load checkpoints",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=8192,
        help="Global batch size across all devices",
    )
    parser.add_argument(
        "--n_iters", type=int, default=30000, help="Total number of training iterations"
    )
    parser.add_argument(
        "--init_grid_dim", type=int, default=128, help="Initial tensor grid size"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed diagnostic logs before training",
    )

    args = parser.parse_args()

    main(args)
