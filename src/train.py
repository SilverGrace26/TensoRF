import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from core.dataset import DataLoader
from core.losses import loss_fn
from core.engine import pmap_train_block, restore_step_count
from model.tensorf import TensoRF, upsample_tensoRF
from visualization import evaluate_test_psnr


def device_put_sharded(shards, devices):
    """Drop-in replacement for jax.device_put_sharded"""
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(lambda *xs: jax.device_put(jnp.stack(xs), sharding), *shards)


def device_put_replicated(tree, devices):
    """Drop-in replacement for jax.device_put_replicated"""
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(
        lambda x: jax.device_put(
            jnp.broadcast_to(x, (len(devices),) + x.shape), sharding
        ),
        tree,
    )


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

    dataset = DataLoader(base_dir=args.data_dir, split="train", half_res=False)
    test_dataset = DataLoader(base_dir=args.data_dir, split="test", half_res=False)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_prefix = os.path.join(args.ckpt_dir, "tensorf_ckpt")

    BATCH_SIZE_PER_DEVICE = args.global_batch_size // max(1, n_devices)
    print(
        f"Global Batch Size: {args.global_batch_size} | Per Device Chunk: {BATCH_SIZE_PER_DEVICE}"
    )

    initial_grid_dim = args.init_grid_dim
    res_map = {2000: 150, 3000: 200, 4000: 300, 5500: 400, 7000: 512}
    schedule = [2000, 3000, 4000, 5500, 7000, args.n_iters]

    TV_START_WEIGHT = 0.5
    TV_END_WEIGHT = 0.01
    PRECROP_ITERS = 1000

    key = jax.random.PRNGKey(42)
    model_key, train_key = jax.random.split(key)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4,
        peak_value=2e-2,
        warmup_steps=2000,
        decay_steps=args.n_iters,
        end_value=1e-3,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(lr_schedule, b1=0.9, b2=0.99)
    )

    current_step = 0

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

        model = TensoRF(model_key, grid_dim=initial_grid_dim)
        model = eqx.tree_at(
            lambda m: m.mlp_render,
            model,
            jax.tree_util.tree_map(
                lambda x: x * 0.1 if eqx.is_array(x) else x, model.mlp_render
            ),
        )
        params, static = eqx.partition(model, eqx.is_array)
        opt_state = optimizer.init(params)

        model = eqx.combine(params, static)
        model = eqx.tree_deserialise_leaves(ckpt_prefix + "_model.eqx", model)
        params, static = eqx.partition(model, eqx.is_array)
        opt_state = eqx.tree_deserialise_leaves(ckpt_prefix + "_opt.eqx", opt_state)
        print(
            f"Resumed successfully at Step {current_step} | Grid Size {initial_grid_dim}."
        )
    else:
        print("No checkpoint found. Initializing fresh model...")
        model = TensoRF(model_key, grid_dim=initial_grid_dim)
        model = eqx.tree_at(
            lambda m: m.mlp_render,
            model,
            jax.tree_util.tree_map(
                lambda x: x * 0.1 if eqx.is_array(x) else x, model.mlp_render
            ),
        )
        params, static = eqx.partition(model, eqx.is_array)
        opt_state = optimizer.init(params)

    print("\n--- XLA Cost Analysis for a Single Step ---")
    mock_rays_o = jnp.zeros((BATCH_SIZE_PER_DEVICE, 3))
    mock_rays_d = jnp.zeros((BATCH_SIZE_PER_DEVICE, 3))
    mock_rgb = jnp.zeros((BATCH_SIZE_PER_DEVICE, 3))
    mock_key = jax.random.PRNGKey(0)

    @jax.jit
    def test_forward_backward(p):
        m_bench = eqx.combine(p, static)
        return eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            m_bench,
            mock_rays_o,
            mock_rays_d,
            mock_rgb,
            mock_key,
            0.5,
            4e-5,
            jnp.array([1.0, 1.0, 1.0]),
        )

    lowered = test_forward_backward.lower(params)
    compiled = lowered.compile()

    try:
        costs = compiled.cost_analysis()
        for key_cost, val_cost in costs.items():
            print(f"{key_cost}: {val_cost}")
    except Exception as e:
        print(f"Cost analysis not available for this backend: {e}")
    print("-------------------------------------------\n")

    print("Replicating parameters across hardware devices...")
    focal, H, W = dataset.focal, dataset.H, dataset.W
    imgs_jax = jnp.array(dataset.imgs)
    poses_jax = jnp.array(dataset.poses)

    params_rep = device_put_replicated(params, devices)
    opt_state_rep = device_put_replicated(opt_state, devices)

    print("Starting optimized block loop...")
    start_time = time.time()

    device_keys_list = list(jax.random.split(train_key, n_devices))
    device_keys = device_put_sharded(device_keys_list, devices)

    for next_upsample in schedule:
        steps_to_run = next_upsample - current_step

        if steps_to_run <= 0:
            continue

        is_precrop = current_step < PRECROP_ITERS
        alpha = current_step / args.n_iters
        current_tv_weight = np.exp(
            (1 - alpha) * np.log(TV_START_WEIGHT) + alpha * np.log(TV_END_WEIGHT)
        )
        actual_tv_lambda = current_tv_weight * 1e-4

        print(
            f"\nRunning block: steps {current_step} to {next_upsample} (Precrop: {is_precrop})..."
        )

        params_rep, opt_state_rep, device_keys, losses, mses = pmap_train_block(
            params_rep,
            opt_state_rep,
            static,
            device_keys,
            imgs_jax,
            poses_jax,
            focal,
            H,
            W,
            steps_to_run,
            is_precrop,
            actual_tv_lambda,
            optimizer,
            BATCH_SIZE_PER_DEVICE,
        )

        final_mse = mses[0][-1].item()
        psnr = -10.0 * np.log10(final_mse)
        print(
            f"Reached Step {next_upsample} | Final Step Loss: {losses[0][-1].item():.5f} | PSNR: {psnr:.2f} dB | Time: {time.time() - start_time:.1f}s"
        )
        start_time = time.time()

        current_step = next_upsample

        if current_step in res_map:
            new_dim = res_map[current_step]
            print(
                f"[Upsampling Boundary] Iter {current_step}: {initial_grid_dim} -> {new_dim}"
            )

            params_single = jax.tree_util.tree_map(lambda x: x[0], params_rep)
            model = eqx.combine(params_single, static)

            model = upsample_tensoRF(model, new_dim, train_key)
            params, static = eqx.partition(model, eqx.is_array)

            new_opt_state = optimizer.init(params)
            old_state_single = jax.tree_util.tree_map(lambda x: x[0], opt_state_rep)
            opt_state = restore_step_count(new_opt_state, old_state_single)

            params_rep = device_put_replicated(params, devices)
            opt_state_rep = device_put_replicated(opt_state, devices)
            initial_grid_dim = new_dim

        print(f"Saving checkpoint at step {current_step}...")
        params_single = jax.tree_util.tree_map(lambda x: x[0], params_rep)
        opt_state_single = jax.tree_util.tree_map(lambda x: x[0], opt_state_rep)

        model_to_save = eqx.combine(params_single, static)
        eqx.tree_serialise_leaves(ckpt_prefix + "_model.eqx", model_to_save)
        eqx.tree_serialise_leaves(ckpt_prefix + "_opt.eqx", opt_state_single)

        with open(ckpt_prefix + "_meta.json", "w") as f:
            json.dump({"step": current_step, "grid_dim": initial_grid_dim}, f)
        print("Checkpoint saved successfully.")

    print("\n============================================================")
    print("Training Complete!")
    print(f"Final grid dimension: {initial_grid_dim}")
    print("============================================================")

    params_final = jax.tree_util.tree_map(lambda x: x[0], params_rep)
    evaluate_test_psnr(params_final, static, test_dataset, train_key)


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
    args = parser.parse_args()

    main(args)
