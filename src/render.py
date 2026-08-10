import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import json
import shutil
import argparse
import jax
import equinox as eqx

from core.dataset import DataLoader
from model.tensorf import TensoRF
from visualization import (
    save_point_cloud,
    save_component_plot,
    save_density_slice,
    save_raw_planes,
    render_360_video,
)


def main(args):
    print(f"🔥 Hardware Detected for Rendering: {jax.default_backend().upper()}")
    print(f"Setting up output directory at {args.out_dir}...")

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(f"{args.out_dir}/planes", exist_ok=True)
    os.makedirs(f"{args.out_dir}/frames", exist_ok=True)

    ckpt_prefix = os.path.join(args.ckpt_dir, "tensorf_ckpt")
    if not os.path.exists(ckpt_prefix + "_meta.json"):
        raise FileNotFoundError("Could not find model metadata. Did training finish?")

    with open(ckpt_prefix + "_meta.json", "r") as f:
        meta = json.load(f)

    key = jax.random.PRNGKey(0)
    model = TensoRF(key, grid_dim=meta["grid_dim"])
    model = eqx.tree_deserialise_leaves(ckpt_prefix + "_model.eqx", model)
    print(f"✔ Model loaded successfully (grid size: {meta['grid_dim']})")

    dataset = DataLoader(base_dir=args.data_dir, split="test", half_res=False)

    save_point_cloud(model, f"{args.out_dir}/pointcloud.ply")
    save_component_plot(model, f"{args.out_dir}/components.png")
    save_density_slice(model, f"{args.out_dir}/slice_z0.png", z_val=0.0)
    save_raw_planes(model, f"{args.out_dir}/planes")

    render_360_video(model, dataset, f"{args.out_dir}/frames", n_frames=args.n_frames)

    print("\n→ Zipping results…")
    shutil.make_archive(args.out_dir, "zip", args.out_dir)
    print(f"\n🎉 DONE! Visualizations packed into {args.out_dir}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluations and render TensoRF")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to NeRF dataset"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Path where checkpoints are saved"
    )
    parser.add_argument(
        "--out_dir", type=str, default="tensorf_results", help="Where to save outputs"
    )
    parser.add_argument(
        "--n_frames", type=int, default=30, help="Number of frames for 360 video"
    )
    args = parser.parse_args()

    main(args)
