import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
from functools import partial


def evaluate_test_psnr(params, static_arrays, static, test_dataset, key=None):
    print("\n--- Running Test Set Evaluation ---")

    model_infer = eqx.combine(params, static_arrays, static)

    @jax.jit
    def render_chunk(rays_o_chunk, rays_d_chunk):
        bg_color = jnp.array([1.0, 1.0, 1.0])
        rgb, _, _ = model_infer(rays_o_chunk, rays_d_chunk, None, bg_color)
        return rgb

    chunk_size = 8192
    total_mse = 0.0

    for i in range(test_dataset.N):
        rays_o, rays_d = test_dataset.get_full_image_rays(i)
        target_rgb = test_dataset.imgs[i]

        flat_o = rays_o.reshape(-1, 3)
        flat_d = rays_d.reshape(-1, 3)
        flat_target = target_rgb.reshape(-1, 3)

        pred_rgb_chunks = []
        for k in range(0, flat_o.shape[0], chunk_size):
            chunk_o = jnp.array(flat_o[k : k + chunk_size])
            chunk_d = jnp.array(flat_d[k : k + chunk_size])

            rgb_chunk = render_chunk(chunk_o, chunk_d)
            pred_rgb_chunks.append(rgb_chunk)

        img_pred = jnp.concatenate(pred_rgb_chunks, axis=0)
        mse = jnp.mean((img_pred - flat_target) ** 2)
        total_mse += mse

    avg_mse = total_mse / test_dataset.N
    test_psnr = -10.0 * np.log10(avg_mse)

    print(f"Test Evaluation Complete | Test PSNR: {test_psnr:.2f} dB")
    return test_psnr


def save_point_cloud(model, filename, threshold=15.0):
    print("\n→ Generating Point Cloud…")
    N = 128
    grid = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, 1, N),
            jnp.linspace(0, 1, N),
            jnp.linspace(0, 1, N),
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 3)

    chunk = 100_000
    sigmas = []

    for i in tqdm(range(0, grid.shape[0], chunk)):
        batch = grid[i : i + chunk]
        sigma, _ = model.get_sigma_feat(batch)
        sigmas.append(sigma)

    sigma = jnp.concatenate(sigmas).squeeze()
    mask = np.array(sigma > threshold)
    pts = np.array(grid)[mask] * 3.0 - 1.5

    header = f"""ply
format ascii 1.0
element vertex {pts.shape[0]}
property float x
property float y
property float z
end_header
"""
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, pts, fmt="%.5f")
    print(f"✔ Saved {pts.shape[0]} points → {filename}")


def save_component_plot(model, out_path):
    print("\n→ Plotting Density Planes…")
    planes = [np.array(p) for p in model.den_planes]
    titles = ["XY", "XZ", "YZ"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, ax in enumerate(axes):
        img = np.sum(planes[i], axis=0)
        im = ax.imshow(img, cmap="viridis", origin="lower")
        ax.set_title(f"{titles[i]} Plane")
        plt.colorbar(im, ax=ax)

    plt.suptitle(f"Learned Components (grid {model.grid_dim})")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"✔ Saved → {out_path}")


def save_density_slice(model, out_path, z_val=0.0):
    print("\n→ Generating Density Slice…")
    N = 400
    x = jnp.linspace(0, 1, N)
    y = jnp.linspace(0, 1, N)
    xx, yy = jnp.meshgrid(x, y)

    z_norm = (z_val + 1.5) / 3.0
    zz = jnp.ones_like(xx) * z_norm

    coords = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    sigma, _ = model.get_sigma_feat(coords)
    sigma = sigma.reshape(N, N)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(sigma), cmap="magma", origin="lower")
    plt.colorbar()
    plt.title(f"Density Slice @ z={z_val}")
    plt.savefig(out_path)
    plt.close()
    print(f"✔ Saved → {out_path}")


def save_raw_planes(model, dir_out):
    print("\n→ Writing Raw Plane Textures…")

    def norm(x):
        x = np.array(x)
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        return (x * 255).astype(np.uint8)

    for name, p in zip(["xy", "xz", "yz"], model.den_planes):
        img = norm(np.sum(np.array(p), axis=0))
        imageio.imwrite(f"{dir_out}/density_{name}.png", img)
    print(f"✔ Saved planes → {dir_out}")


def make_render_chunk_for_model(model):
    @partial(jax.jit, static_argnames=("chunk",))
    def render_chunk(rays_o, rays_d, chunk):
        rays_o = jnp.reshape(rays_o, (-1, 3))
        rays_d = jnp.reshape(rays_d, (-1, 3))
        rgb, _, _ = model(rays_o, rays_d, None, jnp.array([1.0, 1.0, 1.0]))
        return rgb

    return render_chunk


def render_360_video(model, dataset, out_dir, n_frames=30, chunk=8192):
    print("\n→ Rendering 360° video frames…")
    H, W = dataset.H, dataset.W
    base_pose = dataset.poses[0]

    xy_radius = np.sqrt(base_pose[0, 3] ** 2 + base_pose[1, 3] ** 2)
    z_elevation = base_pose[2, 3]

    render_chunk = make_render_chunk_for_model(model)

    def get_rays(pose):
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        dirs = np.stack(
            [
                (i - W * 0.5) / dataset.focal,
                -(j - H * 0.5) / dataset.focal,
                -np.ones_like(i),
            ],
            -1,
        )
        rays_d = np.sum(dirs[..., None, :] * pose[:3, :3], -1)
        rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
        return rays_o, rays_d

    for idx, theta in enumerate(
        tqdm(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))
    ):
        cam = np.array(
            [xy_radius * np.cos(theta), xy_radius * np.sin(theta), z_elevation]
        )

        forward = -cam / np.linalg.norm(cam)
        right = np.cross(np.array([0, 0, 1]), forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)

        pose = np.eye(4)
        pose[:3, :3] = np.column_stack([right, up, -forward])
        pose[:3, 3] = cam

        rays_o, rays_d = get_rays(pose)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        rgb_parts = []
        for i in range(0, rays_o.shape[0], chunk):
            ro = rays_o[i : i + chunk]
            rd = rays_d[i : i + chunk]
            rgb_parts.append(render_chunk(ro, rd, chunk))

        img = jnp.concatenate(rgb_parts, axis=0).reshape(H, W, 3)
        img = np.clip(np.array(img), 0, 1)
        imageio.imwrite(f"{out_dir}/frame_{idx:03d}.jpg", (img * 255).astype(np.uint8))

    print(f"✔ Frames saved → {out_dir}")
