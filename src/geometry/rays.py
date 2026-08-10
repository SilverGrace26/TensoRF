import jax
import jax.numpy as jnp


def sample_batch_on_device(key, imgs, poses, focal, H, W, is_precrop, batch_size):
    k1, k2, k3 = jax.random.split(key, 3)
    N = imgs.shape[0]

    img_ids = jax.random.randint(k1, (batch_size,), 0, N)

    center_crop_size = jnp.minimum(H, W) // 2
    h_start = (H - center_crop_size) // 2
    w_start = (W - center_crop_size) // 2

    min_h = jnp.where(is_precrop, h_start, 0)
    max_h = jnp.where(is_precrop, h_start + center_crop_size, H)
    min_w = jnp.where(is_precrop, w_start, 0)
    max_w = jnp.where(is_precrop, w_start + center_crop_size, W)

    js = jax.random.randint(k2, (batch_size,), min_h, max_h)
    is_ = jax.random.randint(k3, (batch_size,), min_w, max_w)

    dirs = jnp.stack(
        [
            (is_ - W * 0.5) / focal,
            -(js - H * 0.5) / focal,
            -jnp.ones_like(is_, dtype=jnp.float32),
        ],
        axis=-1,
    )

    rays_o = poses[img_ids, :3, 3]
    R = poses[img_ids, :3, :3]
    rays_d = jnp.einsum("...i,...ji->...j", dirs, R)

    selected_px = imgs[img_ids, js, is_, :]
    alpha = selected_px[..., 3:4]
    white_bg = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    gt_rgb = selected_px[..., :3] * alpha + white_bg * (1.0 - alpha)

    return rays_o, rays_d, gt_rgb


def sample_along_rays(rays_o, rays_d, n_samples, key=None):
    near, far = 0.2, 6.0
    z_vals = jnp.linspace(near, far, n_samples)
    z_vals = jnp.broadcast_to(z_vals, (rays_o.shape[0], n_samples))

    if key is not None:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = jnp.concatenate([mids, z_vals[..., -1:]], -1)
        lower = jnp.concatenate([z_vals[..., :1], mids], -1)
        t_rand = jax.random.uniform(key, z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals


def encode_view_directions(directions, num_freqs=4):
    freqs = 2.0 ** jnp.linspace(0, num_freqs - 1, num_freqs)
    dirs_enc = jnp.concatenate(
        [
            jnp.sin(directions[..., None] * freqs),
            jnp.cos(directions[..., None] * freqs),
        ],
        -1,
    )
    dirs_enc = dirs_enc.reshape(directions.shape[0], -1)
    return jnp.concatenate([dirs_enc, directions], axis=-1)
