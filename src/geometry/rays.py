import jax
import jax.numpy as jnp


def sample_batch_on_device(
    key, imgs, rays_o_all, rays_d_all, H, W, is_precrop, batch_size
):
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

    rays_o = rays_o_all[img_ids, js, is_, :]
    rays_d = rays_d_all[img_ids, js, is_, :]
    gt_rgb = imgs[img_ids, js, is_, :]

    return rays_o, rays_d, gt_rgb


def compute_ray_aabb_intersections(rays_o, rays_d, bbox_min, bbox_max):
    # Safely invert ray direction to avoid divide by zero errors
    inv_d = jnp.where(jnp.abs(rays_d) < 1e-6, 1e-6 * jnp.sign(rays_d + 1e-9), rays_d)
    inv_d = 1.0 / inv_d

    t0 = (bbox_min - rays_o) * inv_d
    t1 = (bbox_max - rays_o) * inv_d

    t_min = jnp.minimum(t0, t1)
    t_max = jnp.maximum(t0, t1)

    near = jnp.max(t_min, axis=-1)
    far = jnp.min(t_max, axis=-1)

    hit_mask = (near < far) & (far > 0)
    near = jnp.maximum(near, 0.0)
    return near, far, hit_mask


def sample_along_rays(rays_o, rays_d, near, far, n_samples, key=None):
    t_vals = jnp.linspace(0.0, 1.0, n_samples)

    # Broadcast limits appropriately
    near = near[..., None]
    far = far[..., None]
    z_vals = near + (far - near) * t_vals

    if key is not None:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = jnp.concatenate([mids, z_vals[..., -1:]], -1)
        lower = jnp.concatenate([z_vals[..., :1], mids], -1)
        t_rand = jax.random.uniform(key, z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
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
