import jax.numpy as jnp


def compute_volumetric_rendering(rgb, sigma, dists, z_vals, bg_color):
    alpha = 1.0 - jnp.exp(-sigma * dists)
    transmittance = jnp.cumprod(1.0 - alpha + 1e-10, axis=-1)
    weights = alpha * jnp.concatenate(
        [jnp.ones((alpha.shape[0], 1)), transmittance[..., :-1]], -1
    )

    rgb_map = jnp.sum(weights[..., None] * rgb, axis=-2)
    depth_map = jnp.sum(weights * z_vals, axis=-1)
    acc_map = jnp.sum(weights, -1)
    rgb_out = rgb_map + (1.0 - acc_map[..., None]) * bg_color

    return rgb_out, depth_map, weights
