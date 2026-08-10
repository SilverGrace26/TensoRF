# tests/test_geometry.py
import jax
import jax.numpy as jnp
from geometry.rays import sample_along_rays, encode_view_directions
from geometry.rendering import compute_volumetric_rendering


def test_sample_along_rays():
    rays_o = jnp.zeros((5, 3))
    rays_d = jnp.ones((5, 3))

    # Sample 64 points along 5 rays[cite: 9]
    pts, z_vals = sample_along_rays(rays_o, rays_d, n_samples=64, key=None)

    assert pts.shape == (5, 64, 3)
    assert z_vals.shape == (5, 64)


def test_volumetric_rendering():
    n_rays, n_samples = 4, 32
    rgb = jnp.ones((n_rays, n_samples, 3)) * 0.5
    sigma = jnp.ones((n_rays, n_samples)) * 0.1
    z_vals = jnp.linspace(0.2, 6.0, n_samples)
    z_vals = jnp.broadcast_to(z_vals, (n_rays, n_samples))
    rays_d = jnp.ones((n_rays, 3))
    bg_color = jnp.array([1.0, 1.0, 1.0])

    # Test rendering logic[cite: 10]
    rgb_out, depth_map, weights = compute_volumetric_rendering(
        rgb, sigma, z_vals, rays_d, bg_color
    )

    assert rgb_out.shape == (n_rays, 3)
    assert depth_map.shape == (n_rays,)
