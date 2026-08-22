import jax
import jax.numpy as jnp
from geometry.rays import sample_along_rays, compute_ray_aabb_intersections
from geometry.rendering import compute_volumetric_rendering


def test_compute_ray_aabb_intersections():
    rays_o = jnp.zeros((5, 3))
    rays_d = jnp.array([[1.0, 0.0, 0.0]] * 5)
    bbox_min = jnp.array([-1.0, -1.0, -1.0])
    bbox_max = jnp.array([1.0, 1.0, 1.0])

    near, far, hit_mask = compute_ray_aabb_intersections(
        rays_o, rays_d, bbox_min, bbox_max
    )

    assert near.shape == (5,), "Near bounds shape mismatch"
    assert far.shape == (5,), "Far bounds shape mismatch"
    assert hit_mask.shape == (5,), "Hit mask shape mismatch"
    # Ray originating at origin moving right (+X) should hit boundary at X=1.0
    assert jnp.allclose(near, 0.0)
    assert jnp.allclose(far, 1.0)
    assert jnp.all(hit_mask)


def test_sample_along_rays_dynamic_bounds():
    rays_o = jnp.zeros((5, 3))
    rays_d = jnp.ones((5, 3))
    near = jnp.zeros((5,))
    far = jnp.ones((5,)) * 2.0

    # Test sample distribution along explicitly provided bounds
    pts, z_vals = sample_along_rays(rays_o, rays_d, near, far, n_samples=64, key=None)

    assert pts.shape == (5, 64, 3)
    assert z_vals.shape == (5, 64)


def test_volumetric_rendering():
    n_rays, n_samples = 4, 32
    rgb = jnp.ones((n_rays, n_samples, 3)) * 0.5
    sigma = jnp.ones((n_rays, n_samples)) * 0.1
    z_vals = jnp.broadcast_to(jnp.linspace(0.2, 6.0, n_samples), (n_rays, n_samples))

    step_size = (6.0 - 0.2) / n_samples
    dists = jnp.ones((n_rays, n_samples)) * step_size
    bg_color = jnp.array([1.0, 1.0, 1.0])

    rgb_out, depth_map, weights = compute_volumetric_rendering(
        rgb, sigma, dists, z_vals, bg_color
    )

    assert rgb_out.shape == (n_rays, 3)
    assert depth_map.shape == (n_rays,)
    assert weights.shape == (n_rays, n_samples)
