# tests/test_model.py
import jax
import jax.numpy as jnp
from model.tensorf import TensoRF, upsample_tensoRF


def test_tensorf_initialization_and_forward():
    key = jax.random.PRNGKey(42)
    # Use a tiny grid size to save memory
    model = TensoRF(key, grid_dim=16, n_comp_den=[2, 2, 2], n_comp_app=[4, 4, 4])

    # Mock some ray data
    N_RAYS = 10
    rays_o = jnp.zeros((N_RAYS, 3))
    rays_d = jnp.ones((N_RAYS, 3)) / jnp.sqrt(3)
    bg_color = jnp.array([1.0, 1.0, 1.0])

    # Forward pass[cite: 5]
    rgb, depth, weights = model(rays_o, rays_d, key, bg_color)

    assert rgb.shape == (N_RAYS, 3), (
        f"Expected RGB shape {(N_RAYS, 3)}, got {rgb.shape}"
    )
    assert not jnp.isnan(rgb).any(), "RGB output contains NaNs"


def test_upsampling():
    key = jax.random.PRNGKey(0)
    old_model = TensoRF(key, grid_dim=16)

    # Upsample from 16 to 32[cite: 5]
    new_model = upsample_tensoRF(old_model, new_grid_dim=32, key=key)

    assert new_model.grid_dim == 32
    assert new_model.den_planes[0].shape == (8, 32, 32), (
        "Planes not upsampled correctly"
    )
