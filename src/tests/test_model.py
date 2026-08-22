import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from model.tensorf import TensoRF, upsample_tensoRF, update_alpha_mask, shrink_bbox


def test_tensorf_initialization_and_forward():
    key = jax.random.PRNGKey(42)
    model = TensoRF(key, grid_dim=16, n_comp_den=[2, 2, 2], n_comp_app=[4, 4, 4])

    N_RAYS = 10
    rays_o = jnp.zeros((N_RAYS, 3))
    rays_d = jnp.ones((N_RAYS, 3)) / jnp.sqrt(3)
    bg_color = jnp.array([1.0, 1.0, 1.0])

    rgb, depth, weights = model(rays_o, rays_d, key, bg_color)

    assert rgb.shape == (N_RAYS, 3)
    assert not jnp.isnan(rgb).any(), "RGB output contains NaNs"


def test_update_alpha_mask():
    key = jax.random.PRNGKey(0)
    model = TensoRF(key, grid_dim=16)

    new_model = update_alpha_mask(model)

    assert new_model.alpha_mask.shape == (128, 128, 128), (
        "Mask resolution must match expected grid"
    )
    assert new_model.alpha_mask.dtype == jnp.bool_, "Mask must be boolean type"


def test_shrink_bbox_spatial_crop():
    key = jax.random.PRNGKey(0)
    model = TensoRF(key, grid_dim=32)

    # Mock a condensed active volume in the center of the grid
    mock_mask = np.zeros((128, 128, 128), dtype=bool)
    mock_mask[32:96, 32:96, 32:96] = True
    model = eqx.tree_at(
        lambda m: m.alpha_mask, model, jax.device_put(jnp.array(mock_mask))
    )

    new_model = shrink_bbox(model)

    # Assert bounding box coordinates tightened
    assert jnp.all(new_model.bbox_min > model.bbox_min), (
        "Min bounding box did not shrink"
    )
    assert jnp.all(new_model.bbox_max < model.bbox_max), (
        "Max bounding box did not shrink"
    )

    # Assert tensor planes were dynamically sliced and are smaller than origin
    assert new_model.den_planes[0].shape[-1] > 0
    assert new_model.den_planes[0].shape[-1] < 32


def test_upsampling():
    key = jax.random.PRNGKey(0)
    old_model = TensoRF(key, grid_dim=16)

    new_model = upsample_tensoRF(old_model, new_grid_dim=32, key=key)

    assert new_model.grid_dim == 32
    assert new_model.den_planes[0].shape == (8, 32, 32)
    assert new_model.den_lines[0].shape == (8, 32, 1)
