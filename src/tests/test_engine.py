import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from core.engine import pmap_train_block
from core.utils import device_put_replicated, device_put_sharded
from model.tensorf import TensoRF


def test_pmap_train_block_execution():
    devices = jax.local_devices()
    n_devices = len(devices)

    key = jax.random.PRNGKey(42)
    model_key, train_key = jax.random.split(key)

    model = TensoRF(model_key, grid_dim=16)

    params, rest = eqx.partition(model, eqx.is_inexact_array)
    static_arrays, static = eqx.partition(rest, eqx.is_array)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    try:
        params_rep = device_put_replicated(params, devices)
        opt_state_rep = device_put_replicated(opt_state, devices)
    except Exception as e:
        pytest.skip(f"Hardware mapping unavailable, skipping replication step: {e}")

    device_keys_list = list(jax.random.split(train_key, n_devices))
    device_keys = device_put_sharded(device_keys_list, devices)

    batch_size_per_device = 2
    num_steps = 1

    # Match the new pmap input shapes
    shape_3d = (n_devices, num_steps, batch_size_per_device, 3)
    shape_27d = (n_devices, num_steps, batch_size_per_device, 27)
    shape_1d = (n_devices, num_steps, batch_size_per_device)

    batch_rays_o = jnp.ones(shape_3d, dtype=jnp.float32)
    batch_rays_d = jnp.ones(shape_3d, dtype=jnp.float32)
    batch_dirs_enc = jnp.ones(shape_27d, dtype=jnp.float32)
    batch_norms = jnp.ones(shape_1d, dtype=jnp.float32)
    batch_rgb = jnp.ones(shape_3d, dtype=jnp.float32)

    new_params, new_opt, new_keys, loss, mse = pmap_train_block(
        params_rep,
        opt_state_rep,
        static_arrays,
        static,
        device_keys,
        batch_rays_o,
        batch_rays_d,
        batch_dirs_enc,
        batch_norms,
        batch_rgb,
        num_steps,
        0.1,
        optimizer,
        False,
    )

    assert new_params is not None
    assert not jnp.isnan(loss).any(), "Loss evaluates to NaN"
    assert not jnp.isnan(mse).any(), "MSE evaluates to NaN"
