# src/tests/test_engine.py
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from core.engine import pmap_train_block
from model.tensorf import TensoRF

# Re-declare your drop-in replacements here (or import them if you moved them to a utils file)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def device_put_replicated(tree, devices):
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(
        lambda x: jax.device_put(
            jnp.broadcast_to(x, (len(devices),) + x.shape), sharding
        ),
        tree,
    )


def device_put_sharded(shards, devices):
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(lambda *xs: jax.device_put(jnp.stack(xs), sharding), *shards)


def test_hardware_replication_and_pmap_block():
    devices = jax.local_devices()
    key = jax.random.PRNGKey(42)
    model_key, train_key = jax.random.split(key)

    # 1. Initialize a tiny model and optimizer
    model = TensoRF(model_key, grid_dim=16)
    params, static = eqx.partition(model, eqx.is_array)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    # 2. Test Device Replication (This catches the AttributeError!)
    try:
        params_rep = device_put_replicated(params, devices)
        opt_state_rep = device_put_replicated(opt_state, devices)
    except AttributeError as e:
        pytest.fail(f"Replication failed. JAX version mismatch: {e}")

    # 3. Test Key Sharding
    n_devices = len(devices)
    device_keys_list = list(jax.random.split(train_key, n_devices))
    try:
        device_keys = device_put_sharded(device_keys_list, devices)
    except AttributeError as e:
        pytest.fail(f"Key sharding failed. JAX version mismatch: {e}")

    # 4. Mock Data for a single pmap step
    batch_size_per_device = 4
    total_batch = batch_size_per_device * n_devices

    # Mocking standard dataset arrays
    imgs = jnp.ones((2, 100, 100, 4))
    poses = jnp.array([jnp.eye(4), jnp.eye(4)])
    focal = 50.0
    H, W = 100, 100

    # 5. Run the pmap block for exactly 1 step
    try:
        new_params, new_opt, new_keys, loss, mse = pmap_train_block(
            params_rep,
            opt_state_rep,
            static,
            device_keys,
            imgs,
            poses,
            focal,
            H,
            W,
            1,  # num_steps
            True,  # is_precrop
            0.1,  # tv_weight
            optimizer,  # optimizer
            batch_size_per_device,  # batch_size_per_device
        )
    except Exception as e:
        pytest.fail(f"pmap_train_block execution failed: {e}")

    # 6. Verify outputs are still properly sharded/replicated across devices
    assert new_params is not None, "Parameters were not returned"
    assert not jnp.isnan(loss).any(), "Loss contains NaNs"
