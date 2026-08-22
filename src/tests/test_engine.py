import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from core.engine import pmap_train_block
from model.tensorf import TensoRF


def device_put_replicated(tree, devices):
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(
        lambda x: (
            jax.device_put(jnp.broadcast_to(x, (len(devices),) + x.shape), sharding)
            if isinstance(x, jax.Array)
            else x
        ),
        tree,
    )


def device_put_sharded(shards, devices):
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(lambda *xs: jax.device_put(jnp.stack(xs), sharding), *shards)


def test_pmap_train_block_execution():
    devices = jax.local_devices()
    n_devices = len(devices)

    key = jax.random.PRNGKey(42)
    model_key, train_key = jax.random.split(key)

    model = TensoRF(model_key, grid_dim=16)

    # 3-Way Partition for Inexact Arrays, Exact Arrays, and Pure Static
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

    num_imgs = 2
    H, W = 16, 16
    batch_size_per_device = 2
    num_steps = 1

    imgs = jnp.ones((num_imgs, H, W, 3), dtype=jnp.float32)
    rays_o = jnp.ones((num_imgs, H, W, 3), dtype=jnp.float32)
    rays_d = jnp.ones((num_imgs, H, W, 3), dtype=jnp.float32)

    new_params, new_opt, new_keys, loss, mse = pmap_train_block(
        params_rep,
        opt_state_rep,
        static_arrays,
        static,
        device_keys,
        imgs,
        rays_o,
        rays_d,
        H,
        W,
        num_steps,
        True,
        0.1,
        optimizer,
        batch_size_per_device,
    )

    assert new_params is not None
    assert not jnp.isnan(loss).any(), "Loss evaluates to NaN"
    assert not jnp.isnan(mse).any(), "MSE evaluates to NaN"
