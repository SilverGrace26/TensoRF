import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def device_put_sharded(shards, devices):
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(lambda *xs: jax.device_put(jnp.stack(xs), sharding), *shards)


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
