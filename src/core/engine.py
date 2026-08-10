import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from geometry.rays import sample_batch_on_device
from core.losses import loss_fn


def restore_step_count(new_opt_state, old_opt_state):
    old_step_count = None
    old_leaves, _ = jax.tree_util.tree_flatten(old_opt_state)
    for leaf in old_leaves:
        if hasattr(leaf, "dtype") and leaf.dtype == jnp.int32 and leaf.shape == ():
            old_step_count = leaf
            break

    if old_step_count is None:
        return new_opt_state

    def _replace_count(leaf):
        if hasattr(leaf, "dtype") and leaf.dtype == jnp.int32 and leaf.shape == ():
            return old_step_count
        return leaf

    return jax.tree_util.tree_map(_replace_count, new_opt_state)


@partial(
    jax.pmap,
    axis_name="devices",
    in_axes=(0, 0, None, 0, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(2, 7, 8, 9, 10, 12, 13),
)
def pmap_train_block(
    params,
    opt_state,
    static,
    rng,
    imgs,
    poses,
    focal,
    H,
    W,
    num_steps,
    is_precrop,
    tv_weight,
    optimizer,
    batch_size_per_device,
):

    def step_fn(carry, step_idx):
        p, opt, current_rng = carry
        current_rng, sample_key, model_key = jax.random.split(current_rng, 3)

        rays_o, rays_d, target_rgb = sample_batch_on_device(
            sample_key, imgs, poses, focal, H, W, is_precrop, batch_size_per_device
        )

        model_local = eqx.combine(p, static)

        def loss_func(m):
            return loss_fn(
                m,
                rays_o,
                rays_d,
                target_rgb,
                model_key,
                tv_weight,
                4e-5,
                jnp.array([1.0, 1.0, 1.0]),
            )

        (loss, mse), grads = eqx.filter_value_and_grad(loss_func, has_aux=True)(
            model_local
        )

        if hasattr(grads, "mlp_render"):
            scaled_mlp = jax.tree_util.tree_map(
                lambda x: x * 0.1 if eqx.is_array(x) else x, grads.mlp_render
            )
            grads = eqx.tree_at(lambda t: t.mlp_render, grads, scaled_mlp)

        grads = jax.lax.pmean(grads, axis_name="devices")
        updates, new_opt = optimizer.update(grads, opt, p)
        new_p = eqx.apply_updates(p, updates)

        return (new_p, new_opt, current_rng), (loss, mse)

    init_carry = (params, opt_state, rng)
    final_carry, metrics = jax.lax.scan(step_fn, init_carry, jnp.arange(num_steps))

    final_params, final_opt, final_rng = final_carry
    return final_params, final_opt, final_rng, metrics[0], metrics[1]
