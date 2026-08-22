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
    in_axes=(
        0,
        0,
        None,
        None,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),
    static_broadcasted_argnums=(3, 8, 9, 11, 13, 14, 15),
)
def pmap_train_block(
    params,
    opt_state,
    static_arrays,
    static,
    rng,
    imgs,
    rays_o_all,
    rays_d_all,
    H,
    W,
    start_step,
    num_steps,
    tv_weight,
    optimizer,
    batch_size_per_device,
    verbose,
):
    def step_fn(carry, step_idx):
        p, opt, current_rng = carry
        current_rng, sample_key, model_key = jax.random.split(current_rng, 3)

        global_step = start_step + step_idx
        is_precrop = global_step < 1000

        rays_o, rays_d, target_rgb = sample_batch_on_device(
            sample_key,
            imgs,
            rays_o_all,
            rays_d_all,
            H,
            W,
            is_precrop,
            batch_size_per_device,
        )

        model_local = eqx.combine(p, static_arrays, static)

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

        # --- CONDITIONALLY COMPILED TELEMETRY ---
        if verbose:
            max_den_grad = jnp.max(jnp.abs(grads.den_planes[0]))
            max_app_grad = jnp.max(jnp.abs(grads.app_planes[0]))
            reg_loss = loss - mse
            has_nan = jnp.isnan(loss)

            jax.debug.print(
                "Step {} | MSE: {:.4f} | Reg: {:.4f} | DenGrad: {:.2e} | AppGrad: {:.2e} | BBox Min: {} | NaN: {}",
                global_step,
                mse,
                reg_loss,
                max_den_grad,
                max_app_grad,
                p.bbox_min,
                has_nan,
            )
        # ----------------------------------------

        grads = jax.lax.pmean(grads, axis_name="devices")
        updates, new_opt = optimizer.update(grads, opt, p)
        new_p = eqx.apply_updates(p, updates)

        return (new_p, new_opt, current_rng), (loss, mse)

    init_carry = (params, opt_state, rng)
    final_carry, metrics = jax.lax.scan(step_fn, init_carry, jnp.arange(num_steps))

    final_params, final_opt, final_rng = final_carry
    return final_params, final_opt, final_rng, metrics[0], metrics[1]
