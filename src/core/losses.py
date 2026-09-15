import jax.numpy as jnp


def l1_on_factors(planes, lines):
    l1 = 0.0
    for p in planes + lines:
        l1 += jnp.mean(jnp.abs(p))
    return l1


def compute_tv_loss(planes, lines):
    tv = 0.0
    for p in planes:
        tv += jnp.mean(jnp.abs(p[:, 1:, :] - p[:, :-1, :])) + jnp.mean(
            jnp.abs(p[:, :, 1:] - p[:, :, :-1])
        )
    for l in lines:
        tv += jnp.mean(jnp.abs(l[:, 1:, :] - l[:, :-1, :]))
    return tv


def loss_fn(model, rays_o, rays_d, target_rgb, key, tv_weight, l1_weight, bg_color):
    pred_rgb, _, weights = model(rays_o, rays_d, key, bg_color)
    mse_loss = jnp.mean((pred_rgb - target_rgb) ** 2)

    tv_den = compute_tv_loss(model.den_planes, model.den_lines)
    tv_app = compute_tv_loss(model.app_planes, model.app_lines)

    all_planes = model.den_planes + model.app_planes
    all_lines = model.den_lines + model.app_lines
    l1_factors = l1_on_factors(all_planes, all_lines)

    total_loss = mse_loss + tv_weight * (tv_den + 0.1 * tv_app) + l1_weight * l1_factors
    return total_loss, mse_loss
