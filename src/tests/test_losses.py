import jax.numpy as jnp
from core.losses import compute_tv_loss, l1_on_factors


def test_compute_tv_loss():
    planes = [jnp.ones((2, 16, 16)) for _ in range(3)]
    lines = [jnp.ones((2, 16, 1)) for _ in range(3)]

    tv_loss = compute_tv_loss(planes, lines)
    assert tv_loss == 0.0, f"Expected 0.0 TV on flat tensors, got {tv_loss}"


def test_l1_on_factors():
    planes = [jnp.ones((2, 16, 16))]
    lines = [jnp.ones((2, 16, 1))]

    l1_loss = l1_on_factors(planes, lines)

    # L1 penalty over two tensors of all ones should sum directly to 2.0
    assert jnp.isclose(l1_loss, 2.0), f"Expected 2.0 L1 penalty, got {l1_loss}"
