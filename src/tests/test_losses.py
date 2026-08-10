# tests/test_losses.py
import jax.numpy as jnp
from core.losses import compute_tv_loss


def test_compute_tv_loss():
    # Mock planes and lines
    planes = [jnp.ones((2, 16, 16)) for _ in range(3)]
    lines = [jnp.ones((2, 16, 1)) for _ in range(3)]

    # Because the arrays are all ones, TV loss should be exactly 0.0[cite: 8]
    tv_loss = compute_tv_loss(planes, lines)
    assert tv_loss == 0.0, f"Expected 0.0, got {tv_loss}"
