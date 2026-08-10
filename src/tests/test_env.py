# tests/test_env.py
import os
from kaggle_setup import detect_accelerator


def test_detect_accelerator():
    # This should return "CPU", "GPU", or "TPU" without crashing[cite: 1]
    accelerator = detect_accelerator()
    assert accelerator in ["CPU", "GPU", "TPU"], "Invalid accelerator detected"
