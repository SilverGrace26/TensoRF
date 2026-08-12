# kaggle_setup.py
import os
import sys
import subprocess


def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def detect_accelerator():
    tpu_env_keys = ["TPU_NAME", "KAGGLE_TPU_NAME", "TPU_ACCELERATOR_TYPE"]
    if any(k in os.environ for k in tpu_env_keys) or os.path.exists("/dev/accel0"):
        return "TPU"

    try:
        res = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if res.returncode == 0:
            return "GPU"
    except FileNotFoundError:
        pass

    return "CPU"


def main():
    accelerator = detect_accelerator()
    print(f" Detected Accelerator: {accelerator}\n")

    # 1. Install uv
    print("Installing uv for strict dependency resolution...")
    run_cmd(f"{sys.executable} -m pip install uv")

    uv_cmd = f"{sys.executable} -m uv pip install --system"

    # 2. Clear out Kaggle's pre-existing conflicting system packages
    print("Uninstalling conflicting system packages...")
    run_cmd(f"{sys.executable} -m pip uninstall -y tensorflow numpy matplotlib")

    # 3. Fast, single-pass reproducible installations using uv
    if accelerator == "TPU":
        print("Installing JAX for TPU using uv...")
        run_cmd(f"{uv_cmd} requests")
        # Combined into one resolution pass!
        run_cmd(
            f'{uv_cmd} -U "jax[tpu]" equinox optax "numpy<2.0.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'
        )

    elif accelerator == "GPU":
        print("Installing JAX for GPU (CUDA 12) using uv...")
        # Combined into one resolution pass!
        run_cmd(f'{uv_cmd} -U "jax[cuda12]" equinox optax "numpy<2.0.0"')

    else:
        print("No accelerator detected. Installing standard CPU JAX using uv...")
        run_cmd(f'{uv_cmd} -U jax jaxlib equinox optax "numpy<2.0.0"')

    # 4. Install remaining dependencies with strict limits
    print("Resolving and installing standard dependencies...")
    run_cmd(f'{uv_cmd} imageio "matplotlib<3.9.0" tqdm')

    print("\n Environment Setup Complete!")


if __name__ == "__main__":
    main()
