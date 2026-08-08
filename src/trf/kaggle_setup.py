# setup_env.py
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

    run_cmd(f"{sys.executable} -m pip install --upgrade pip")
    run_cmd(f"{sys.executable} -m pip uninstall -y tensorflow")

    if accelerator == "TPU":
        print("Installing JAX for TPU...")
        run_cmd(f"{sys.executable} -m pip install -q requests")
        run_cmd(
            f'{sys.executable} -m pip install -U -q "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'
        )
        run_cmd(
            f'{sys.executable} -m pip install --upgrade equinox jax jaxlib optax "numpy<2.0.0"'
        )

    elif accelerator == "GPU":
        print("Installing JAX for GPU (CUDA 12)...")
        run_cmd(f'{sys.executable} -m pip install --upgrade "jax[cuda12]"')
        run_cmd(f'{sys.executable} -m pip install "numpy<2.0.0" --force-reinstall')
        run_cmd(f"{sys.executable} -m pip install equinox jaxlib optax")

    else:
        print("No accelerator detected. Installing standard CPU JAX...")
        run_cmd(
            f'{sys.executable} -m pip install --upgrade jax jaxlib equinox optax "numpy<2.0.0"'
        )

    run_cmd(f"{sys.executable} -m pip install imageio matplotlib tqdm")
    print("\n Environment Setup Complete!")


if __name__ == "__main__":
    main()
