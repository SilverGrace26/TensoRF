# TensoRF in JAX / Equinox (TPU Optimized)

This repository contains a high-performance, TPU-optimized implementation of [TensoRF](https://www.google.com/search?q=https://arxiv.org/abs/2203.09517&utm_source=gemini) written in JAX and Equinox. It enables ultra-fast radiance field training by factorizing the 3D scene representation into lightweight tensor components.

**[[Kaggle Notebook Placeholder]]()** | **[[360° Video Render Placeholder]](https://www.google.com/search?q=%23&utm_source=gemini)**

## Results

* **Dataset:** NeRF Synthetic Dataset (`mic` and `lego` scenes)
* **Hardware:** TPU v5e-8
* **Performance:** **30 dB Test PSNR**
* **Training Time:** **~45 minutes**

## Core Approach

Instead of querying a massive MLP for every point in a 3D volume (like standard NeRF), this approach models the scene as a 4D tensor and factorizes it:

* **Vector-Matrix (VM) Decomposition:** The 3D volume is decomposed into a set of 2D Matrix planes (XY, XZ, YZ) and 1D Vector lines (Z, Y, X). Querying a point's features just requires projecting its coordinates onto these planes and lines, interpolating, and multiplying the results.
* **Coarse-to-Fine Upsampling:** Training starts on a low-resolution grid (e.g., 128³) and progressively upsamples the tensor factors (up to 512³) at specific milestones.
* **Shrinking Occupancy Grid:** We maintain a coarse binary occupancy grid (`alpha_mask`). Periodically, we evaluate the scene's density, cull empty space, and shrink the bounding box (`shrink_bbox`). This concentrates grid resolution and compute strictly on occupied space.

## Deep Learning Architecture & Training

* **Architecture:** Two sets of VM factors are learned: one for **Density** (geometry) and one for **Appearance** (features).
* **The MLP:** We use a tiny Equinox MLP (2 hidden layers, 128 width) to decode the appearance features. It takes a 54-dimensional input (27D projected appearance features + 27D sinusoidally encoded view directions) and outputs RGB color.
* **Iterations:** The model trains for **30,000 iterations**.
* **Optimizers & Schedulers:** We use `optax.adam` with decoupled learning rates via `optax.multi_transform`. The tensor grids and the MLP have separate learning rates, both utilizing an Optax **Warmup Exponential Decay** schedule.
* **Loss Function:**
* **MSE:** Standard photometric loss against ground truth pixels.
* **Total Variation (TV) Loss:** Applied to the tensor grids to enforce spatial smoothness.
* **L1 Penalty:** Enforces sparsity in the tensor factors.
* *Why it matters:* Without TV and L1 regularization, grid-based methods are highly prone to high-frequency noise and "floaters" (artifacts in empty space).



## TPU-Specific Optimizations

To fully saturate Google's Tensor Processing Units (TPUs), this codebase implements several hardware-specific optimizations:

* **`bfloat16` Compute:** Grid interpolations and accumulations are downcast to `bfloat16` just-in-time (`get_sigma_feat`) to aggressively leverage TPU Matrix Multiplication Units (MXUs) without destabilizing the optimizer state (which remains in fp32).
* **Data Parallelism (`jax.pmap`):** The training loop uses `jax.pmap` (`pmap_train_block`) to distribute batched ray rendering seamlessly across all 8 TPU cores, synchronizing gradients with `jax.lax.pmean`.
* **Hand-rolled Bilinear Interpolation:** Instead of using `scipy.ndimage.map_coordinates`, we implemented a custom bilinear interpolation function (`bilinear_interp`) using standard JAX array slicing and broadcasting. This is vastly more predictable for the XLA compiler to trace and vectorize on TPU hardware.
* **XLA Padding:** Ray batch chunks are dynamically padded to multiples of 128 (`get_safe_chunk_size` / `_pad_and_shard`). This prevents XLA from repeatedly recompiling the graph when handling edge-case batch sizes at the boundaries of images.

## Acknowledgements

* **Original Paper:** [TensoRF: Tensorial Radiance Fields (Chen et al., ECCV 2022)](https://www.google.com/search?q=https://arxiv.org/abs/2203.09517&utm_source=gemini)
* **Original PyTorch Implementation:** [apchenstu/TensoRF](https://www.google.com/search?q=https://github.com/apchenstu/TensoRF&utm_source=gemini)
