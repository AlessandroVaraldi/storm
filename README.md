# STORM

A tiny Transformer model for Human Activity Recognition (HAR) on IMU sensor data, designed for deployment on resource-constrained microcontrollers.

## Overview

STORM trains a small Transformer-based classifier on accelerometer and gyroscope data from multiple public HAR datasets (UCI HAR, MotionSense, PAMAP2). The model is quantized to 8-bit integers and exported as C header files for bare-metal inference with no dynamic memory allocation.

## Architecture

STORM is a ~19K-parameter Transformer that takes 6-channel IMU windows (3-axis accelerometer + 3-axis gyroscope) and classifies them into 8 activity classes. The design is driven by the constraints of integer-only inference on microcontrollers: every component must be expressible in fixed-point arithmetic with no dynamic allocation.

### Signal path

1. **Conv stem** — A single `Conv1d` (kernel 5) projects the 6 raw channels to the embedding dimension `d_model`. SiLU activation.
2. **Depthwise positional mixing** — A depthwise `Conv1d` (kernel 3, groups = d_model) with a residual connection. This replaces learned or sinusoidal positional encodings: because the convolution is causal-padded and per-channel, it injects local ordering information without adding parameters that are hard to quantize.
3. **Transformer blocks** (×`depth`, default 2) — Each block contains multi-head self-attention followed by a two-layer MLP (GELU activation, expansion factor `ffn_mult`). Both sub-layers use pre-norm residual connections. Optionally, attention can be windowed (fixed window size) to cap the O(T²) cost when sequence length grows.
4. **Attention pooling** — Rather than global average pooling or a CLS token, a lightweight learned attention mechanism computes per-timestep scalar weights and produces a single feature vector via weighted sum. This lets the model learn _which_ parts of the window are informative, which matters because activities like "going upstairs" have discriminative phases (foot strike) interleaved with uninformative ones.
5. **Classifier head** — LayerNorm → dropout → linear → 8 logits.

### Integer-only LayerNorm

Standard LayerNorm requires a reciprocal square root, which has no cheap integer equivalent. STORM replaces it with a **LUT-based rsqrt**: a precomputed table of 256 entries, log-uniformly spaced over the expected variance range. During training, an optional simulation mode (`int_layernorm=True`) runs the forward pass through the integer path (Q14 fixed-point affine, LUT lookup) while backpropagating through the exact float computation via straight-through estimation (STE). This closes the train/deploy gap: the model learns to tolerate the quantization noise of its own normalization layers.

### Pluggable operation dispatch

Every nonlinear operation (GELU, SiLU, softmax) is accessed through an `ops` object that can be swapped at runtime via `set_ops()`. The default uses PyTorch native functions; the deployment-simulation variant (`DeploySimOps`) replaces them with LUT-evaluated int8→int8 approximations that mirror the C kernels. This lets the same model class serve for training, float evaluation, and integer-only simulation without code duplication.

## Training

The training loop implements several techniques that interact in non-obvious ways with quantization.

### Quantization-aware training (QAT)

STORM uses two complementary QAT mechanisms:

- **Input QAT (IQAT)** — Inputs are fake-quantized to int8 before entering the model. The quantization scale is derived from a dataset-wide percentile, not from per-batch statistics, so it is fixed at export time. An optional scale jitter during training teaches the model to tolerate sensor calibration drift.
- **Activation QAT** — Forward hooks on every module track running EMA statistics of output ranges. These ranges drive fake quantization of intermediate activations during training. The EMA approach avoids the instability of per-batch min/max while still adapting as the weight distribution shifts during learning.

Both mechanisms use STE: the forward pass sees quantized values, but gradients flow through the rounding operation as if it were the identity.

### Sharpness-Aware Minimization (SAM)

SAM perturbs weights in the direction of steepest loss ascent before computing the actual gradient, steering optimization toward flatter minima. This is particularly important for quantized models: sharp minima are more likely to shift significantly when weights are rounded to int8, whereas flat minima are robust to small perturbations. The perturbation radius controls the trade-off.

### Deployment simulation scheduling

During training, the model periodically switches to full integer-only inference simulation (`deploy_sim` mode) and evaluates on the validation set. This produces a second set of metrics (`val_quant_acc`, `val_quant_macro_f1`) alongside the float metrics, making the quantization gap visible epoch by epoch. Three scheduling modes are supported: `last_epochs` (simulate only near the end), `periodic` (every N batches), and `always`.

### Data augmentation and robustness

Beyond standard augmentations (jitter, scaling, time warping, time masking, mixup, cutmix), the training loop includes **sensor dropout**: with configurable probability, all accelerometer channels or all gyroscope channels are zeroed. This simulates real-world sensor failure modes and forces the model to extract signal from either sensor alone, which is critical for deployment on heterogeneous hardware where one sensor may be absent or miscalibrated.

### Self-distillation

An optional second training phase uses the model's own soft predictions as teacher labels. The intuition: after QAT, the float model's confidence distribution encodes information about inter-class similarities (e.g., "walking" and "upstairs" are more confused than "walking" and "sitting"). Training the quantized student against these soft labels preserves that structure better than hard labels alone.

## Export and quantization

The export pipeline (`utils/export.py`) converts a trained checkpoint into a single C header file containing all weights, biases, scales, and lookup tables.

- **Per-output-channel weight quantization** — Each output channel of every linear/conv layer gets its own int8 scale, maximizing dynamic range utilization. Biases are quantized to int32 to match the accumulator precision.
- **Mult-shift requantization** — To convert int32 accumulator results back to int8 without floating-point division, each channel's real-valued scale factor is decomposed into an integer multiplier M and a right-shift R such that `(x * M) >> R ≈ x * scale`.
- **Activation function LUTs** — GELU and sigmoid are precomputed as int8→int16 tables. Index computation uses integer affine mapping (`idx = (q * alpha + beta) >> rshift`) to avoid floating-point addressing.
- **Preprocessing coefficients** — Input standardization (mean/std from training data) is folded into quantized bias terms, so the C code receives raw int16 sensor readings and produces int8 model inputs with a single multiply-add per channel.

## Deployment pipeline

The `app/` directory is a git submodule containing the C inference engine targeting bare-metal microcontrollers. The pipeline from trained model to running firmware is:

1. **Export** — `utils/export.py` generates a C header with quantized weights, LUTs, and preprocessing constants from a trained checkpoint and calibration data.
2. **Compile** — The C codebase links the generated header and compiles with a standard ARM toolchain (or GCC for desktop validation). All inference code is statically allocated — no `malloc`, no floating-point unit required.
3. **Validate** — Golden-vector tests (`utils/gen_test_vector.py`) produce deterministic input/output pairs using a shared PRNG (xorshift32) that is identical in Python and C, enabling bit-exact numerical verification across the two implementations.
4. **Flash** — The resulting binary is flashed to the target MCU for on-device inference.

The C code mirrors the Python model's structure: separate modules for convolution, multi-head self-attention, MLP, attention pooling, and classification, plus low-level kernels for LayerNorm, linear layers, and LUT evaluation. The deployment simulation mode in training (`DeploySimOps`) uses the same LUT headers as the C code, so any numerical divergence is caught before compilation.

## Repository Structure

- **`storm.py`** — Model definition (`STORM`): conv stem, positional mixing, multi-head self-attention, MLP blocks, attention pooling, and classifier head. Supports integer-only LayerNorm with LUT-based rsqrt approximation.
- **`train.py`** — Training loop with quantization-aware training (QAT), Sharpness-Aware Minimization (SAM), EMA, focal loss, mixup, and learning rate scheduling.
- **`app/`** — Git submodule with the C inference engine for microcontroller deployment.
- **`utils/`**
  - `create_dataset.py` — Downloads and preprocesses UCI HAR, MotionSense, and PAMAP2 into a unified format with a common label set.
  - `export.py` — Exports trained weights to quantized C headers (int8 weights, int32 biases, requantization parameters).
  - `quant_utils.py` — Symmetric int8 quantization, per-channel weight quantization, and mult-shift requantization helpers.
  - `deploy_sim.py` — Python-side simulation of the integer-only inference pipeline.
  - `int_layernorm.py` — Integer LayerNorm implementation with LUT-based rsqrt.
  - `lstm_cnn_models.py` — Baseline models (LSTM, 1D-CNN) for comparison.
  - `experiments.py` — Automated experiment runner and hyperparameter search.
  - `gen_test_vector.py` — Generates deterministic test inputs for golden-value verification.
