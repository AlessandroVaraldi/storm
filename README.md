# STORM

A tiny Transformer model for Human Activity Recognition (HAR) on IMU sensor data, designed for deployment on resource-constrained microcontrollers.

## Overview

STORM trains a small Transformer-based classifier on accelerometer and gyroscope data from multiple public HAR datasets (UCI HAR, MotionSense, PAMAP2). The model is quantized to 8-bit integers and exported as C header files for bare-metal inference with no dynamic memory allocation.

## Repository Structure

- **`storm.py`** — Model definition (`TinyTransformerHAR`): conv stem, positional mixing, multi-head self-attention, MLP blocks, attention pooling, and classifier head. Supports integer-only LayerNorm with LUT-based rsqrt approximation.
- **`train.py`** — Training loop with quantization-aware training (QAT), Sharpness-Aware Minimization (SAM), EMA, focal loss, mixup, and learning rate scheduling.
- **`utils/`**
  - `create_dataset.py` — Downloads and preprocesses UCI HAR, MotionSense, and PAMAP2 into a unified format with a common label set.
  - `export.py` — Exports trained weights to quantized C headers (int8 weights, int32 biases, requantization parameters).
  - `quant_utils.py` — Symmetric int8 quantization, per-channel weight quantization, and mult-shift requantization helpers.
  - `deploy_sim.py` — Python-side simulation of the integer-only inference pipeline.
  - `int_layernorm.py` — Integer LayerNorm implementation with LUT-based rsqrt.
  - `lstm_cnn_models.py` — Baseline models (LSTM, 1D-CNN) for comparison.
  - `experiments.py` — Automated experiment runner and hyperparameter search.
  - `gen_test_vector.py` — Generates deterministic test inputs for golden-value verification.
- **`deployment/`** — Pure C inference engine targeting microcontrollers.
  - `transformer.h` — Top-level inference function orchestrating all modules.
  - `main.c` — Test harness for running inference with exported weights.
  - `modules/` — Individual C modules (conv stem, positional mixing, MHSA, MLP, attention pool, classifier).
  - `kernels/` — Low-level kernels (LayerNorm, linear).
  - `luts/` — Precomputed lookup tables for GELU, sigmoid, exp, and reciprocal.
- **`checkpoints/`** — Saved model weights and training metrics.
- **`datasets/`** — Raw and processed HAR datasets.
