# Quantum-Enhanced DiffMICv2 (Quantum-v5) Architecture & Documentation

## Overview
This directory implements the from-scratch redesign of **DiffMICv2** and **Dual Attention-Guided Classification (DCG)** incorporating **MobileNetV3-Small** and **Differentiable Quantum Reservoir / Entanglement Layers (2 or 4 Qubits)**.

---

## 1. Key Architectural Upgrades

### A. Compact & Efficient CNN Backbone: MobileNetV3-Small
- **Parameter Reduction**:
  - Original ResNet18 / ResNet50: 11.7M – 25M parameters.
  - ViT-SAM: ~86M parameters.
  - **MobileNetV3-Small: ~2.01M total parameters (>80% reduction!)**
- **Native Spatial Alignment**:
  - Feature map spatial dimension at 224x224 input is natively **7x7** (576 channels).
  - Perfectly matches the DCG Class Activation Map (CAM) size without interpolation distortion.
- **Inverted Residual & Squeeze-and-Excitation (SE)**:
  - Depthwise separable convolutions provide high-resolution retinal lesion feature extraction with ~4x lower memory bandwidth and training latency.

---

### B. 4-Qubit & 2-Qubit Quantum Layers in Pure PyTorch
- **GPU-Native Differentiable Execution**:
  - Implemented with batched tensor contractions and precomputed CNOT index gather permutations.
  - Executes in **<0.1 ms per batch** on CUDA with zero external dependencies.
- **Quantum Feature Fusion (DCG)**:
  - Takes global pooled representation $h_g \in \mathbb{R}^{576}$ and local lesion context $z_{loc} \in \mathbb{R}^{576}$.
  - Projects to $N=4$ qubit rotation angles $\theta_i = \pi \tanh(W x)$.
  - Applies single-qubit $R_Y(\theta)$ gates and ring CNOT entanglement across all 4 qubits.
  - Measures single-qubit expectations $\langle Z_i \rangle$ (4 observables) and pairwise correlators $\langle Z_i Z_j \rangle$ (6 observables).
  - Generates 10-dimensional quantum representation $q_{fuse}$ for linear classification readout, replacing heavy multi-layer MLP fusion.
- **Quantum Cross-Conditioning Bridge (DiffMICv2 Diffusion)**:
  - Entangles the diffusion noisy label subspace (Qubits 0 & 1) directly with the visual evidence subspace (Qubits 2 & 3).
  - Cross-register CNOT gates $(0 \to 2, 1 \to 3)$ couple generative diffusion dynamics with retinal features.
  - Joint correlators $\langle Z_0 Z_2 \rangle$ and $\langle Z_1 Z_3 \rangle$ modulate the denoising vector via quantum FiLM parameters.

---

## 2. Configuration & Hyperparameter Standards

| Parameter | Previous retrain_v4 | New Quantum-v5 |
| :--- | :--- | :--- |
| **CNN Backbone** | EfficientSAM / ResNet18 | **MobileNetV3-Small** (Pretrained ImageNet) |
| **Total Model Parameters**| ~23.5M (DCG) + ~15M (DiffMIC) | **2.01M (DCG) + 2.17M (DiffMIC)** |
| **Batch Size** | 64 (DCG), 8 w/ Accum 8 (DiffMIC) | **16 (DCG), 16 w/ Accum 2 (DiffMIC)** |
| **Validation Frequency** | Every 5 epochs | **After EVERY Epoch (`check_val_every_n_epoch=1`)** |
| **Early Stopping** | None / fixed 100 epochs | **Patience=8 (DCG), Patience=12 (DiffMIC)** |
| **Sampling Timesteps** | 1000 train, 10 DDIM test | **1000 train, 10 DDIM test** |

---

## 3. Directory Layout

```
quantum_v5/
├── README.md                  <-- This documentation
├── config_quantum.yml         <-- Unified configuration
├── quantum_layers.py          <-- 2Q & 4Q GPU-native quantum circuits
├── quantum_dcg.py             <-- MobileNetV3-Small + Quantum Fusion DCG
├── quantum_diffmic.py         <-- DiffMICv2 + Quantum Cross-Conditioning Bridge
├── pretrain_dcg_quantum.py    <-- DCG pretraining with proper early stopping
├── train_diffmic_quantum.py   <-- DiffMICv2 training with per-epoch validation
├── smoke_test_quantum.py      <-- Complete 6-point preflight validation
├── run_quantum_pipeline.sh    <-- End-to-end execution script
├── ckpts/                     <-- Checkpoints directory
└── logs/                      <-- Training logs & per-epoch CSV metrics
```
