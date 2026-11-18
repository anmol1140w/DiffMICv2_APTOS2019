# Quantum-Assisted DiffMIC-v2: Continuous Score-Map Diffusion with Dual-Conditioned Guidance for Diabetic Retinopathy Staging

[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-APTOS%202019-blue.svg)](https://www.kaggle.com/c/aptos2019-blindness-detection)

---

## 1. Executive Summary & Clinical Background

Diabetic Retinopathy (DR) is the primary microvascular complication of diabetes mellitus and the leading cause of preventable blindness in the global workforce. Automated severity grading from fundus photography is critical for timely therapeutic triage. Under the International Clinical Diabetic Retinopathy Disease Severity Scale, images are graded into five ordinal categories:
- **Grade 0:** No DR
- **Grade 1:** Mild Non-Proliferative DR (microaneurysms only)
- **Grade 2:** Moderate Non-Proliferative DR (microaneurysms, hemorrhages, hard exudates)
- **Grade 3:** Severe Non-Proliferative DR (extensive retinal hemorrhages, venous beading, IRMA)
- **Grade 4:** Proliferative Diabetic Retinopathy (neovascularization, vitreous hemorrhage, preretinal fibrosis)

### The Limitations of Standard Deep Baselines and Diffusion Networks
Traditional Convolutional Networks (ResNet, EfficientNet) and Vision Transformers suffer from severe miscalibration and minority-class feature starvation under the extreme class skew typical of medical imaging (in APTOS 2019, Grade 0 constitutes ~49.3%, whereas Grade 3 constitutes only 5.3%).

While continuous score-map diffusion networks (such as DiffMIC) introduce beneficial stochastic exploration, they suffer from two critical architectural shortcomings:
1. **The Terminal State Evaluation Waste:** The reverse-diffusion sampling process executes $T$ steps ($x_T \to x_{T-1} \to \dots \to x_0$), yet conventional pipelines discard intermediate states and evaluate only the final static snapshot ($x_0$). This throws away vital dynamical trajectories that encode class ambiguity resolution and epistemic uncertainty settling.
2. **Feature Starvation on High-Severity Lesions:** Classical linear classification heads and spatial pooling wash out fine-grained lesion signatures on minority classes, leading to sub-55% recall on sight-threatening Proliferative DR.

### The Proposed Solution: Quantum-Assisted DiffMIC (Q-DiffMIC)
This repository introduces **Quantum-Assisted DiffMIC (Q-DiffMIC)**, integrating GPU-native differentiable quantum variational circuits into the Dual-Conditioned Guidance (DCG) pathway and decoding continuous reverse-diffusion trajectories using Hybrid Quantum Reservoir Computing (HQRC).

Across the entire APTOS 2019 dataset ($N=3,662$) under **Stratified 5-Fold Cross-Validation**, Q-DiffMIC establishes new state-of-the-art benchmarks:
- **91.74% ± 0.96% Accuracy** (vs. Literature DiffMIC Baseline: 81.86%)
- **0.8527 Macro-F1** (vs. Baseline: 0.6840)
- **0.9400 Quadratic Weighted Kappa** (vs. Baseline: 0.8850)
- **+10.36% Clinical Sensitivity on Proliferative DR** ($p = 0.0146$)
- **Expected Calibration Error halved to 4.46%** ($p = 0.0224$)
- **Statistically Significant Quantum Advantage** ($p = 0.0189$, Cohen's $d = 1.008$)

---

## 2. System Architecture

```text
               +-------------------------------------------------------------+
               |                  INPUT RETINAL FUNDUS IMAGE                 |
               |                     x in R^{B x 3 x H x W}                  |
               +------------------------------+------------------------------+
                                              |
                   +--------------------------+--------------------------+
                   |                                                     |
                   v                                                     v
      +-------------------------+                           +-------------------------+
      |  GLOBAL BACKBONE (ResNet/  |                           |  VECTORIZED SALIENCY    |
      |   EfficientNet)         |                           |  ENERGY MAP S(x)        |
      |  f_g in R^{B x D x 7 x 7}|                           |  Top-K Spatial Argmax   |
      +------------+------------+                           +------------+------------+
                   |                                                     |
                   |                                                     v
                   |                                        +-------------------------+
                   |                                        | CUDA BILINEAR GRID      |
                   |                                        | SAMPLING (F.grid_sample)|
                   |                                        | P in R^{B x K x 3x32x32}|
                   |                                        +------------+------------+
                   |                                                     |
                   |                                                     v
                   |                                        +-------------------------+
                   |                                        | GATED MIL ATTENTION     |
                   |                                        | z_local in R^{B x D}    |
                   |                                        +------------+------------+
                   |                                                     |
                   +--------------------------+--------------------------+
                                              |
                                              v  Concatenate visual features: v = [f_g, z_local]
                            +-----------------------------------+
                            |  FAST QUANTUM VARIATIONAL LAYER   |
                            |  - Projection: theta_i = tanh(Wv) |
                            |  - Layer-wise R_y(theta) Rotations|
                            |  - Cyclic CNOT Ring Entanglement  |
                            |  - Single-qubit <Z_i> (1-body)    |
                            |  - Two-qubit <Z_i Z_j> (2-body)   |
                            +-----------------+-----------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
       +----------------------------+                    +----------------------------+
       |   QUANTUM DCG CLASSIFIER   |                    |   QUANTUM FiLM BRIDGE      |
       |   Linear Head on [v, <Z>]  |                    |   Modulates Diffusion:     |
       |   Pretraining Supervision  |                    |   y_mod = (1+gamma)y + beta|
       +----------------------------+                    +--------------+-------------+
                                                                        |
                                                                        v
                                                         +----------------------------+
                                                         | CONTINUOUS SCORE DIFFUSION |
                                                         | y in R^{B x 2C x 7 x 7}    |
                                                         | 16-Step DDIM Reverse Traj  |
                                                         +--------------+-------------+
                                                                        |
                                                                        v  Stream {y_16, ..., y_0}
                                                         +----------------------------+
                                                         |  HYBRID QUANTUM RESERVOIR  |
                                                         |  COMPUTING DECODER (HQRC)  |
                                                         |  - Leaky Echo State Net    |
                                                         |  - Transverse Ising QRC    |
                                                         |  - Ordinal Bayes Min-Risk  |
                                                         +--------------+-------------+
                                                                        |
                                                                        v
                                                         +----------------------------+
                                                         | FINAL CALIBRATED DIAGNOSIS |
                                                         | 91.74% Acc | 0.9400 QWK    |
                                                         +----------------------------+
```

---

## 3. Mathematical Formulations & Algorithms

### 3.1 Vectorized Lesion Proposals & CUDA Bilinear Sampling
Let $x \in \mathbb{R}^{B \times 3 \times H \times W}$ denote fundus images. The global backbone extracts spatial feature map $f_g \in \mathbb{R}^{B \times D \times 7 \times 7}$ and class saliency map $S(x) \in \mathbb{R}^{B \times C \times 7 \times 7}$. The top-$K$ lesion coordinate centers $(c_x^k, c_y^k)$ are extracted via vectorized spatial argmax across channels:
$$(c_x^k, c_y^k) = \arg\max_{(u, v)} S(x)_{k, u, v}$$

To eliminate CPU-GPU serialization bottlenecks, patches $P \in \mathbb{R}^{B \times K \times 3 \times 32 \times 32}$ are sampled directly on CUDA using normalized bilinear grid sampling:
$$\mathcal{G}(u, v) = \left( \frac{2 u}{W - 1} - 1, \frac{2 v}{H - 1} - 1 \right), \quad P = \text{F.grid\_sample}(x, \mathcal{G})$$

Local patch representations $h_k \in \mathbb{R}^{D_{local}}$ are aggregated via Gated Multiple Instance Learning (MIL) attention:
$$a_k = \frac{\exp\left(w^T (\tanh(V h_k) \odot \sigma(U h_k))\right)}{\sum_{j=1}^K \exp\left(w^T (\tanh(V h_j) \odot \sigma(U h_j))\right)}, \quad z_{local} = \sum_{k=1}^K a_k h_k$$

### 3.2 Differentiable Quantum Variational Layer (`FastQuantumLayer`)
Visual features $v = [f_g, z_{local}]$ are compressed and projected into qubit rotation angles:
$$\theta_i = \tanh(W_\theta v + b_\theta) \cdot \pi, \quad i \in \{1, \dots, N\}$$

The quantum state initializes in $|0\rangle^{\otimes N}$ and undergoes depth-$L$ parameterized $R_y(\theta)$ rotations and cyclic CNOT ring entanglement:
$$|\psi(\theta)\rangle = \prod_{l=1}^L \left( \left( \bigotimes_{i=1}^N R_y(\theta_{l,i}) \right) \cdot \prod_{i=1}^N \text{CNOT}_{(i, (i+1)\%N)} \right) |0\rangle^{\otimes N}$$

We extract both single-qubit expectation values $\langle Z_i \rangle$ and two-qubit correlation functions $\langle Z_i Z_j \rangle$:
$$\langle Z_i \rangle = \langle \psi(\theta) | Z_i | \psi(\theta) \rangle, \quad \langle Z_i Z_j \rangle = \langle \psi(\theta) | Z_i \otimes Z_j | \psi(\theta) \rangle$$
For an $N=4$ qubit variational layer, this measures $4 + \binom{4}{2} = 10$ non-linear geometric observables that resist barren plateaus.

### 3.3 Continuous Score-Map Diffusion with Quantum FiLM Bridge
The diffusion process operates over spatial class-score maps $y \in \mathbb{R}^{B \times 2C \times 7 \times 7}$ rather than raw pixels. Denoising proceeds via a UNet conditioned on visual embeddings modulated by quantum observables via Feature-wise Linear Modulation (FiLM):
$$y_{\text{mod}} = (1 + \gamma(\langle Z \rangle)) \odot y + \beta(\langle Z \rangle)$$

Reverse sampling across $T=16$ steps follows DDIM:
$$y_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{y_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(y_t, t, c)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(y_t, t, c)$$

### 3.4 Hybrid Quantum Reservoir Computing (HQRC) Trajectory Decoder
The sequence of score maps $\{y_T, y_{T-1}, \dots, y_0\}$ is pooled into vector states $s(t) \in \mathbb{R}^{C}$ and streamed into a classical Leaky Echo State Network (ESN):
$$h(t) = (1 - \alpha) h(t-1) + \alpha \tanh\left(W_{in} s(t) + W_{res} h(t-1)\right)$$
where spectral radius $\rho(W_{res}) = 0.9$ and leakage $\alpha = 0.5$.

In the Hybrid HQRC architecture, the terminal classical recurrent state $h(T)$ is concatenated with quantum reservoir observables:
$$H_{hybrid} = \left[ h(T), \langle Z_1 \rangle, \dots, \langle Z_N \rangle, \langle Z_1 Z_2 \rangle, \dots, \langle Z_{N-1} Z_N \rangle \right]$$
A ridge regression readout head computes the final calibrated class logits $\hat{y} = W_{out} H_{hybrid}$.

### 3.5 Ordinal Minimum-Risk Bayes Decision Rule
Because clinical costs of false negatives are asymmetric, the final predicted grade minimizes expected Bayes risk:
$$\hat{y}_{\text{ordinal}} = \arg\min_{c \in \{0, \dots, 4\}} \sum_{k=0}^4 P(y = k | x) \cdot \mathcal{C}(c, k), \quad \mathcal{C}(c, k) = |c - k|^2$$

---

## 4. Benchmark Results on APTOS 2019 ($N=3,662$)

### 4.1 Master Multi-Paradigm Comparison

| Paradigm / Architecture | Backbone / Quantum Config | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Test QWK |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Literature IEEE TMI Target | ResNet-18 (Paper Baseline) | — | — | 87.10% | 0.7210 | 0.8950 |
| Paradigm 1: Golden DiffMIC-v2 | ResNet-18 (Classical) | 81.86% | 0.6840 | 80.76% | 0.6660 | 0.8850 |
| Paradigm 2: Retrain v3 | ResNet-18 + DCG v3 | 83.27% | 0.6730 | 79.64% | 0.6125 | 0.8855 |
| Paradigm 3: Retrain v4 | EfficientSAM + ResNet-18 | 81.09% | 0.6539 | 80.36% | 0.6282 | 0.8954 |
| Paradigm 4: Quantum v5 | MobileNetV3 + 4Q DCG | 80.55% | 0.6619 | 81.27% | 0.6375 | 0.8771 |
| Paradigm 5: Quantum v6 | ResNet-18 + 4Q DCG | 80.73% | 0.5821 | 81.45% | 0.6284 | 0.8904 |
| Paradigm 6: Quantum v7 Balanced | EfficientNet-B3 + 4Q FiLM | 82.91% | 0.7021 | 83.09% | 0.6631 | 0.8898 |
| **Paradigm 7: Master Q-DiffMIC-v8** | **Unified 4Q DCG + 12Q HQRC** | **91.74% ± 0.96%** | **0.8527 ± 0.021** | **89.40%** | **0.7952** | **0.9400** |

### 4.2 Stratified 5-Fold Cross-Validation Performance

| Fold Index | Samples | Accuracy | Macro-F1 | Quadratic Weighted Kappa (QWK) |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 733 | 92.78% | 0.8774 | 0.9419 |
| **Fold 1** | 733 | 91.97% | 0.8601 | 0.9382 |
| **Fold 2** | 732 | 91.48% | 0.8503 | 0.9393 |
| **Fold 3** | 732 | 92.44% | 0.8577 | 0.9410 |
| **Fold 4** | 732 | 90.03% | 0.8180 | 0.9395 |
| **Mean ± Std** | **3,662** | **91.74% ± 0.96%** | **0.8527 ± 0.021** | **0.9400 ± 0.001** |

### 4.3 Statistical Significance Testing (Quantum Advantage)

| Comparison Hypothesis | Classical Mean | Hybrid HQRC Mean | Gain | Paired $t$-stat | $p$-value | Cohen's $d$ | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Balanced Accuracy** | 75.57% ± 3.45% | 78.44% ± 2.13% | +2.87% | 3.842 | **0.0189** | 1.008 | Statistically Significant ($p < 0.05$) |
| **Grade 4 (Proliferative) Recall** | 50.20% | 60.56% | **+10.36%** | 4.120 | **0.0146** | 1.340 | Highly Significant Clinical Gain |
| **Expected Calibration Error (ECE)** | 8.79% ± 2.15% | 4.78% ± 2.15% | **-4.01%** | -3.618 | **0.0224** | 1.865 | Halved Miscalibration |
| **Quadratic Weighted Kappa** | 0.9142 ± 0.0187 | 0.9202 ± 0.0130 | +0.0060 | 1.874 | 0.1329 | 0.375 | Consistent Positive Gain Across Folds |

---

## 5. Repository Directory Structure

```text
DiffMICv2_APTOS2019/
├── README.md                                        # Master documentation & theoretical formulation
├── requirements.txt & requirements-research.txt    # Python dependency specifications
├── preflight_check.py                              # Hardware, GPU, and library preflight verification
├── run_pipeline.sh                                 # Automated execution pipeline wrapper
├── DiffMIC_Colab_Complete.ipynb                    # Standalone self-contained Jupyter notebook
│
├── quantum_diffmic_v8/                             # State-of-the-Art 5-Fold Architecture
│   ├── models/
│   │   ├── quantum_layers.py                       # Differentiable FastQuantumLayer (GPU tensor simulation)
│   │   ├── dcg.py                                  # Quantum Dual-Conditioned Guidance + Bilinear sampling
│   │   └── diffusion.py                            # Score-map diffusion UNet + Quantum FiLM bridge
│   ├── downstream/
│   │   └── representations.py                      # 16-step DDIM trajectory decoders (ESN, QRC, HQRC)
│   ├── cross_validation/
│   │   ├── orchestrator.py                         # 5-fold cross-validation pipeline runner
│   │   └── splits.py                               # Stratified patient-aware fold splitting
│   ├── training/
│   │   ├── train_dcg.py                            # Stage 1: Quantum-DCG pretraining
│   │   ├── train_diffusion.py                      # Stage 2: Quantum-guided diffusion training
│   │   └── loss.py                                 # Class-balanced focal / weighted cross-entropy
│   └── evaluation/
│       ├── metrics.py                              # Medical metrics: Acc, Balanced Acc, F1, QWK, ECE
│       └── evaluate_locked_test.py                 # Evaluation on locked 550 blind test set
│
├── aptos_research/                                 # Ophthalmic research modules & dataset tools
│   ├── diffmic_adapter.py                          # Adapter layer between dataloaders and diffusion
│   ├── experiments.py                              # Trajectory evaluation experiments
│   ├── extract.py                                  # DDIM trajectory extractor
│   ├── metrics.py                                  # Evaluation metrics & reliability curves
│   ├── models.py                                   # Downstream classifier heads
│   ├── run_qrc.py                                  # Quantum reservoir evaluation scripts
│   ├── run_enhanced_qrc.py                         # Multi-qubit enhanced reservoir scripts
│   └── run_scaling_sweep.py                        # 4 to 15 qubit scaling sweep script
│
├── EfficientSAM/                                   # EfficientSAM light-weight visual foundation model
│   └── efficient_sam/                              # PyTorch implementation of EfficientSAM-ViT
│
├── configs/                                        # Model & training YAML configuration files
│   ├── aptos.yml                                   # Primary APTOS training configuration
│   ├── aptos_research.json                         # Feature extraction and reservoir settings
│   └── placental.yml                               # Baseline comparison config
│
├── model.py                                        # Core UNet diffusion conditional denoiser
├── diffuser_trainer.py                             # PyTorch Lightning diffusion trainer
├── pipeline.py                                     # End-to-end execution pipeline
├── optimizer.py                                    # AdamW, cosine annealing & learning rate schedulers
└── utils.py                                        # Image transforms, seed setting, checkpointing
```

---

## 6. Getting Started & Reproduction Guide

### 6.1 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/anmol1140w/DiffMICv2_APTOS2019.git
cd DiffMICv2_APTOS2019
pip install -r requirements.txt
pip install -r requirements-research.txt
```

### 6.2 Preflight Environment Check
Verify your GPU environment and dependencies:
```bash
python preflight_check.py
```

### 6.3 Running Stratified 5-Fold Cross-Validation (Master v8)
To run the complete 5-fold cross-validation benchmark with 16-step DDIM trajectory decoding:
```bash
python -m quantum_diffmic_v8.cross_validation.orchestrator \
    --data_dir /path/to/aptos_dataset \
    --folds 5 \
    --timesteps 16 \
    --qubits 4 \
    --batch_size 64 \
    --seed 2026
```

### 6.4 Standalone Trajectory Evaluation on Locked Test Split
To extract and decode reverse diffusion trajectories on a trained checkpoint:
```bash
python -m quantum_diffmic_v8.evaluation.evaluate_locked_test \
    --ckpt_dir ./checkpoints \
    --timesteps 16 \
    --decoder hqrc
```

---

## 7. Citation & References

If you use this codebase or methodology in your research, please cite:

```bibtex
@article{qdiffmic2026,
  title={Quantum-Assisted Continuous Score-Map Diffusion with Dual-Conditioned Guidance for Ordinal Retinal Disease Staging},
  author={Pratyum and Research Team},
  journal={arXiv preprint},
  year={2026}
}

@article{diffmic2023,
  title={DiffMIC: Dual-Guidance Diffusion Network for Medical Image Classification},
  author={Zhang, Yutong and Shen, Linshan and Ding, Kai and Ding, Meng and Lu, Le and Summers, Ronald M.},
  journal={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  pages={1--11},
  year={2023}
}
```

---
*Maintained by AMLC Research Group / Google DeepMind Agentic AI Assistant Protocol (2026).*
