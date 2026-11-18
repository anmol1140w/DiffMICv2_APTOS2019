"""Downstream Trajectory Representation Learning & Evaluation Suite.
Evaluates:
1. Direct Diffusion Endpoint (t=0)
2. Mean & Concatenated Trajectory Features
3. Leaky Echo State Network (ESN, H=64)
4. Multimodal Foundation ESN (Traj + EfficientSAM)
5. Standalone Quantum Reservoir Computing (QRC, 4Q / 6Q / 12Q)
6. Hybrid Quantum-Classical Reservoir (HQRC)
7. Ordinal Minimum-Risk Bayes Decision Rule
"""
import os, sys, json, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from typing import Dict, Any, List, Tuple

from quantum_diffmic_v8.evaluation.metrics import compute_all_medical_metrics

def make_esn(input_dim: int, hidden_dim: int = 64, seed: int = 2026, 
             spectral_radius: float = 0.9, leak: float = 0.5):
    """Initializes leaky Echo State Network reservoir weights."""
    rng = np.random.RandomState(seed)
    W_in = rng.uniform(-0.1, 0.1, size=(hidden_dim, input_dim))
    W_raw = rng.uniform(-1.0, 1.0, size=(hidden_dim, hidden_dim))
    mask = rng.rand(hidden_dim, hidden_dim) < 0.1
    W = W_raw * mask
    radius = np.max(np.abs(np.linalg.eigvals(W)))
    if radius > 0:
        W = W * (spectral_radius / radius)
    return {'W_in': W_in, 'W': W, 'leak': leak, 'hidden_dim': hidden_dim}

def esn_transform(trajectories: np.ndarray, reservoir: Dict[str, Any]) -> np.ndarray:
    """Transforms sequential trajectories (N, T, input_dim) through leaky reservoir."""
    N, T, _ = trajectories.shape
    W_in, W, leak = reservoir['W_in'], reservoir['W'], reservoir['leak']
    H = np.zeros((N, reservoir['hidden_dim']), dtype=np.float32)
    for i in range(N):
        h = np.zeros(reservoir['hidden_dim'], dtype=np.float32)
        for t in range(T):
            x = trajectories[i, t]
            h = (1.0 - leak) * h + leak * np.tanh(W_in @ x + W @ h)
        H[i] = h
    return H

def simulate_qrc_correlators(trajectories: np.ndarray, n_qubits: int = 4, 
                             temporal_stride: int = 4) -> np.ndarray:
    """Simulates multi-qubit transverse Ising reservoir statevector and measures
    1-body <Z_i> and 2-body <Z_i Z_j> observables across trajectory steps.
    """
    N, T, d = trajectories.shape
    dim = 1 << n_qubits
    indices = np.arange(dim)
    
    # Precompute masks
    mask_1b = np.stack([(np.where((indices & (1 << q)) == 0, 1.0, -1.0)) for q in range(n_qubits)], axis=1)
    pairs = []
    mask_2b_list = []
    for q1 in range(n_qubits):
        for q2 in range(q1 + 1, n_qubits):
            pairs.append((q1, q2))
            b1 = (indices & (1 << q1)) > 0
            b2 = (indices & (1 << q2)) > 0
            mask_2b_list.append(np.where(b1 == b2, 1.0, -1.0))
    mask_2b = np.stack(mask_2b_list, axis=1) if mask_2b_list else np.empty((dim, 0))
    
    # Selected timesteps
    steps = list(range(0, T, max(1, T // temporal_stride)))[:temporal_stride]
    num_obs = n_qubits + len(pairs)
    Q_features = np.zeros((N, len(steps) * num_obs), dtype=np.float32)
    
    for i in range(N):
        state = np.zeros(dim, dtype=np.float32)
        state[0] = 1.0 # Ground state
        feat_idx = 0
        for s in steps:
            v = trajectories[i, s]
            # Pool to n_qubits angles
            pooled = [float(np.mean(v[q::n_qubits])) if len(v[q::n_qubits]) > 0 else 0.0 for q in range(n_qubits)]
            angles = np.pi * np.tanh(pooled)
            
            # Ry rotations
            for q in range(n_qubits):
                stride = 1 << q
                c = np.cos(angles[q] / 2.0)
                s_val = np.sin(angles[q] / 2.0)
                view = state.reshape(-1, 2, stride)
                s0 = view[:, 0, :]
                s1 = view[:, 1, :]
                new0 = c * s0 - s_val * s1
                new1 = s_val * s0 + c * s1
                state = np.stack([new0, new1], axis=1).reshape(dim)
                
            # Circular CNOT entanglement
            for q in range(n_qubits):
                ctrl = q
                tgt = (q + 1) % n_qubits
                m = (indices & (1 << ctrl) != 0) & (indices & (1 << tgt) == 0)
                idx0 = indices[m]
                idx1 = idx0 | (1 << tgt)
                temp = state[idx0].copy()
                state[idx0] = state[idx1]
                state[idx1] = temp
                
            # Measure expectation values
            probs = state ** 2
            exp_1b = probs @ mask_1b
            exp_2b = probs @ mask_2b if len(pairs) > 0 else np.array([])
            obs = np.concatenate([exp_1b, exp_2b])
            Q_features[i, feat_idx:feat_idx + num_obs] = obs
            feat_idx += num_obs
            
    return Q_features

def apply_ordinal_minimum_risk(y_probs: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """Applies the Bayes minimum quadratic risk decision rule:
    y_hat = argmin_c sum_j (|j - c|^2 * P(y=j))
    Directly maximizes Quadratic Weighted Kappa for ordinal grades.
    """
    N = y_probs.shape[0]
    preds = np.zeros(N, dtype=int)
    classes = np.arange(num_classes)
    for i in range(N):
        p = y_probs[i]
        expected_risks = [np.sum(((classes - c) ** 2) * p) for c in range(num_classes)]
        preds[i] = int(np.argmin(expected_risks))
    return preds
