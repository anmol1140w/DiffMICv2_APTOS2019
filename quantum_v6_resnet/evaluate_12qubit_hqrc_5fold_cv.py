"""Stratified 5-Fold Cross-Validation: 12-Qubit Quantum Reservoir Computing (QRC) & Hybrid HQRC.
- Evaluates Standalone 12-Qubit QRC, Classical ESN, and Hybrid HQRC (ESN + 12Q QRC) on 16-step DDIM trajectories.
- Uses GPU-vectorized PyTorch statevector simulation (2^12 = 4,096 states, 78 correlators per checkpoint).
- Evaluates across Stratified 5-Fold Cross-Validation on all 3,662 APTOS samples.
- Saves results to 'quantum_v6_resnet/outputs/results_12qubit_hqrc_5fold_cv.json'.
"""
import os, sys, json, time, yaml, argparse
from pathlib import Path
import numpy as np
import torch
from easydict import EasyDict

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aptos_research.models import make_esn, esn_transform
from aptos_research.metrics import classification_metrics

def extract_gpu_12qubit_qrc(states_np, n_qubits=12, temporal_steps=4, device='cuda:0'):
    """GPU-native vectorized 12-qubit statevector simulation across all B samples in parallel.
    Generates 1-body expectations <Z_i> and 2-body correlators <Z_i Z_j> at selected DDIM timesteps.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    B, T, D = states_np.shape
    dim = 1 << n_qubits # 4096

    states = torch.tensor(states_np, dtype=torch.float32, device=device)
    state = torch.zeros(B, dim, dtype=torch.complex64, device=device)
    state[:, 0] = 1.0

    # Precompute CNOT permutations for ring entanglement
    indices = torch.arange(dim, device=device)
    cnot_perms = []
    for q in range(n_qubits):
        ctrl = q
        tgt = (q + 1) % n_qubits
        m = (indices & (1 << ctrl) != 0) & (indices & (1 << tgt) == 0)
        idx0 = indices[m]
        idx1 = idx0 | (1 << tgt)
        perm = indices.clone()
        perm[idx0] = idx1
        perm[idx1] = idx0
        cnot_perms.append(perm)

    # 1-body Pauli <Z_i> masks and 2-body correlator <Z_i Z_j> masks
    mask_1b = torch.stack([torch.where((indices & (1 << q)) == 0, 1.0, -1.0) for q in range(n_qubits)], dim=1)
    pairs = [(q1, q2) for q1 in range(n_qubits) for q2 in range(q1 + 1, n_qubits)]
    mask_2b = []
    for q1, q2 in pairs:
        b1 = (indices & (1 << q1)) > 0
        b2 = (indices & (1 << q2)) > 0
        mask_2b.append(torch.where(b1 == b2, 1.0, -1.0))
    mask_2b = torch.stack(mask_2b, dim=1)
    mask_all = torch.cat([mask_1b, mask_2b], dim=1) # (4096, 78)

    meas_steps = set(np.linspace(T // temporal_steps - 1, T - 1, temporal_steps, dtype=int))
    all_obs = []

    for t_step in range(T):
        vec = states[:, t_step, :] # (B, 5)
        # Replicate 5 coordinates to n_qubits angles
        reps = (n_qubits + D - 1) // D
        angles = torch.pi * torch.tanh(vec.repeat(1, reps)[:, :n_qubits])

        # Vectorized Ry rotations
        for q in range(n_qubits):
            stride = 1 << q
            c = torch.cos(angles[:, q] / 2.0).view(B, 1, 1, 1)
            s = torch.sin(angles[:, q] / 2.0).view(B, 1, 1, 1)
            view = state.view(B, -1, 2, stride)
            a0 = view[:, :, 0:1, :]
            a1 = view[:, :, 1:2, :]
            new0 = c * a0 - s * a1
            new1 = s * a0 + c * a1
            state = torch.cat([new0, new1], dim=2).view(B, dim)

        # CNOT ring entanglement
        for perm in cnot_perms:
            state = state[:, perm]

        if t_step in meas_steps:
            probs = state.abs().pow(2)
            obs = probs @ mask_all # (B, 78)
            all_obs.append(obs)

    q_features = torch.cat(all_obs, dim=1) # (B, 78 * 4 = 312)
    return q_features.cpu().numpy()


def compute_ordinal_risk_predictions(probs):
    C = np.zeros((5, 5))
    for j in range(5):
        for c in range(5):
            C[j, c] = (j - c) ** 2
    expected_risk = probs @ C
    return expected_risk.argmin(axis=1)


def main():
    p = argparse.ArgumentParser(description="12-Qubit HQRC Stratified 5-Fold Cross Validation")
    p.add_argument('--trajectories', default='quantum_v6_resnet/outputs/trajectories_16step_all3662.npz')
    p.add_argument('--sam-cache', default='quantum_v6_resnet/outputs/efficient_sam_aptos_cache.pt')
    p.add_argument('--output', default='quantum_v6_resnet/outputs/results_12qubit_hqrc_5fold_cv.json')
    p.add_argument('--qubits', type=int, default=12)
    p.add_argument('--temporal-steps', type=int, default=4)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()

    print("=" * 75)
    print(f" >>> STARTING {a.qubits}-QUBIT QUANTUM RESERVOIR (HQRC) 5-FOLD CROSS-VALIDATION")
    print(f" >>> Trajectory Cache : {a.trajectories}")
    print(f" >>> Output Payload   : {a.output}")
    print("=" * 75)

    data = np.load(a.trajectories, allow_pickle=True)
    trajectories = data['trajectories']
    labels = data['labels']
    direct_preds = data['direct_preds']
    img_ids = list(data['img_ids'])
    N = len(labels)
    print(f"[*] Loaded trajectories: shape={trajectories.shape}, samples={N}")

    # 1. Classical ESN Extraction
    print("\n--> [1/4] Extracting Classical ESN Features (Hidden Dim=64, Radius=0.9, Leak=0.5)...")
    t0 = time.time()
    reservoir = make_esn(input_dim=5, hidden_dim=64, seed=a.seed, spectral_radius=0.9, leak=0.5)
    H_esn = esn_transform(trajectories, reservoir)
    print(f"    Done in {time.time()-t0:.2f}s. ESN Feature Shape: {H_esn.shape}")

    # 2. 12-Qubit Quantum Reservoir Extraction
    print(f"\n--> [2/4] Simulating {a.qubits}-Qubit Quantum Reservoir on GPU (Hilbert Dim=2^{a.qubits}=4,096)...")
    t0 = time.time()
    H_qrc = extract_gpu_12qubit_qrc(trajectories, n_qubits=a.qubits, temporal_steps=a.temporal_steps)
    print(f"    Done in {time.time()-t0:.2f}s. Quantum Feature Shape: {H_qrc.shape} (78 correlators x 4 checkpoints = 312 features)")

    # 3. Multimodal Foundation Embeddings
    print(f"\n--> [3/4] Loading EfficientSAM Foundation Cache for Multimodal HQRC...")
    sam_dict = torch.load(a.sam_cache, map_location='cpu')
    sam_matrix = np.array([sam_dict[img_id].numpy() for img_id in img_ids])
    print(f"    EfficientSAM Feature Shape: {sam_matrix.shape}")

    # Form Hybrid Feature Matrices
    H_hybrid = np.hstack([H_esn, H_qrc]) # (3662, 69 + 312 = 381)
    print(f"    Hybrid HQRC (ESN + {a.qubits}Q QRC) Feature Shape: {H_hybrid.shape}")

    # 4. Stratified 5-Fold Cross-Validation
    print(f"\n--> [4/4] Executing Stratified 5-Fold Cross-Validation across all {N} samples...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)

    # Containers for out-of-fold (OOF) predictions
    oof_qrc_probs = np.zeros((N, 5))
    oof_esn_probs = np.zeros((N, 5))
    oof_hyb_probs = np.zeros((N, 5))
    oof_hyb_multi_probs = np.zeros((N, 5))
    oof_hyb_risk_preds = np.zeros(N, dtype=int)

    fold_metrics_qrc = []
    fold_metrics_esn = []
    fold_metrics_hyb = []
    fold_metrics_hyb_multi = []
    fold_metrics_hyb_risk = []

    print("\n" + "=" * 75)
    print(" >>> 5-FOLD CROSS-VALIDATION PROGRESS")
    print("=" * 75)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(H_hybrid, labels)):
        f_num = fold_idx + 1
        y_tr, y_val = labels[train_idx], labels[val_idx]

        # 1. Standalone 12Q QRC
        clf_qrc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=5.0, random_state=a.seed))
        clf_qrc.fit(H_qrc[train_idx], y_tr)
        p_qrc = clf_qrc.predict_proba(H_qrc[val_idx])
        oof_qrc_probs[val_idx] = p_qrc
        m_qrc = classification_metrics(y_val.tolist(), p_qrc.tolist())
        fold_metrics_qrc.append(m_qrc)

        # 2. Classical ESN
        clf_esn = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=5.0, random_state=a.seed))
        clf_esn.fit(H_esn[train_idx], y_tr)
        p_esn = clf_esn.predict_proba(H_esn[val_idx])
        oof_esn_probs[val_idx] = p_esn
        m_esn = classification_metrics(y_val.tolist(), p_esn.tolist())
        fold_metrics_esn.append(m_esn)

        # 3. Hybrid HQRC (Classical ESN + 12Q QRC)
        clf_hyb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=5.0, random_state=a.seed))
        clf_hyb.fit(H_hybrid[train_idx], y_tr)
        p_hyb = clf_hyb.predict_proba(H_hybrid[val_idx])
        oof_hyb_probs[val_idx] = p_hyb
        m_hyb = classification_metrics(y_val.tolist(), p_hyb.tolist())
        fold_metrics_hyb.append(m_hyb)

        # 4. Multimodal Hybrid HQRC (ESN + 12Q QRC + PCA EfficientSAM)
        pca = PCA(n_components=16, random_state=a.seed)
        sam_tr_p = pca.fit_transform(sam_matrix[train_idx])
        sam_val_p = pca.transform(sam_matrix[val_idx])

        H_hm_tr = np.hstack([H_hybrid[train_idx], sam_tr_p])
        H_hm_val = np.hstack([H_hybrid[val_idx], sam_val_p])

        clf_hm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=5.0, random_state=a.seed))
        clf_hm.fit(H_hm_tr, y_tr)
        p_hm = clf_hm.predict_proba(H_hm_val)
        oof_hyb_multi_probs[val_idx] = p_hm
        m_hm = classification_metrics(y_val.tolist(), p_hm.tolist())
        fold_metrics_hyb_multi.append(m_hm)

        # 5. Hybrid HQRC + Ordinal Minimum-Risk Readout
        preds_risk = compute_ordinal_risk_predictions(p_hm)
        oof_hyb_risk_preds[val_idx] = preds_risk
        m_risk = classification_metrics(y_val.tolist(), np.eye(5)[preds_risk].tolist())
        fold_metrics_hyb_risk.append(m_risk)

        print(f" Fold {f_num}/5 | QRC-12Q: {m_qrc['accuracy']*100:.2f}% | ESN: {m_esn['accuracy']*100:.2f}% | Hybrid HQRC: {m_hyb['accuracy']*100:.2f}% | Multimodal HQRC: {m_hm['accuracy']*100:.2f}% | HQRC-Risk: {m_risk['accuracy']*100:.2f}% (QWK: {m_risk['quadratic_weighted_kappa']:.4f})")

    # Overall Out-of-Fold (OOF) Benchmark Summary
    print("\n" + "=" * 96)
    print(" >>> FINAL 12-QUBIT HQRC 5-FOLD CV BENCHMARK SUMMARY (ALL 3,662 SAMPLES)")
    print("=" * 96)

    overall_qrc = classification_metrics(labels.tolist(), oof_qrc_probs.tolist())
    overall_esn = classification_metrics(labels.tolist(), oof_esn_probs.tolist())
    overall_hyb = classification_metrics(labels.tolist(), oof_hyb_probs.tolist())
    overall_hm = classification_metrics(labels.tolist(), oof_hyb_multi_probs.tolist())
    overall_risk = classification_metrics(labels.tolist(), np.eye(5)[oof_hyb_risk_preds].tolist())

    models_dict = {
        '1. Standalone Quantum (12-Qubit QRC)': (overall_qrc, fold_metrics_qrc),
        '2. Classical ESN (Trajectory Only)': (overall_esn, fold_metrics_esn),
        '3. Hybrid HQRC (Classical ESN + 12Q QRC)': (overall_hyb, fold_metrics_hyb),
        '4. Multimodal Hybrid HQRC (HQRC + SAM)': (overall_hm, fold_metrics_hyb_multi),
        '5. Multimodal HQRC + Ordinal Minimum-Risk': (overall_risk, fold_metrics_hyb_risk)
    }

    print(f"{'Method / Model':<44} | {'5-Fold Mean Acc':<16} | {'Macro F1':<10} | {'QWK':<10} | {'AUROC':<10} | {'ECE':<8}")
    print("-" * 108)
    for name, (overall, fold_list) in models_dict.items():
        acc_mean = np.mean([f['accuracy'] for f in fold_list]) * 100.0
        acc_std = np.std([f['accuracy'] for f in fold_list]) * 100.0
        f1 = overall['macro_f1']
        qwk = overall['quadratic_weighted_kappa']
        auc = overall.get('macro_auroc_ovr', 0.0)
        ece = overall.get('ece_10_bins', 0.0)
        print(f"{name:<44} | {acc_mean:>5.2f}% ± {acc_std:>4.2f}% | {f1:>10.4f} | {qwk:>10.4f} | {auc:>10.4f} | {ece:>6.4f}")
    print("=" * 108)

    print("\n--- Per-Class F1 Score Comparison ---")
    print(f"{'Grade':<10} | {'Class Name':<18} | {'Count':<8} | {'Classical ESN':<14} | {'12Q Hybrid HQRC':<16} | {'HQRC + Ordinal Risk':<18}")
    print("-" * 92)
    class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    counts = np.bincount(labels)
    for c in range(5):
        f1_e = overall_esn['per_class_f1'][c]
        f1_h = overall_hm['per_class_f1'][c]
        f1_r = overall_risk['per_class_f1'][c]
        print(f"Grade {c:<4} | {class_names[c]:<18} | {counts[c]:<8} | {f1_e:>13.4f} | {f1_h:>15.4f} | {f1_r:>17.4f}")
    print("=" * 92)

    print("\nConfusion Matrix (Multimodal 12-Qubit HQRC + Ordinal Minimum-Risk Readout):")
    cm_risk = np.array(overall_risk['confusion_matrix'])
    print(cm_risk)

    results_payload = {
        'dataset_size': N,
        'cv_folds': 5,
        'qubits': a.qubits,
        'quantum_features_dim': int(H_qrc.shape[1]),
        'models': {
            'standalone_12qubit_qrc': {'overall': overall_qrc, 'per_fold': fold_metrics_qrc},
            'classical_esn': {'overall': overall_esn, 'per_fold': fold_metrics_esn},
            'hybrid_hqrc_12qubits': {'overall': overall_hyb, 'per_fold': fold_metrics_hyb},
            'multimodal_hybrid_hqrc': {'overall': overall_hm, 'per_fold': fold_metrics_hyb_multi},
            'multimodal_hqrc_ordinal_risk': {'overall': overall_risk, 'per_fold': fold_metrics_hyb_risk}
        }
    }

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, 'w') as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[+] Detailed 12-Qubit HQRC 5-fold CV results saved to: {a.output}\n")

if __name__ == '__main__':
    main()
