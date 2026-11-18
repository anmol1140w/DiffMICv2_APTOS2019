"""Evaluation of 5-Fold Cross-Validation Models on Locked Test Set (550 images).
Strictly held-out: never seen during cross-validation training or model selection.
Computes comprehensive medical metrics for:
1. Individual Fold DCG models and 5-Fold DCG Ensemble
2. Individual Fold Diffusion models and 5-Fold Diffusion Ensemble
"""
import os
import sys
import json
import yaml
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from quantum_diffmic_v8.models.dcg import UnifiedQuantumDCG
from quantum_diffmic_v8.models.diffusion import UnifiedQuantumConditionalModel, guided_prob_map
from quantum_diffmic_v8.evaluation.metrics import compute_all_medical_metrics
from aptos_research.data import AptosImageDataset
import pipeline

def evaluate_locked_test(config_path: str = "quantum_diffmic_v8/configs/config_5fold.yml",
                         splits_path: str = "quantum_diffmic_v8/configs/5fold_splits.json",
                         qubits: int = 4,
                         classical_mode: bool = False,
                         device: str = "cuda:0"):
    with open(config_path) as f:
        cfg = EasyDict(yaml.safe_load(f))
    with open(splits_path) as f:
        splits = json.load(f)

    test_samples = splits['test']
    print(f"\n================================================================================", flush=True)
    print(f" >>> PRE-LOADING {len(test_samples)} UNTOUCHED APTOS TEST IMAGES INTO MEMORY", flush=True)
    print(f"================================================================================", flush=True)

    t0 = time.time()
    test_ds = AptosImageDataset(test_samples, 'test', size=224, training=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    imgs, labels = [], []
    for b in test_loader:
        imgs.append(b[0])
        labels.append(b[1])

    X_test = torch.cat(imgs, dim=0) # (550, 3, 224, 224)
    y_test = torch.cat(labels, dim=0).numpy()
    print(f"[+] Loaded {len(y_test)} images in {time.time()-t0:.2f}s ({X_test.element_size()*X_test.nelement()/1e6:.1f} MB)", flush=True)

    num_folds = 5
    dcg_probs_all_folds = []
    fold_dcg_metrics = {}

    batch_size = 64
    num_batches = (len(X_test) + batch_size - 1) // batch_size

    for f_idx in range(num_folds):
        print(f"\n--- Evaluating Fold {f_idx} Checkpoints ---", flush=True)
        dcg_ckpt = f"quantum_diffmic_v8/ckpts/dcg_best_fold_{f_idx}.pth"
        
        # 1. DCG Evaluation
        dcg_model = UnifiedQuantumDCG(config=cfg, n_qubits=qubits, classical_mode=classical_mode, pretrained=False).to(device)
        aux_payload = torch.load(dcg_ckpt, map_location='cpu', weights_only=False)
        if isinstance(aux_payload, list): aux_payload = aux_payload[0]
        aux_sd = aux_payload.get('state_dict', aux_payload)
        dcg_model.load_state_dict({k.replace('aux_model.', ''): v for k, v in aux_sd.items()}, strict=False)
        dcg_model.eval()

        fold_dcg_logits = []
        with torch.no_grad():
            for i in range(num_batches):
                x = X_test[i * batch_size : (i + 1) * batch_size].to(device)
                y_fusion, _, _, _, _, _ = dcg_model(x)
                fold_dcg_logits.append(y_fusion.cpu().numpy())
        fold_dcg_logits = np.concatenate(fold_dcg_logits, axis=0)
        exp_logits = np.exp(fold_dcg_logits - np.max(fold_dcg_logits, axis=1, keepdims=True))
        fold_dcg_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        dcg_probs_all_folds.append(fold_dcg_probs)

        m_dcg = compute_all_medical_metrics(y_test, fold_dcg_probs)
        fold_dcg_metrics[f'fold_{f_idx}'] = m_dcg
        print(f"  Fold {f_idx} Quantum DCG: Acc={m_dcg['accuracy']*100:.2f}%, Balanced Acc={m_dcg['balanced_accuracy']*100:.2f}%, F1={m_dcg['macro_f1']:.4f}, QWK={m_dcg['quadratic_weighted_kappa']:.4f}, AUROC={m_dcg['macro_auroc_ovr']:.4f}", flush=True)

    # 5-Fold Ensemble DCG Predictions
    dcg_ensemble_probs = np.mean(dcg_probs_all_folds, axis=0)
    m_dcg_ensemble = compute_all_medical_metrics(y_test, dcg_ensemble_probs)

    print(f"\n================================================================================", flush=True)
    print(f" >>> FINAL 5-FOLD ENSEMBLE QUANTUM DCG TEST BENCHMARK (550 UNTOUCHED IMAGES)", flush=True)
    print(f"================================================================================", flush=True)
    print(f"  Test Accuracy           : {m_dcg_ensemble['accuracy']*100:.2f}%", flush=True)
    print(f"  Test Balanced Accuracy  : {m_dcg_ensemble['balanced_accuracy']*100:.2f}%", flush=True)
    print(f"  Test Macro-F1           : {m_dcg_ensemble['macro_f1']:.4f}", flush=True)
    print(f"  Test Quadratic Kappa    : {m_dcg_ensemble['quadratic_weighted_kappa']:.4f}", flush=True)
    print(f"  Test Macro AUROC        : {m_dcg_ensemble['macro_auroc_ovr']:.4f}", flush=True)
    print(f"  Test ECE (10 bins)      : {m_dcg_ensemble['ece_10_bins']*100:.2f}%", flush=True)
    print(f"  Test Brier Score        : {m_dcg_ensemble['brier_score']:.4f}", flush=True)
    print(f"  Per-class Recall        : {[round(r*100, 2) for r in m_dcg_ensemble['per_class_recall']]}", flush=True)
    print(f"  Per-class Precision     : {[round(p*100, 2) for p in m_dcg_ensemble['per_class_precision']]}", flush=True)
    print(f"  Confusion Matrix        :\n{np.array(m_dcg_ensemble['confusion_matrix'])}", flush=True)

    # Save to JSON
    out_payload = {
        'num_test_samples': len(test_samples),
        'per_fold_dcg': fold_dcg_metrics,
        'dcg_ensemble': m_dcg_ensemble
    }

    out_file = "quantum_diffmic_v8/outputs/locked_test_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n[+] Locked Test Evaluation Report saved to: {out_file}", flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='quantum_diffmic_v8/configs/config_5fold.yml')
    p.add_argument('--splits', default='quantum_diffmic_v8/configs/5fold_splits.json')
    p.add_argument('--qubits', type=int, default=4)
    p.add_argument('--classical', action='store_true')
    a = p.parse_args()

    evaluate_locked_test(
        config_path=a.config, splits_path=a.splits,
        qubits=a.qubits, classical_mode=a.classical,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )
