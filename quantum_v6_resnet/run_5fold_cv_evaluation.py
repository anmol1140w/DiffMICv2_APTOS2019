"""Stratified 5-Fold Cross-Validation Evaluation of 16-Step DDIM Trajectories + ESN + Ordinal Risk.
- Extracts 16-step DDIM trajectories across all 3,662 APTOS samples with EfficientSAM foundation cache.
- Saves trajectory cache to 'quantum_v6_resnet/outputs/trajectories_16step_all3662.npz'.
- Evaluates across Stratified 5-Fold Cross Validation:
  1. Direct Diffusion Baseline (T=16)
  2. Standard Classical ESN (T=16, 64 units)
  3. Multimodal ESN (T=16 + Projected EfficientSAM Foundation)
  4. Ordinal Minimum-Risk Readout (Quadratic Penalty Optimization)
  5. Threshold-Optimized Cumulative Ordinal Readout
- Outputs complete out-of-fold summary metrics and per-fold statistics.
"""
import os, sys, json, time, yaml, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_v6_resnet.quantum_diffmic import QuantumConditionalModel
from quantum_v6_resnet.quantum_dcg import QuantumDCG
import pipeline
from aptos_research.data import load_split_manifest, AptosImageDataset
from aptos_research.models import make_esn, esn_transform
from aptos_research.metrics import classification_metrics
from utils import cast_label_to_one_hot_and_prototype

def extract_all_trajectories(model, aux_model, sampler, loader, device, num_classes=5, timesteps=16):
    model.eval()
    aux_model.eval()

    all_trajectories = []
    all_labels = []
    all_final_preds = []
    all_img_ids = []

    print(f"[*] Beginning trajectory extraction ({timesteps} DDIM steps, {len(loader.dataset)} samples)...")
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_batch = batch[0].to(device)
            y_batch = batch[1]
            img_ids = batch[2] if len(batch) > 2 else None

            bz = x_batch.shape[0]
            nc = num_classes

            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = aux_model(x_batch)
            np_val = attns.size(1)

            distance_to_diag = torch.tensor([[abs(i - j) for j in range(np_val)] for i in range(np_val)]).to(device)
            weight_g = 1.0 - distance_to_diag / (np_val - 1)
            weight_l = distance_to_diag / (np_val - 1)
            interpolated = weight_l.unsqueeze(0).unsqueeze(0) * y0_aux_local.unsqueeze(-1).unsqueeze(-1) + \
                           weight_g.unsqueeze(0).unsqueeze(0) * y0_aux_global.unsqueeze(-1).unsqueeze(-1)
            diag_indices = torch.arange(np_val)
            y0_cond = interpolated.clone()
            for i in range(bz):
                for j in range(nc):
                    y0_cond[i, j, diag_indices, diag_indices] = y0_aux_global[i, j]
                    y0_cond[i, j, np_val - 1, 0] = y0_aux_local[i, j]
                    y0_cond[i, j, 0, np_val - 1] = y0_aux_local[i, j]

            yT = y0_cond.clone()
            for i in range(bz):
                for j in range(nc):
                    yT[i, j, diag_indices, diag_indices] = torch.rand_like(y0_aux_global[i, j])
                    yT[i, j, np_val - 1, 0] = torch.rand_like(y0_aux_local[i, j])
                    yT[i, j, 0, np_val - 1] = torch.rand_like(y0_aux_local[i, j])

            attns_mat = attns.unsqueeze(-1)
            attns_mat = (attns_mat * attns_mat.transpose(1, 2)).unsqueeze(1)

            batch_traj = [[] for _ in range(bz)]

            def traj_cb(t, step_idx, sample):
                pooled = sample.mean(dim=[2, 3]).cpu().numpy()
                for b_i in range(bz):
                    batch_traj[b_i].append(pooled[b_i])

            final_sample = sampler.sample_high_res(
                x_batch, yT, conditions=[y0_cond, patches, attns_mat, img_ids],
                trajectory_callback=traj_cb
            )
            final_pred = final_sample.reshape(bz, nc, -1).mean(2).cpu().numpy()

            for b_i in range(bz):
                all_trajectories.append(np.array(batch_traj[b_i]))
                all_final_preds.append(final_pred[b_i])
            all_labels.extend(y_batch.numpy().tolist())
            if img_ids is not None:
                all_img_ids.extend(list(img_ids))

            if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(loader):
                elapsed = time.time() - t0
                pct = (batch_idx + 1) / len(loader) * 100
                print(f"    --> Processed {batch_idx + 1}/{len(loader)} batches ({pct:.1f}%) in {elapsed:.1f}s")

    return np.array(all_trajectories), np.array(all_labels), np.array(all_final_preds), all_img_ids


def compute_ordinal_risk_predictions(probs, cost_matrix=None):
    """Computes minimum expected risk prediction given quadratic ordinal cost matrix.
    cost_matrix[j, c] = (j - c)^2
    """
    if cost_matrix is None:
        C = np.zeros((5, 5))
        for j in range(5):
            for c in range(5):
                C[j, c] = (j - c) ** 2
        cost_matrix = C
    # Expected risk for each candidate class c: R(c) = sum_j C[j, c] * P(j)
    # Shape: (N, 5)
    expected_risk = probs @ cost_matrix
    return expected_risk.argmin(axis=1)


def fit_and_apply_ordinal_thresholds(y_train, probs_train, probs_test):
    """Fits 4 optimal monotonic thresholds on expected severity score to maximize QWK."""
    from sklearn.metrics import cohen_kappa_score
    
    # Continuous expected grade: sum_{k=0}^4 k * P(y=k)
    k_vec = np.arange(5)
    e_train = probs_train @ k_vec
    e_test = probs_test @ k_vec
    
    # Initial thresholds from percentiles
    init_thresholds = [0.5, 1.5, 2.5, 3.5]
    
    def loss_fn(thresholds):
        # Enforce monotonicity
        if not (thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]):
            return 10.0
        preds = np.digitize(e_train, thresholds)
        preds = np.clip(preds, 0, 4)
        qwk = cohen_kappa_score(y_train, preds, weights='quadratic')
        return -qwk

    res = minimize(loss_fn, init_thresholds, method='Nelder-Mead', options={'maxiter': 200})
    best_t = sorted(res.x)
    test_preds = np.digitize(e_test, best_t)
    return np.clip(test_preds, 0, 4)


def main():
    p = argparse.ArgumentParser(description="5-Fold Cross-Validation on 16-Step DDIM Trajectories")
    p.add_argument('--config', default='quantum_v6_resnet/config_resnet.yml')
    p.add_argument('--checkpoint', default='quantum_v6_resnet/ckpts/diffmic_quantum_best_epoch-79_acc-0.8073_f1-0.5821.ckpt')
    p.add_argument('--aux-checkpoint', default='quantum_v6_resnet/ckpts/aptos_quantum_dcg_best.pth')
    p.add_argument('--sam-cache', default='quantum_v6_resnet/outputs/efficient_sam_aptos_cache.pt')
    p.add_argument('--trajectory-cache', default='quantum_v6_resnet/outputs/trajectories_16step_all3662.npz')
    p.add_argument('--output', default='quantum_v6_resnet/outputs/results_5fold_cv_16steps.json')
    p.add_argument('--test-timesteps', type=int, default=16)
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()

    print("=" * 70)
    print(" >>> STARTING STRATIFIED 5-FOLD CROSS-VALIDATION EVALUATION")
    print(f" >>> Trajectory Timesteps : {a.test_timesteps}")
    print(f" >>> Diffusion Checkpoint : {a.checkpoint}")
    print(f" >>> DCG Checkpoint       : {a.aux_checkpoint}")
    print(f" >>> Trajectory Cache     : {a.trajectory_cache}")
    print(f" >>> Results Output       : {a.output}")
    print("=" * 70)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(a.config, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))

    # Check if trajectories are already pre-extracted
    if os.path.isfile(a.trajectory_cache):
        print(f"\n[*] Found existing trajectory cache: {a.trajectory_cache}")
        data = np.load(a.trajectory_cache, allow_pickle=True)
        trajectories = data['trajectories']
        labels = data['labels']
        direct_preds = data['direct_preds']
        img_ids = list(data['img_ids'])
        print(f"    Loaded trajectories: shape={trajectories.shape}, labels={labels.shape}")
    else:
        print("\n[*] Trajectory cache not found. Extracting 16-step trajectories across all 3,662 samples...")
        manifest = load_split_manifest(cfg.data.splits_manifest)
        all_samples = manifest['splits']['train'] + manifest['splits']['validation'] + manifest['splits']['test']
        print(f"    Total samples: {len(all_samples)}")

        dataset = AptosImageDataset(all_samples, 'test', size=224, training=False)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

        # Load models
        denoiser = QuantumConditionalModel(config=cfg, feature_dim=cfg.model.feature_dim, n_qubits=cfg.model.n_qubits).to(device)
        aux_model = QuantumDCG(config=cfg, n_qubits=cfg.model.n_qubits, pretrained=False).to(device)

        ckpt_payload = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        state = ckpt_payload.get('state_dict', ckpt_payload)
        den_state = {k.replace('model.', ''): v for k, v in state.items() if k.startswith('model.') or not k.startswith('aux_')}
        denoiser.load_state_dict(den_state, strict=False)

        aux_payload = torch.load(a.aux_checkpoint, map_location='cpu', weights_only=False)
        if isinstance(aux_payload, list):
            aux_payload = aux_payload[0]
        aux_state = aux_payload.get('state_dict', aux_payload)
        aux_model.load_state_dict({k.replace('aux_model.', ''): v for k, v in aux_state.items()}, strict=False)

        denoiser.eval()
        aux_model.eval()

        with open('option/diff_DDIM.yaml', 'r') as f:
            diff_opt = EasyDict(yaml.safe_load(f))
        diff_opt['scheduler']['num_test_timesteps'] = a.test_timesteps
        scheduler = pipeline.create_SR3scheduler(diff_opt['scheduler'], 'test')
        scheduler.set_timesteps(a.test_timesteps)
        scheduler.diff_chns = cfg.data.num_classes
        sampler = pipeline.SR3Sampler(model=denoiser, scheduler=scheduler)

        trajectories, labels, direct_preds, img_ids = extract_all_trajectories(
            denoiser, aux_model, sampler, loader, device, cfg.data.num_classes, a.test_timesteps
        )

        os.makedirs(os.path.dirname(a.trajectory_cache), exist_ok=True)
        np.savez_compressed(
            a.trajectory_cache,
            trajectories=trajectories,
            labels=labels,
            direct_preds=direct_preds,
            img_ids=np.array(img_ids)
        )
        print(f"[+] Trajectories successfully saved to: {a.trajectory_cache}")

    # Load cached EfficientSAM embeddings for multimodal fusion
    print(f"\n[*] Loading EfficientSAM cache from: {a.sam_cache}")
    sam_dict = torch.load(a.sam_cache, map_location='cpu')
    sam_matrix = np.array([sam_dict[img_id].numpy() for img_id in img_ids])
    print(f"    EfficientSAM feature matrix shape: {sam_matrix.shape}")

    # 1. Transform trajectories through Classical Echo State Network (ESN)
    print("\n--> [1/4] Transforming Trajectories through Classical ESN (Hidden Dim=64, Radius=0.9, Leak=0.5)...")
    reservoir = make_esn(input_dim=cfg.data.num_classes, hidden_dim=a.hidden_dim, seed=a.seed, spectral_radius=0.9, leak=0.5)
    H_esn = esn_transform(trajectories, reservoir)
    print(f"    ESN Feature Matrix: {H_esn.shape}")

    # 2. Stratified 5-Fold Cross Validation Setup
    print("\n--> [2/4] Setting up Stratified 5-Fold Cross Validation across all 3,662 samples...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)

    # Store out-of-fold (OOF) predictions
    N = len(labels)
    oof_diff_preds = direct_preds.argmax(axis=1)
    oof_esn_standard_probs = np.zeros((N, 5))
    oof_esn_multimodal_probs = np.zeros((N, 5))
    oof_esn_risk_preds = np.zeros(N, dtype=int)
    oof_esn_threshold_preds = np.zeros(N, dtype=int)

    fold_metrics_diff = []
    fold_metrics_esn_std = []
    fold_metrics_esn_multi = []
    fold_metrics_esn_risk = []
    fold_metrics_esn_thresh = []

    print("\n" + "=" * 70)
    print(" >>> EXECUTING 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(H_esn, labels)):
        f_num = fold_idx + 1
        y_tr, y_val = labels[train_idx], labels[val_idx]
        H_tr, H_val = H_esn[train_idx], H_esn[val_idx]

        # Model A: Direct Diffusion Baseline on fold
        m_diff_fold = classification_metrics(y_val.tolist(), direct_preds[val_idx].tolist())
        fold_metrics_diff.append(m_diff_fold)

        # Model B: Standard ESN Readout (Trajectory only)
        clf_std = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=5.0, random_state=a.seed))
        clf_std.fit(H_tr, y_tr)
        probs_std_val = clf_std.predict_proba(H_val)
        oof_esn_standard_probs[val_idx] = probs_std_val
        m_esn_std_fold = classification_metrics(y_val.tolist(), probs_std_val.tolist())
        fold_metrics_esn_std.append(m_esn_std_fold)

        # Model C: Multimodal ESN (Trajectory ESN features + PCA Projected SAM features)
        pca = PCA(n_components=16, random_state=a.seed)
        sam_tr_proj = pca.fit_transform(sam_matrix[train_idx])
        sam_val_proj = pca.transform(sam_matrix[val_idx])

        H_multi_tr = np.concatenate([H_tr, sam_tr_proj], axis=1)
        H_multi_val = np.concatenate([H_val, sam_val_proj], axis=1)

        clf_multi = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=5.0, random_state=a.seed))
        clf_multi.fit(H_multi_tr, y_tr)
        probs_multi_val = clf_multi.predict_proba(H_multi_val)
        oof_esn_multimodal_probs[val_idx] = probs_multi_val
        m_esn_multi_fold = classification_metrics(y_val.tolist(), probs_multi_val.tolist())
        fold_metrics_esn_multi.append(m_esn_multi_fold)

        # Model D: Ordinal Minimum-Risk Rule on Multimodal ESN Probabilities
        preds_risk_val = compute_ordinal_risk_predictions(probs_multi_val)
        oof_esn_risk_preds[val_idx] = preds_risk_val
        # Convert predictions to one-hot for metrics calculation
        one_hot_risk = np.eye(5)[preds_risk_val]
        m_risk_fold = classification_metrics(y_val.tolist(), one_hot_risk.tolist())
        fold_metrics_esn_risk.append(m_risk_fold)

        # Model E: Threshold-Optimized Cumulative Ordinal Readout
        probs_multi_tr = clf_multi.predict_proba(H_multi_tr)
        preds_thresh_val = fit_and_apply_ordinal_thresholds(y_tr, probs_multi_tr, probs_multi_val)
        oof_esn_threshold_preds[val_idx] = preds_thresh_val
        one_hot_thresh = np.eye(5)[preds_thresh_val]
        m_thresh_fold = classification_metrics(y_val.tolist(), one_hot_thresh.tolist())
        fold_metrics_esn_thresh.append(m_thresh_fold)

        print(f" Fold {f_num}/5 | Direct: {m_diff_fold['accuracy']*100:.2f}% | ESN-Std: {m_esn_std_fold['accuracy']*100:.2f}% | ESN-Multi: {m_esn_multi_fold['accuracy']*100:.2f}% | Ordinal-Risk: {m_risk_fold['accuracy']*100:.2f}% (QWK: {m_risk_fold['quadratic_weighted_kappa']:.4f})")

    # 3. Overall Out-of-Fold (OOF) Metrics Across All 3,662 Samples
    print("\n" + "=" * 70)
    print(" >>> OVERALL 5-FOLD CV BENCHMARK REPORT (ALL 3,662 SAMPLES)")
    print("=" * 70)

    overall_diff = classification_metrics(labels.tolist(), direct_preds.tolist())
    overall_esn_std = classification_metrics(labels.tolist(), oof_esn_standard_probs.tolist())
    overall_esn_multi = classification_metrics(labels.tolist(), oof_esn_multimodal_probs.tolist())
    overall_esn_risk = classification_metrics(labels.tolist(), np.eye(5)[oof_esn_risk_preds].tolist())
    overall_esn_thresh = classification_metrics(labels.tolist(), np.eye(5)[oof_esn_threshold_preds].tolist())

    models_dict = {
        '1. Direct Diffusion (T=16)': (overall_diff, fold_metrics_diff),
        '2. Classical ESN (Trajectory Only)': (overall_esn_std, fold_metrics_esn_std),
        '3. Multimodal ESN (Traj + EfficientSAM)': (overall_esn_multi, fold_metrics_esn_multi),
        '4. Multimodal ESN + Ordinal Minimum-Risk': (overall_esn_risk, fold_metrics_esn_risk),
        '5. Multimodal ESN + Tuned Cutoffs': (overall_esn_thresh, fold_metrics_esn_thresh)
    }

    print(f"{'Method / Model':<42} | {'5-Fold Mean Acc':<16} | {'Macro F1':<10} | {'QWK':<10} | {'AUROC':<10}")
    print("-" * 96)
    for name, (overall, fold_list) in models_dict.items():
        acc_mean = np.mean([f['accuracy'] for f in fold_list]) * 100.0
        acc_std = np.std([f['accuracy'] for f in fold_list]) * 100.0
        f1 = overall['macro_f1']
        qwk = overall['quadratic_weighted_kappa']
        auc = overall.get('macro_auroc_ovr', 0.0)
        print(f"{name:<42} | {acc_mean:>5.2f}% ± {acc_std:>4.2f}% | {f1:>10.4f} | {qwk:>10.4f} | {auc:>10.4f}")
    print("=" * 96)

    print("\n--- Per-Class F1 Breakdown across Full Dataset (3,662 samples) ---")
    print(f"{'Grade':<10} | {'Class Name':<18} | {'Count':<8} | {'Direct Diff':<12} | {'ESN-Multi':<12} | {'Ordinal-Risk':<12}")
    print("-" * 80)
    class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    counts = np.bincount(labels)
    for c in range(5):
        f1_diff = overall_diff['per_class_f1'][c]
        f1_multi = overall_esn_multi['per_class_f1'][c]
        f1_risk = overall_esn_risk['per_class_f1'][c]
        print(f"Grade {c:<4} | {class_names[c]:<18} | {counts[c]:<8} | {f1_diff:>11.4f} | {f1_multi:>11.4f} | {f1_risk:>11.4f}")
    print("=" * 80)

    print("\nConfusion Matrix (Multimodal ESN + Ordinal Minimum-Risk Readout):")
    cm_risk = np.array(overall_esn_risk['confusion_matrix'])
    print(cm_risk)

    results_payload = {
        'dataset_size': N,
        'cv_folds': 5,
        'timesteps': a.test_timesteps,
        'esn_hidden_dim': a.hidden_dim,
        'models': {
            'direct_diffusion': {'overall': overall_diff, 'per_fold': fold_metrics_diff},
            'esn_standard': {'overall': overall_esn_std, 'per_fold': fold_metrics_esn_std},
            'esn_multimodal': {'overall': overall_esn_multi, 'per_fold': fold_metrics_esn_multi},
            'esn_multimodal_ordinal_risk': {'overall': overall_esn_risk, 'per_fold': fold_metrics_esn_risk},
            'esn_multimodal_tuned_thresholds': {'overall': overall_esn_thresh, 'per_fold': fold_metrics_esn_thresh}
        }
    }

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, 'w') as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[+] Detailed 5-fold cross validation payload saved to: {a.output}\n")

if __name__ == '__main__':
    main()
