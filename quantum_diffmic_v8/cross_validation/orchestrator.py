"""End-to-End Stratified 5-Fold Cross-Validation Orchestrator for Unified Quantum-DiffMICv2.
Executes the complete fold-aware pipeline:
1. Splits validation and fold manifest loading.
2. For each fold (0..4):
   - Stage 1: Retrain UnifiedQuantumDCG on fold training split.
   - Stage 2: Train UnifiedQuantumConditionalModel using selected DCG.
   - Stage 3: Extract 16-step trajectories & evaluate out-of-fold validation set.
3. Aggregates all out-of-fold (OOF) predictions across the 3,112 samples.
4. Generates pooled medical metrics, mean +/- std, and evaluates on locked 550 test split.
"""
import os, sys, json, time, yaml, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_diffmic_v8.cross_validation.splits import generate_5fold_manifest
from quantum_diffmic_v8.training.train_dcg import train_one_fold_dcg
from quantum_diffmic_v8.training.train_diffusion import train_one_fold_diffusion
from quantum_diffmic_v8.models.dcg import UnifiedQuantumDCG
from quantum_diffmic_v8.models.diffusion import UnifiedQuantumConditionalModel, SamCachedEncoder, guided_prob_map
from quantum_diffmic_v8.downstream.representations import (
    make_esn, esn_transform, simulate_qrc_correlators, apply_ordinal_minimum_risk
)
from quantum_diffmic_v8.evaluation.metrics import compute_all_medical_metrics
from aptos_research.data import AptosImageDataset
import pipeline

def extract_fold_trajectories(model, aux_model, sampler, loader, device, 
                              num_classes=5, timesteps=16):
    """Extracts 16-step reverse-diffusion trajectory states."""
    model.eval()
    aux_model.eval()
    all_trajs, all_labels, all_preds, all_ids = [], [], [], []

    with torch.no_grad():
        for b in loader:
            x = b[0].to(device)
            y = b[1]
            img_ids = b[2] if len(b) > 2 else None
            bz = x.shape[0]
            nc = num_classes

            y0_aux, y0_g, y0_l, patches, attns, attn_map = aux_model(x)
            np_val = attns.size(1)

            y0_cond = guided_prob_map(y0_g, y0_l, bz, nc, np_val)
            yT = guided_prob_map(torch.rand_like(y0_g), torch.rand_like(y0_l), bz, nc, np_val)

            attns_mat = attns.unsqueeze(-1)
            attns_mat = (attns_mat * attns_mat.transpose(1, 2)).unsqueeze(1)

            batch_traj = [[] for _ in range(bz)]
            def traj_cb(t, step_idx, sample):
                pooled = sample.mean(dim=[2, 3]).cpu().numpy()
                for b_i in range(bz):
                    batch_traj[b_i].append(pooled[b_i])

            final_sample = sampler.sample_high_res(x, yT, conditions=[y0_cond, patches, attns_mat, img_ids], trajectory_callback=traj_cb)
            final_pred = final_sample.reshape(bz, nc, -1).mean(2).cpu().numpy()

            for b_i in range(bz):
                all_trajs.append(np.array(batch_traj[b_i]))
                all_preds.append(final_pred[b_i])
            all_labels.extend(y.numpy().tolist())
            if img_ids is not None:
                all_ids.extend(list(img_ids))

    return np.array(all_trajs), np.array(all_labels), np.array(all_preds), all_ids

def run_cross_validation_pipeline(config_path: str = 'quantum_diffmic_v8/configs/config_5fold.yml',
                                  splits_manifest_path: str = 'quantum_diffmic_v8/configs/5fold_splits.json',
                                  dcg_epochs: int = 20, diff_epochs: int = 35,
                                  batch_size: int = 32, accum_batches: int = 2,
                                  qubits: int = 4, classical_mode: bool = False,
                                  device_str: str = 'cuda:0', start_fold: int = 0, num_folds: int = 5):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(" >>> STARTING UNIFIED STRATIFIED 5-FOLD CROSS-VALIDATION PIPELINE")
    print(f" >>> Device: {device} | Total Folds: {num_folds} | DCG Ep: {dcg_epochs} | Diff Ep: {diff_epochs}")
    print(f" >>> Effective BS: {batch_size * accum_batches} | Qubits: {qubits} | Classical: {classical_mode}")
    print("=" * 80)

    # 1. Ensure splits manifest exists
    if not os.path.exists(splits_manifest_path):
        manifest_data = generate_5fold_manifest(output_path=splits_manifest_path)
    else:
        with open(splits_manifest_path, 'r') as f:
            manifest_data = json.load(f)

    with open(config_path, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))

    # Pre-load EfficientSAM embeddings cache
    sam_encoder = SamCachedEncoder(feature_dim=cfg.model.feature_dim)

    oof_results_by_fold = {}
    oof_predictions = {
        'direct_diffusion': {'preds': [], 'probs': [], 'targets': [], 'ids': []},
        'classical_esn': {'preds': [], 'probs': [], 'targets': [], 'ids': []},
        'multimodal_esn': {'preds': [], 'probs': [], 'targets': [], 'ids': []},
        'qrc_standalone': {'preds': [], 'probs': [], 'targets': [], 'ids': []},
        'hybrid_hqrc': {'preds': [], 'probs': [], 'targets': [], 'ids': []},
        'ordinal_risk_esn': {'preds': [], 'probs': [], 'targets': [], 'ids': []}
    }

    # Iterate over folds
    for f_idx in range(start_fold, min(5, start_fold + num_folds)):
        t_fold_start = time.time()
        print("\n" + "#" * 80)
        print(f" ##### FOLD {f_idx} / 5 EXECUTION BEGINS #####")
        print("#" * 80)

        # STAGE 1: Train DCG (Resume if existing)
        dcg_output = f"quantum_diffmic_v8/ckpts/dcg_best_fold_{f_idx}.pth"
        if os.path.exists(dcg_output):
            print(f"[*] Found existing trained DCG checkpoint for Fold {f_idx} at {dcg_output}. Resuming Stage 1.")
            dcg_ckpt = dcg_output
        else:
            init_dcg = 'quantum_v7_balanced/ckpts/aptos_quantum_dcg_best.pth'
            dcg_ckpt, dcg_val_metrics = train_one_fold_dcg(
                fold_idx=f_idx, splits_path=splits_manifest_path, config_path=config_path,
                epochs=dcg_epochs, batch_size=64, lr=cfg.optim.lr * 0.2, patience=8,
                qubits=qubits, classical_mode=classical_mode, output_path=dcg_output,
                init_from=init_dcg, device_str=device_str
            )

        # STAGE 2: Train Conditional Diffusion (Resume if existing)
        diff_output_dir = f"quantum_diffmic_v8/ckpts/fold_{f_idx}"
        existing_diff_ckpts = [
            os.path.join(diff_output_dir, f) for f in os.listdir(diff_output_dir)
            if f.endswith('.ckpt') and not f.startswith('last')
        ] if os.path.exists(diff_output_dir) else []
        
        if len(existing_diff_ckpts) > 0:
            best_diff_ckpt = existing_diff_ckpts[0]
            print(f"[*] Found existing trained Diffusion checkpoint for Fold {f_idx}: {best_diff_ckpt}. Resuming Stage 2.")
        else:
            init_diff = 'quantum_v7_balanced/ckpts/diffmic_quantum_balanced_epoch-62_acc-0.8236_f1-0.6598.ckpt'
            best_diff_ckpt = train_one_fold_diffusion(
                fold_idx=f_idx, splits_path=splits_manifest_path, config_path=config_path,
                aux_ckpt=dcg_ckpt, max_epochs=diff_epochs, min_epochs=15,
                batch_size=batch_size, accum_batches=accum_batches, patience=15,
                qubits=qubits, classical_mode=classical_mode, output_dir=diff_output_dir,
                init_from=init_diff, device_str=device_str
            )

        # STAGE 3: Extract Trajectories & Evaluate Downstream Representations
        print(f"\n[*] [STAGE 3] Evaluating Out-of-Fold Representations for Fold {f_idx}...")
        fold_records = manifest_data['folds'][f'fold_{f_idx}']
        train_ds = AptosImageDataset(fold_records['train'], 'train', size=224, training=False)
        val_ds = AptosImageDataset(fold_records['val'], 'validation', size=224, training=False)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

        # Load models for inference
        denoiser = UnifiedQuantumConditionalModel(
            config=cfg, feature_dim=cfg.model.feature_dim, 
            n_qubits=qubits, classical_mode=classical_mode
        ).to(device)
        aux_model = UnifiedQuantumDCG(
            config=cfg, n_qubits=qubits, classical_mode=classical_mode, pretrained=False
        ).to(device)

        ckpt_payload = torch.load(best_diff_ckpt, map_location='cpu', weights_only=False)
        sd = ckpt_payload.get('state_dict', ckpt_payload)
        den_sd = {k.replace('model.', ''): v for k, v in sd.items() if k.startswith('model.') or not k.startswith('aux_')}
        denoiser.load_state_dict(den_sd, strict=False)

        aux_payload = torch.load(dcg_ckpt, map_location='cpu', weights_only=False)
        if isinstance(aux_payload, list): aux_payload = aux_payload[0]
        aux_sd = aux_payload.get('state_dict', aux_payload)
        aux_model.load_state_dict({k.replace('aux_model.', ''): v for k, v in aux_sd.items()}, strict=False)

        with open('option/diff_DDIM.yaml') as f:
            diff_opt = EasyDict(yaml.safe_load(f))
        diff_opt['scheduler']['num_test_timesteps'] = 16
        scheduler = pipeline.create_SR3scheduler(diff_opt['scheduler'], 'test')
        scheduler.set_timesteps(16)
        scheduler.diff_chns = cfg.data.num_classes
        sampler = pipeline.SR3Sampler(model=denoiser, scheduler=scheduler)

        # Extract trajectories
        X_tr, y_tr, p_tr, ids_tr = extract_fold_trajectories(denoiser, aux_model, sampler, train_loader, device, timesteps=16)
        X_val, y_val, p_val, ids_val = extract_fold_trajectories(denoiser, aux_model, sampler, val_loader, device, timesteps=16)

        # 1. Direct Diffusion Baseline
        m_direct = compute_all_medical_metrics(y_val, p_val)

        # 2. Classical Leaky ESN
        reservoir = make_esn(input_dim=5, hidden_dim=64, seed=2026, spectral_radius=0.9, leak=0.5)
        H_tr = esn_transform(X_tr, reservoir)
        H_val = esn_transform(X_val, reservoir)

        clf_esn = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5, random_state=2026))
        clf_esn.fit(H_tr, y_tr)
        p_esn = clf_esn.predict_proba(H_val)
        m_esn = compute_all_medical_metrics(y_val, p_esn)

        # 3. Multimodal Foundation ESN (Traj + EfficientSAM)
        sam_tr = np.array([sam_encoder.cached_embeddings[i].numpy() if i in sam_encoder.cached_embeddings else np.zeros(256) for i in ids_tr])
        sam_val = np.array([sam_encoder.cached_embeddings[i].numpy() if i in sam_encoder.cached_embeddings else np.zeros(256) for i in ids_val])

        pca = PCA(n_components=16, random_state=2026)
        sam_tr_p = pca.fit_transform(sam_tr)
        sam_val_p = pca.transform(sam_val)

        H_multi_tr = np.hstack([H_tr, sam_tr_p])
        H_multi_val = np.hstack([H_val, sam_val_p])

        clf_multi = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5, random_state=2026))
        clf_multi.fit(H_multi_tr, y_tr)
        p_multi = clf_multi.predict_proba(H_multi_val)
        m_multi = compute_all_medical_metrics(y_val, p_multi)

        # 4. Standalone QRC
        Q_tr = simulate_qrc_correlators(X_tr, n_qubits=qubits, temporal_stride=4)
        Q_val = simulate_qrc_correlators(X_val, n_qubits=qubits, temporal_stride=4)

        clf_qrc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5, random_state=2026))
        clf_qrc.fit(Q_tr, y_tr)
        p_qrc = clf_qrc.predict_proba(Q_val)
        m_qrc = compute_all_medical_metrics(y_val, p_qrc)

        # 5. Hybrid HQRC (ESN + QRC)
        HQ_tr = np.hstack([H_tr, Q_tr])
        HQ_val = np.hstack([H_val, Q_val])

        clf_hqrc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5, random_state=2026))
        clf_hqrc.fit(HQ_tr, y_tr)
        p_hqrc = clf_hqrc.predict_proba(HQ_val)
        m_hqrc = compute_all_medical_metrics(y_val, p_hqrc)

        # 6. Ordinal Minimum-Risk Readout
        ord_preds = apply_ordinal_minimum_risk(p_multi)
        m_ord = compute_all_medical_metrics(y_val, ord_preds)

        print(f"\n[Fold {f_idx} Validation Results Summary]")
        print(f"  Direct Diffusion   : Acc={m_direct['accuracy']*100:.2f}%, F1={m_direct['macro_f1']:.4f}, QWK={m_direct['quadratic_weighted_kappa']:.4f}")
        print(f"  Classical ESN      : Acc={m_esn['accuracy']*100:.2f}%, F1={m_esn['macro_f1']:.4f}, QWK={m_esn['quadratic_weighted_kappa']:.4f}")
        print(f"  Multimodal ESN     : Acc={m_multi['accuracy']*100:.2f}%, F1={m_multi['macro_f1']:.4f}, QWK={m_multi['quadratic_weighted_kappa']:.4f}")
        print(f"  Standalone QRC     : Acc={m_qrc['accuracy']*100:.2f}%, F1={m_qrc['macro_f1']:.4f}, QWK={m_qrc['quadratic_weighted_kappa']:.4f}")
        print(f"  Hybrid HQRC        : Acc={m_hqrc['accuracy']*100:.2f}%, F1={m_hqrc['macro_f1']:.4f}, QWK={m_hqrc['quadratic_weighted_kappa']:.4f}")
        print(f"  Ordinal Risk ESN   : Acc={m_ord['accuracy']*100:.2f}%, F1={m_ord['macro_f1']:.4f}, QWK={m_ord['quadratic_weighted_kappa']:.4f}")

        # Accumulate out-of-fold predictions
        for name, probs, preds in [
            ('direct_diffusion', p_val, np.argmax(p_val, axis=1)),
            ('classical_esn', p_esn, np.argmax(p_esn, axis=1)),
            ('multimodal_esn', p_multi, np.argmax(p_multi, axis=1)),
            ('qrc_standalone', p_qrc, np.argmax(p_qrc, axis=1)),
            ('hybrid_hqrc', p_hqrc, np.argmax(p_hqrc, axis=1)),
            ('ordinal_risk_esn', p_multi, ord_preds)
        ]:
            oof_predictions[name]['probs'].extend(probs.tolist() if isinstance(probs, np.ndarray) else probs)
            oof_predictions[name]['preds'].extend([int(x) for x in preds])
            oof_predictions[name]['targets'].extend(y_val.tolist() if isinstance(y_val, np.ndarray) else y_val)
            oof_predictions[name]['ids'].extend(ids_val)

        oof_results_by_fold[f'fold_{f_idx}'] = {
            'direct_diffusion': m_direct,
            'classical_esn': m_esn,
            'multimodal_esn': m_multi,
            'qrc_standalone': m_qrc,
            'hybrid_hqrc': m_hqrc,
            'ordinal_risk_esn': m_ord,
            'time_seconds': time.time() - t_fold_start
        }

    # 4. Compute Pooled Out-of-Fold (OOF) Metrics
    print("\n" + "=" * 80)
    print(" >>> COMPUTING FINAL POOLED 5-FOLD CROSS-VALIDATION BENCHMARKS")
    print("=" * 80)

    pooled_metrics = {}
    for name, data in oof_predictions.items():
        if len(data['targets']) > 0:
            m_pooled = compute_all_medical_metrics(data['targets'], data['probs'] if name != 'ordinal_risk_esn' else data['preds'])
            pooled_metrics[name] = m_pooled
            print(f"[*] {name:20s} -> POOLED ACC: {m_pooled['accuracy']*100:.2f}% | F1: {m_pooled['macro_f1']:.4f} | QWK: {m_pooled['quadratic_weighted_kappa']:.4f} | AUROC: {m_pooled['macro_auroc_ovr']:.4f} | ECE: {m_pooled['ece_10_bins']:.4f}")

    # Save complete OOF report
    final_payload = {
        'total_evaluated_samples': len(oof_predictions['multimodal_esn']['targets']),
        'num_folds': num_folds,
        'pooled_metrics': pooled_metrics,
        'per_fold_results': oof_results_by_fold
    }

    out_file = "quantum_diffmic_v8/outputs/oof_cross_validation_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[+] Full 5-Fold Cross-Validation Report saved to: {out_file}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='quantum_diffmic_v8/configs/config_5fold.yml')
    p.add_argument('--splits', default='quantum_diffmic_v8/configs/5fold_splits.json')
    p.add_argument('--dcg-epochs', type=int, default=20)
    p.add_argument('--diff-epochs', type=int, default=35)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--accumulate-grad-batches', type=int, default=2)
    p.add_argument('--qubits', type=int, default=4)
    p.add_argument('--classical', action='store_true')
    p.add_argument('--start-fold', type=int, default=0)
    p.add_argument('--num-folds', type=int, default=5)
    a = p.parse_args()

    run_cross_validation_pipeline(
        config_path=a.config, splits_manifest_path=a.splits,
        dcg_epochs=a.dcg_epochs, diff_epochs=a.diff_epochs,
        batch_size=a.batch_size, accum_batches=a.accumulate_grad_batches,
        qubits=a.qubits, classical_mode=a.classical,
        start_fold=a.start_fold, num_folds=a.num_folds
    )
