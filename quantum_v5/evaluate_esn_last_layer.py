"""Evaluate Classical Echo State Network (ESN) as the Last Layer Prediction on Diffusion Trajectories.
- Eliminates manual extraction/stages: runs directly after diffusion training.
- Extracts 10-step reverse DDIM trajectories on APTOS 2019.
- Passes temporal trajectory states through Echo State Network (ESN K=10, 64 hidden units).
- Fits standardized linear readout and outputs full test metrics (Acc, F1, Kappa, AUROC, ECE).
"""
import argparse, os, sys, yaml, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_v5.quantum_diffmic import QuantumConditionalModel
from quantum_v5.quantum_dcg import QuantumDCG
import pipeline
from aptos_research.data import load_split_manifest, AptosImageDataset
from aptos_research.models import make_esn, esn_transform, fit_classifier
from aptos_research.metrics import classification_metrics
from utils import cast_label_to_one_hot_and_prototype

def extract_trajectories(model, aux_model, sampler, loader, device, num_classes=5, timesteps=10):
    model.eval()
    aux_model.eval()
    
    all_trajectories = []
    all_labels = []
    all_final_preds = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_batch = batch[0].to(device)
            y_batch = batch[1]
            img_ids = batch[2] if len(batch) > 2 else None
            y_one_hot, _ = cast_label_to_one_hot_and_prototype(y_batch, EasyDict({'data': {'num_classes': num_classes, 'label_min_max': [0.001, 0.999]}}))
            y_one_hot = y_one_hot.to(device)

            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = aux_model(x_batch)
            bz, nc, H, W = attn_map.size()
            bz, np_val = attns.size()

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

            # Capture trajectory at each DDIM step
            batch_traj = [[] for _ in range(bz)]

            def traj_cb(t, step_idx, sample):
                # sample: (B, num_classes, np, np)
                pooled = sample.mean(dim=[2, 3]).cpu().numpy() # (B, num_classes)
                for b_i in range(bz):
                    batch_traj[b_i].append(pooled[b_i])

            final_sample = sampler.sample_high_res(
                x_batch, yT, conditions=[y0_cond, patches, attns_mat, img_ids],
                trajectory_callback=traj_cb
            )
            final_pred = final_sample.reshape(bz, nc, -1).mean(2).cpu().numpy()

            for b_i in range(bz):
                all_trajectories.append(np.array(batch_traj[b_i])) # (T, num_classes)
                all_final_preds.append(final_pred[b_i])
            all_labels.extend(y_batch.numpy().tolist())

    return np.array(all_trajectories), np.array(all_labels), np.array(all_final_preds)


def main():
    p = argparse.ArgumentParser(description="Evaluate Classical ESN on Diffusion Trajectories")
    p.add_argument('--config', default='quantum_v5/config_quantum.yml')
    p.add_argument('--checkpoint', default=None, help="Path to best QuantumDiffMIC checkpoint")
    p.add_argument('--aux-checkpoint', default='quantum_v5/ckpts/aptos_quantum_dcg_best.pth')
    p.add_argument('--output', default='quantum_v5/outputs/final_esn_results.json')
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(a.config) as f:
        cfg = EasyDict(yaml.safe_load(f))

    # Resolve diffusion checkpoint
    ckpt_path = a.checkpoint
    if ckpt_path is None:
        # Search for best checkpoint in quantum_v5/ckpts
        ckpts = list(Path('quantum_v5/ckpts').glob('diffmic_quantum_best_*.ckpt'))
        if ckpts:
            ckpt_path = str(sorted(ckpts, key=lambda x: os.path.getmtime(x))[-1])
        elif os.path.isfile('quantum_v5/ckpts/last.ckpt'):
            ckpt_path = 'quantum_v5/ckpts/last.ckpt'
        else:
            raise FileNotFoundError("No trained Quantum-DiffMIC checkpoint found in quantum_v5/ckpts!")

    print("\n" + "=" * 65)
    print(" >>> RUNNING CLASSICAL ESN LAST-LAYER PREDICTION")
    print(f" >>> Diffusion Checkpoint: {ckpt_path}")
    print(f" >>> DCG Checkpoint: {a.aux_checkpoint}")
    print(f" >>> Reservoir Hidden Dim: {a.hidden_dim} (Spectral Radius: 0.9, Leak: 0.5)")
    print(f" >>> Output JSON: {a.output}")
    print("=" * 65 + "\n")

    # Load dataset
    manifest = load_split_manifest(cfg.data.splits_manifest)
    train_ds = AptosImageDataset(manifest['splits']['train'], 'train', size=224, training=False)
    test_ds = AptosImageDataset(manifest['splits']['test'], 'test', size=224, training=False)

    batch_size = getattr(cfg.testing, 'batch_size', 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load models
    denoiser = QuantumConditionalModel(config=cfg, feature_dim=cfg.model.feature_dim, n_qubits=cfg.model.n_qubits).to(device)
    aux_model = QuantumDCG(config=cfg, n_qubits=cfg.model.n_qubits).to(device)

    # Load denoiser weights
    ckpt_payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt_payload.get('state_dict', ckpt_payload)
    den_state = {k.replace('model.', ''): v for k, v in state.items() if k.startswith('model.') or not k.startswith('aux_')}
    denoiser.load_state_dict(den_state, strict=False)

    # Load aux weights
    aux_payload = torch.load(a.aux_checkpoint, map_location='cpu', weights_only=False)
    if isinstance(aux_payload, list):
        aux_payload = aux_payload[0]
    aux_state = aux_payload.get('state_dict', aux_payload)
    aux_model.load_state_dict({k.replace('aux_model.', ''): v for k, v in aux_state.items()}, strict=False)

    denoiser.eval()
    aux_model.eval()

    # Create DDIM sampler
    with open('option/diff_DDIM.yaml', 'r') as f:
        diff_opt = EasyDict(yaml.safe_load(f))
    scheduler = pipeline.create_SR3scheduler(diff_opt['scheduler'], 'test')
    test_steps = getattr(cfg.diffusion, 'test_timesteps', 10)
    scheduler.set_timesteps(test_steps)
    scheduler.diff_chns = cfg.data.num_classes
    sampler = pipeline.SR3Sampler(model=denoiser, scheduler=scheduler)

    print(f"--> [1/3] Extracting {test_steps}-step DDIM trajectories on Train ({len(train_ds)}) & Test ({len(test_ds)})...")
    t0 = time.time()
    X_train, y_train, preds_train_direct = extract_trajectories(denoiser, aux_model, sampler, train_loader, device, cfg.data.num_classes, test_steps)
    X_test, y_test, preds_test_direct = extract_trajectories(denoiser, aux_model, sampler, test_loader, device, cfg.data.num_classes, test_steps)
    print(f"    Done in {time.time() - t0:.1f}s. Trajectory shapes: Train={X_train.shape}, Test={X_test.shape}")

    print("--> [2/3] Transforming trajectories through Classical Echo State Network (ESN)...")
    reservoir = make_esn(input_dim=cfg.data.num_classes, hidden_dim=a.hidden_dim, seed=a.seed, spectral_radius=0.9, leak=0.5)
    H_train = esn_transform(X_train, reservoir)
    H_test = esn_transform(X_test, reservoir)
    print(f"    Reservoir state feature shape: Train={H_train.shape}, Test={H_test.shape}")

    print("--> [3/3] Optimizing ESN Readout (Grid Search C for Top Acc & F1) and Evaluating on Test Split...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)
    best_c_score = -1.0
    best_c = 1.0
    best_cw = 'balanced'
    
    for c_cand in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        for cw in ['balanced', None]:
            fold_scores = []
            for tr_idx, val_idx in skf.split(H_train, y_train):
                pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=c_cand, class_weight=cw, random_state=a.seed))
                pipe.fit(H_train[tr_idx], y_train[tr_idx])
                val_preds_c = pipe.predict_proba(H_train[val_idx])
                m_c = classification_metrics(y_train[val_idx].tolist(), val_preds_c.tolist())
                # Prioritize Macro F1 score to drive target > 72%
                fold_scores.append(m_c['macro_f1'] * 2.0 + m_c['accuracy'])
            mean_score = np.mean(fold_scores)
            if mean_score > best_c_score:
                best_c_score = mean_score
                best_c = c_cand
                best_cw = cw

    print(f"    [*] Optimal ESN Regularization Selected: C={best_c}, class_weight={best_cw} (CV F1-Weighted Score={best_c_score:.4f})")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=best_c, class_weight=best_cw, random_state=a.seed))
    clf.fit(H_train, y_train)
    esn_probs = clf.predict_proba(H_test)

    # Metrics
    m_direct = classification_metrics(y_test, preds_test_direct.tolist())
    m_esn = classification_metrics(y_test, esn_probs.tolist())

    results = {
        'checkpoint': ckpt_path,
        'aux_checkpoint': a.aux_checkpoint,
        'timesteps': test_steps,
        'esn_config': {
            'hidden_dim': a.hidden_dim,
            'spectral_radius': 0.9,
            'leak': 0.5,
            'seed': a.seed
        },
        'direct_diffusion_test': m_direct,
        'esn_last_layer_test': m_esn
    }

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print(" >>> FINAL BENCHMARK SUMMARY (HELD-OUT TEST SPLIT)")
    print("=" * 65)
    print(f"{'Metric':<28} | {'Direct Diffusion':<16} | {'ESN Last Layer':<16}")
    print("-" * 65)
    print(f"{'Test Accuracy':<28} | {m_direct['accuracy']*100.0:>14.2f}% | {m_esn['accuracy']*100.0:>14.2f}%")
    print(f"{'Macro-F1 Score':<28} | {m_direct['macro_f1']:>15.4f} | {m_esn['macro_f1']:>15.4f}")
    print(f"{'Quadratic Weighted Kappa':<28} | {m_direct['quadratic_weighted_kappa']:>15.4f} | {m_esn['quadratic_weighted_kappa']:>15.4f}")
    print(f"{'Macro AUROC':<28} | {m_direct['macro_auroc_ovr']:>15.4f} | {m_esn['macro_auroc_ovr']:>15.4f}")
    print(f"{'Expected Calibration (ECE)':<28} | {m_direct['ece_10_bins']:>15.4f} | {m_esn['ece_10_bins']:>15.4f}")
    print("=" * 65)
    print(f"Results successfully saved to: {a.output}\n")

if __name__ == '__main__':
    main()
