"""Evaluate Trained Quantum-DiffMICv2 on Clean 550 Held-Out Test Set.
- Extracts 16-step DDIM trajectories on 550 test images.
- Evaluates Direct Diffusion, Multimodal ESN (Traj + EfficientSAM), and Standalone DCG.
- Saves clean test benchmark results to 'quantum_v7_balanced/outputs/clean_test_results.json'.
"""
import argparse, os, sys, yaml, json, time
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_v7_balanced.quantum_diffmic import QuantumConditionalModel
from quantum_v7_balanced.quantum_dcg import QuantumDCG
import pipeline
from aptos_research.data import load_split_manifest, AptosImageDataset
from aptos_research.models import make_esn, esn_transform
from aptos_research.metrics import classification_metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='quantum_v7_balanced/config_balanced.yml')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--aux-checkpoint', default='quantum_v7_balanced/ckpts/aptos_quantum_dcg_best.pth')
    p.add_argument('--output', default='quantum_v7_balanced/outputs/clean_test_results.json')
    p.add_argument('--timesteps', type=int, default=16)
    a = p.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(a.config) as f:
        cfg = EasyDict(yaml.safe_load(f))

    ckpt_path = a.checkpoint
    if ckpt_path is None:
        ckpts = list(Path('quantum_v7_balanced/ckpts').glob('diffmic_quantum_balanced_epoch-*.ckpt'))
        if ckpts:
            ckpt_path = str(sorted(ckpts, key=lambda x: os.path.getmtime(x))[-1])
        else:
            ckpt_path = 'quantum_v7_balanced/ckpts/last.ckpt'

    print("=" * 65)
    print(f" >>> EVALUATING ON CLEAN HELD-OUT TEST SPLIT (550 SAMPLES)")
    print(f" >>> Checkpoint: {ckpt_path}")
    print(f" >>> Timesteps : {a.timesteps}")
    print("=" * 65)

    manifest = load_split_manifest(cfg.data.splits_manifest)
    train_ds = AptosImageDataset(manifest['splits']['train'], 'train', size=224, training=False)
    test_ds = AptosImageDataset(manifest['splits']['test'], 'test', size=224, training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    denoiser = QuantumConditionalModel(config=cfg, feature_dim=cfg.model.feature_dim, n_qubits=cfg.model.n_qubits).to(device)
    aux_model = QuantumDCG(config=cfg, n_qubits=cfg.model.n_qubits, pretrained=False).to(device)

    ckpt_payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
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

    with open('option/diff_DDIM.yaml') as f:
        diff_opt = EasyDict(yaml.safe_load(f))
    diff_opt['scheduler']['num_test_timesteps'] = a.timesteps
    scheduler = pipeline.create_SR3scheduler(diff_opt['scheduler'], 'test')
    scheduler.set_timesteps(a.timesteps)
    scheduler.diff_chns = cfg.data.num_classes
    sampler = pipeline.SR3Sampler(model=denoiser, scheduler=scheduler)

    def extract(loader):
        all_trajs, all_labels, all_preds, all_ids = [], [], [], []
        with torch.no_grad():
            for b in loader:
                x = b[0].to(device)
                y = b[1]
                img_ids = b[2] if len(b) > 2 else None
                bz = x.shape[0]
                nc = cfg.data.num_classes

                y0_aux, y0_g, y0_l, patches, attns, attn_map = aux_model(x)
                np_val = attns.size(1)

                distance_to_diag = torch.tensor([[abs(i - j) for j in range(np_val)] for i in range(np_val)]).to(device)
                weight_g = 1.0 - distance_to_diag / (np_val - 1)
                weight_l = distance_to_diag / (np_val - 1)
                interpolated = weight_l.unsqueeze(0).unsqueeze(0) * y0_l.unsqueeze(-1).unsqueeze(-1) + \
                               weight_g.unsqueeze(0).unsqueeze(0) * y0_g.unsqueeze(-1).unsqueeze(-1)
                diag_indices = torch.arange(np_val)
                y0_cond = interpolated.clone()
                for i in range(bz):
                    for j in range(nc):
                        y0_cond[i, j, diag_indices, diag_indices] = y0_g[i, j]
                        y0_cond[i, j, np_val - 1, 0] = y0_l[i, j]
                        y0_cond[i, j, 0, np_val - 1] = y0_l[i, j]

                yT = y0_cond.clone()
                for i in range(bz):
                    for j in range(nc):
                        yT[i, j, diag_indices, diag_indices] = torch.rand_like(y0_g[i, j])
                        yT[i, j, np_val - 1, 0] = torch.rand_like(y0_l[i, j])
                        yT[i, j, 0, np_val - 1] = torch.rand_like(y0_l[i, j])

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

    print("[*] Extracting 16-step trajectories on Train (2,562) & Test (550)...")
    X_tr, y_tr, p_tr, ids_tr = extract(train_loader)
    X_te, y_te, p_te, ids_te = extract(test_loader)

    # 1. Direct Diffusion
    m_direct = classification_metrics(y_te.tolist(), p_te.tolist())

    # 2. Multimodal ESN
    sam_dict = torch.load('quantum_v7_balanced/outputs/efficient_sam_aptos_cache.pt', map_location='cpu')
    sam_tr = np.array([sam_dict[i].numpy() for i in ids_tr])
    sam_te = np.array([sam_dict[i].numpy() for i in ids_te])

    reservoir = make_esn(input_dim=5, hidden_dim=64, seed=2026, spectral_radius=0.9, leak=0.5)
    H_tr = esn_transform(X_tr, reservoir)
    H_te = esn_transform(X_te, reservoir)

    pca = PCA(n_components=16, random_state=2026)
    sam_tr_p = pca.fit_transform(sam_tr)
    sam_te_p = pca.transform(sam_te)

    H_multi_tr = np.hstack([H_tr, sam_tr_p])
    H_multi_te = np.hstack([H_te, sam_te_p])

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5, random_state=2026))
    clf.fit(H_multi_tr, y_tr)
    p_multi_te = clf.predict_proba(H_multi_te)
    m_esn = classification_metrics(y_te.tolist(), p_multi_te.tolist())

    print("\n" + "=" * 65)
    print(" >>> FINAL CLEAN TEST RESULTS (550 SAMPLES)")
    print("=" * 65)
    print(f"Direct Diffusion : Acc={m_direct['accuracy']*100:.2f}%, F1={m_direct['macro_f1']:.4f}, QWK={m_direct['quadratic_weighted_kappa']:.4f}")
    print(f"Multimodal ESN   : Acc={m_esn['accuracy']*100:.2f}%, F1={m_esn['macro_f1']:.4f}, QWK={m_esn['quadratic_weighted_kappa']:.4f}")
    print(f"Per-Class F1     : {[round(x, 4) for x in m_esn['per_class_f1']]}")

    out_payload = {
        'checkpoint': ckpt_path,
        'direct_diffusion': m_direct,
        'multimodal_esn': m_esn
    }
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, 'w') as f:
        json.dump(out_payload, f, indent=2)
    print(f"[+] Saved to: {a.output}")

if __name__ == '__main__':
    main()
