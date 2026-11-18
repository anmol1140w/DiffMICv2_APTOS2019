"""Fold-Aware DCG Trainer for Unified Quantum-DiffMICv2.
Trains UnifiedQuantumDCG on fold training split and validates on fold validation split.
Maintains strict top-1 checkpointing to conserve NVMe storage.
"""
import os, sys, json, time, argparse, yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_diffmic_v8.models.dcg import UnifiedQuantumDCG
from quantum_diffmic_v8.training.loss import MultiBranchFocalLoss
from quantum_diffmic_v8.evaluation.metrics import compute_all_medical_metrics
from aptos_research.data import AptosImageDataset

def train_one_fold_dcg(fold_idx: int, splits_path: str, config_path: str,
                       epochs: int = 25, batch_size: int = 64, lr: float = 2e-4,
                       patience: int = 10, qubits: int = 4, classical_mode: bool = False,
                       output_path: str = None, init_from: str = None, device_str: str = 'cuda:0'):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(f" >>> [STAGE 1] TRAINING DCG FOR FOLD {fold_idx}")
    print(f" >>> Device: {device} | Batch Size: {batch_size} | Qubits: {qubits} | Classical: {classical_mode}")
    print("=" * 70)

    with open(splits_path, 'r') as f:
        splits_manifest = json.load(f)
    with open(config_path, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))

    fold_data = splits_manifest['folds'][f'fold_{fold_idx}']
    train_ds = AptosImageDataset(fold_data['train'], 'train', size=224, training=True)
    val_ds = AptosImageDataset(fold_data['val'], 'validation', size=224, training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UnifiedQuantumDCG(
        config=cfg, 
        num_classes=5, 
        num_k=6, 
        n_qubits=qubits, 
        classical_mode=classical_mode, 
        pretrained=(init_from is None)
    ).to(device)

    # Transfer learning / initialization if available
    if init_from and os.path.exists(init_from):
        try:
            ckpt = torch.load(init_from, map_location='cpu', weights_only=False)
            if isinstance(ckpt, list) and len(ckpt) > 0:
                ckpt = ckpt[0]
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            ckpt = {k.replace('aux_model.', ''): v for k, v in ckpt.items()}
            msg = model.load_state_dict(ckpt, strict=False)
            print(f"[*] Initialized DCG from {init_from}: {msg}")
        except Exception as e:
            print(f"[!] Warning: Could not initialize from {init_from}: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = MultiBranchFocalLoss(gamma=1.5).to(device)

    if output_path is None:
        output_path = f"quantum_diffmic_v8/ckpts/dcg_best_fold_{fold_idx}.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    best_score = -1.0
    best_metrics = {}
    epochs_no_improve = 0

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for b_idx, (images, labels, ids, _) in enumerate(train_loader):
            images = images.to(device)
            targets_oh = F.one_hot(labels.to(device), num_classes=5).float()

            optimizer.zero_grad()
            y_f, y_g, y_l, _, _, _ = model(images)
            loss = criterion(y_f, y_g, y_l, targets_oh)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        scheduler.step()

        # Validation loop
        model.eval()
        val_preds, val_gts = [], []
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images = images.to(device)
                y_f, _, _, _, _, _ = model(images)
                val_preds.append(y_f.cpu().numpy())
                val_gts.append(labels.numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_gts = np.concatenate(val_gts, axis=0)
        m = compute_all_medical_metrics(val_gts, val_preds)

        score = m['accuracy'] + m['macro_f1']
        print(f"[Fold {fold_idx} Ep {ep:02d}] Train Loss: {train_loss:.4f} | Val Acc: {m['accuracy']*100:.2f}% | Val F1: {m['macro_f1']:.4f} | QWK: {m['quadratic_weighted_kappa']:.4f} | Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_metrics = m
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_path)
            print(f"  [+] New best DCG checkpoint saved -> {output_path} (Score: {best_score:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and ep >= 12:
                print(f"[*] Early stopping triggered for DCG Fold {fold_idx} after {ep + 1} epochs.")
                break

    print(f"\n[+] DCG Fold {fold_idx} Finished: Best Val Acc={best_metrics['accuracy']*100:.2f}%, F1={best_metrics['macro_f1']:.4f}, QWK={best_metrics['quadratic_weighted_kappa']:.4f}")
    return output_path, best_metrics

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--splits', default='quantum_diffmic_v8/configs/5fold_splits.json')
    p.add_argument('--config', default='quantum_diffmic_v8/configs/config_5fold.yml')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--qubits', type=int, default=4)
    p.add_argument('--classical', action='store_true')
    p.add_argument('--output', default=None)
    p.add_argument('--init-from', default='quantum_v7_balanced/ckpts/aptos_quantum_dcg_best.pth')
    a = p.parse_args()

    train_one_fold_dcg(
        fold_idx=a.fold, splits_path=a.splits, config_path=a.config,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, patience=a.patience,
        qubits=a.qubits, classical_mode=a.classical, output_path=a.output,
        init_from=a.init_from
    )
