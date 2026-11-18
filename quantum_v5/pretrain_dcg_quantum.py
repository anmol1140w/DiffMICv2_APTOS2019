"""Pretrain QuantumDCG on APTOS 2019 dataset.
- Lightweight MobileNetV3-Small backbone + 4-Qubit Quantum Feature Fusion.
- Smaller batch size (16) with fast CUDA execution.
- Validation after EVERY epoch.
- Proper Early Stopping with patience and automatic saving of best checkpoint.
"""
import argparse, os, sys, yaml, time, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from easydict import EasyDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_v5.quantum_dcg import QuantumDCG
from aptos_research.data import load_split_manifest, AptosImageDataset
from aptos_research.metrics import classification_metrics

def main():
    p = argparse.ArgumentParser(description="Pretrain QuantumDCG with Proper Early Stopping")
    p.add_argument('--config', default='quantum_v5/config_quantum.yml')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--patience', type=int, default=None, help="Early stopping patience")
    p.add_argument('--min-epochs', type=int, default=None, help="Minimum epochs before early stopping")
    p.add_argument('--qubits', type=int, default=4, choices=[2, 4], help="Qubits for Quantum Fusion (2 or 4)")
    p.add_argument('--output', default='quantum_v5/ckpts/aptos_quantum_dcg_best.pth')
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(a.config) as f:
        cfg = EasyDict(yaml.safe_load(f))

    epochs = a.epochs if a.epochs is not None else cfg.dcg.epochs
    batch_size = a.batch_size if a.batch_size is not None else cfg.dcg.batch_size
    lr = a.lr if a.lr is not None else cfg.dcg.lr
    patience = a.patience if a.patience is not None else cfg.dcg.early_stopping_patience
    min_epochs = a.min_epochs if a.min_epochs is not None else cfg.dcg.min_epochs
    n_qubits = a.qubits

    print("\n" + "=" * 65)
    print(" >>> STARTING QUANTUM-DCG PRETRAINING (ULTRA-FAST VECTORIZED)")
    print(f" >>> CNN Backbone: MobileNetV3-Small (2.01M total params)")
    print(f" >>> Quantum Fusion: {n_qubits}-Qubit Entangled Correlator Circuit")
    print(f" >>> Batch Size: {batch_size} | Learning Rate: {lr} | Max Epochs: {epochs}")
    print(f" >>> Accelerated Mode: Vectorized GPU Grid-Sampling + BF16 Mixed Precision")
    print(f" >>> Early Stopping: Patience={patience}, Min Epochs={min_epochs}")
    print(f" >>> Validation: After EVERY epoch")
    print(f" >>> Output Checkpoint: {a.output}")
    print("=" * 65 + "\n")

    manifest = load_split_manifest(cfg.data.splits_manifest)
    train_ds = AptosImageDataset(manifest['splits']['train'], 'train', size=224, training=True)
    val_ds = AptosImageDataset(manifest['splits']['validation'], 'validation', size=224, training=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    class_weights = torch.tensor([0.496, 1.097, 0.666, 1.516, 1.225], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    model = QuantumDCG(config=cfg, n_qubits=n_qubits, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_score = -1.0
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_val_qwk = 0.0
    best_epoch = 0
    patience_counter = 0

    log_csv = 'quantum_v5/logs/dcg_quantum_metrics.csv'
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    with open(log_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_acc', 'val_f1', 'acc_f1_score', 'val_qwk', 'time_sec'])

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for x, y, _, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                y_pred, y_glob, y_loc, patches, attns, saliency = model(x)
                l_fuse = loss_fn(y_pred, y)
                l_glob = loss_fn(y_glob, y)
                l_loc  = loss_fn(y_loc, y)
                total_loss = l_fuse + 0.5 * l_glob + 0.5 * l_loc

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        epoch_time = time.time() - t0

        # Per-epoch validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y, _, _ in val_loader:
                x = x.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    y_pred, *_ = model(x)
                probs = torch.softmax(y_pred.float(), dim=-1).cpu().numpy()
                val_preds.extend(probs.tolist())
                val_targets.extend(y.numpy().tolist())

        m = classification_metrics(val_targets, val_preds)
        val_acc = m['accuracy'] * 100.0
        val_f1 = m['macro_f1']
        val_qwk = m['quadratic_weighted_kappa']
        acc_f1_score = val_f1 + (val_acc / 100.0)

        # Log CSV
        with open(log_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_acc:.2f}", f"{val_f1:.4f}", f"{acc_f1_score:.4f}", f"{val_qwk:.4f}", f"{epoch_time:.1f}"])

        print(f"Epoch [{epoch:03d}/{epochs:03d}] ({epoch_time:.1f}s) - Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f} | Acc+F1: {acc_f1_score:.4f} | QWK: {val_qwk:.4f}", end="")

        # Check for improvement: prioritize Acc & F1 Score (Best and Atop)
        if acc_f1_score > best_val_score:
            best_val_score = acc_f1_score
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_val_qwk = val_qwk
            best_epoch = epoch
            patience_counter = 0

            os.makedirs(os.path.dirname(a.output), exist_ok=True)
            # Save format compatible with DiffMIC aux_model loading
            torch.save([model.state_dict()], a.output)
            print(f"  --> [* BEST ATOP ACC: {val_acc:.2f}% | F1: {val_f1:.4f} (Score: {acc_f1_score:.4f}) saved!]")
        else:
            patience_counter += 1
            print(f"  [patience {patience_counter}/{patience}]")

        # Early stopping condition
        if epoch >= min_epochs and patience_counter >= patience:
            print(f"\n >>> [*] Early stopping triggered at Epoch {epoch}! No Acc+F1 improvement for {patience} epochs.")
            break

    print("\n" + "=" * 65)
    print(" >>> QUANTUM-DCG PRETRAINING COMPLETE!")
    print(f" >>> Best Epoch: {best_epoch:03d}")
    print(f" >>> Best Val Acc: {best_val_acc:.2f}% | Best Val F1: {best_val_f1:.4f} | Best Val QWK: {best_val_qwk:.4f}")
    print(f" >>> Checkpoint Saved: {a.output}")
    print("=" * 65 + "\n")

if __name__ == '__main__':
    main()
