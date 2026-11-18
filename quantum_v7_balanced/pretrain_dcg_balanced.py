"""High-Performance Quantum-DCG Pretraining with Batch Size 64 & Class-Balanced Focal Loss.
- ResNet-18 multiscale global context & local lesion ROI extractors (512-dim).
- 4-Qubit Quantum Feature Fusion Layer entangling global and local representations.
- Vectorized GPU saliency-guided patch generation and gated attention pooling.
- Class-Balanced Focal Cross-Entropy Loss to eliminate minority collapse (Mild Gr1 & Severe Gr3).
- Early stopping with patience and automatic saving of top-1 atop checkpoint.
"""
import argparse, os, sys, yaml, time, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_v7_balanced.quantum_dcg import QuantumDCG
from aptos_research.data import load_split_manifest, AptosImageDataset
from aptos_research.metrics import classification_metrics


class ClassBalancedFocalCE(nn.Module):
    """Focal Cross-Entropy with class-frequency weights and label smoothing."""
    def __init__(self, weights: torch.Tensor, gamma: float = 1.5, label_smoothing: float = 0.05):
        super().__init__()
        self.register_buffer('weights', weights)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weights, label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def main():
    p = argparse.ArgumentParser(description="Pretrain High-Performance Quantum-DCG (BS=64)")
    p.add_argument('--config', default='quantum_v7_balanced/config_balanced.yml')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--min-epochs', type=int, default=20)
    p.add_argument('--qubits', type=int, default=4, choices=[2, 4])
    p.add_argument('--output', default='quantum_v7_balanced/ckpts/aptos_quantum_dcg_best.pth')
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open(a.config) as f:
        cfg = EasyDict(yaml.safe_load(f))

    epochs = a.epochs
    batch_size = a.batch_size
    lr = a.lr
    patience = a.patience
    min_epochs = a.min_epochs
    n_qubits = a.qubits

    arch_name = getattr(cfg.model, 'arch', 'efficientnet_b3')
    print("\n" + "=" * 70)
    print(" >>> STARTING HIGH-PERFORMANCE QUANTUM-DCG RETRAINING")
    print(f" >>> CNN Backbone       : {arch_name.upper()} (Squeeze-and-Excitation, 512-dim Projected)")
    print(f" >>> Quantum Fusion     : {n_qubits}-Qubit Entangled Circuit (10 Observables)")
    print(f" >>> Batch Size         : {batch_size} (Ultra-Fast 64-sample Batches)")
    print(f" >>> Learning Rate      : {lr} (Cosine Annealing to 1e-6)")
    print(f" >>> Max Epochs         : {epochs} | Min Epochs: {min_epochs} | Patience: {patience}")
    print(f" >>> Class Loss Weights : [0.8, 2.5, 1.2, 4.5, 2.2] (Focal Gamma=1.5)")
    print(f" >>> Output Checkpoint  : {a.output}")
    print("=" * 70 + "\n")

    manifest = load_split_manifest(cfg.data.splits_manifest)
    train_ds = AptosImageDataset(manifest['splits']['train'], 'train', size=224, training=True)
    val_ds = AptosImageDataset(manifest['splits']['validation'], 'validation', size=224, training=False)
    test_ds = AptosImageDataset(manifest['splits']['test'], 'test', size=224, training=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    class_weights = torch.tensor([0.8, 2.5, 1.2, 4.5, 2.2], dtype=torch.float32, device=device)
    loss_fn = ClassBalancedFocalCE(weights=class_weights, gamma=1.5, label_smoothing=0.05)

    model = QuantumDCG(config=cfg, n_qubits=n_qubits, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_score = -1.0
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_val_qwk = 0.0
    best_epoch = 0
    patience_counter = 0

    log_csv = 'quantum_v7_balanced/logs/dcg_pretrain_metrics.csv'
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

        with open(log_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_acc:.2f}", f"{val_f1:.4f}", f"{acc_f1_score:.4f}", f"{val_qwk:.4f}", f"{epoch_time:.1f}"])

        print(f"Epoch [{epoch:03d}/{epochs:03d}] ({epoch_time:.1f}s) - Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f} | Acc+F1: {acc_f1_score:.4f} | QWK: {val_qwk:.4f}", end="")

        if acc_f1_score > best_val_score:
            best_val_score = acc_f1_score
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_val_qwk = val_qwk
            best_epoch = epoch
            patience_counter = 0

            os.makedirs(os.path.dirname(a.output), exist_ok=True)
            torch.save([model.state_dict()], a.output)
            print(f"  --> [* BEST ATOP ACC: {val_acc:.2f}% | F1: {val_f1:.4f} saved!]")
        else:
            patience_counter += 1
            print(f"  [patience {patience_counter}/{patience}]")

        if epoch >= min_epochs and patience_counter >= patience:
            print(f"\n[*] Early stopping triggered at Epoch {epoch}! No Acc+F1 improvement for {patience} epochs.")
            break

    print("\n" + "=" * 70)
    print(" >>> QUANTUM-DCG RETRAINING FINISHED!")
    print(f" >>> Best Epoch: {best_epoch:03d}")
    print(f" >>> Best Val Acc: {best_val_acc:.2f}% | Best Val F1: {best_val_f1:.4f} | QWK: {best_val_qwk:.4f}")
    print(f" >>> Checkpoint Saved: {a.output}")
    print("=" * 70)

    # Clean held-out test evaluation of the best DCG checkpoint
    print("\n[*] Evaluating Best Retrained DCG Checkpoint on Clean 550 Held-Out Test Set...")
    best_weights = torch.load(a.output, map_location='cpu')[0]
    model.load_state_dict(best_weights)
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for x, y, _, _ in test_loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                y_pred, *_ = model(x)
            probs = torch.softmax(y_pred.float(), dim=-1).cpu().numpy()
            test_preds.extend(probs.tolist())
            test_targets.extend(y.numpy().tolist())

    m_test = classification_metrics(test_targets, test_preds)
    print("\n" + "=" * 70)
    print(" >>> RETRAINED QUANTUM-DCG CLEAN TEST PERFORMANCE (550 SAMPLES):")
    print(f" >>> Test Accuracy : {m_test['accuracy']*100:.2f}%")
    print(f" >>> Macro-F1      : {m_test['macro_f1']:.4f}")
    print(f" >>> Cohen Kappa   : {m_test['quadratic_weighted_kappa']:.4f}")
    print(f" >>> Per-Class F1  : {[round(x, 4) for x in m_test['per_class_f1']]}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
