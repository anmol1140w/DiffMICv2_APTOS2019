"""Train Balanced Quantum-Enhanced DiffMICv2 (512-dim, Class-Weighted Focal Loss).
- EfficientSAM ViT foundation model global image features with cached fast-path.
- 4-Qubit Quantum Cross-Conditioning Bridge.
- Pretrained ResNet-18 QuantumDCG auxiliary guidance.
- Widen hidden capacity: feature_dim = 512, hidden_dim = 512.
- Class-weighted focal loss [0.8, 2.8, 1.2, 5.0, 2.5] (zero data duplication).
- Runs at ~65s per epoch.
"""
import argparse, os, sys, yaml, time
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from easydict import EasyDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_v7_balanced.quantum_diffmic import QuantumDiffMICSystem
from utils import get_dataset
from torch.utils.data import DataLoader

def main():
    p = argparse.ArgumentParser(description="Train Balanced Quantum-DiffMICv2")
    p.add_argument('--config', default='quantum_v7_balanced/config_balanced.yml')
    p.add_argument('--logdir', default='quantum_v7_balanced')
    p.add_argument('--aux-checkpoint', default='quantum_v7_balanced/ckpts/aptos_quantum_dcg_best.pth')
    p.add_argument('--max-epochs', type=int, default=150)
    p.add_argument('--min-epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--patience', type=int, default=35)
    p.add_argument('--accumulate-grad-batches', type=int, default=2)
    p.add_argument('--qubits', type=int, default=4, choices=[2, 4])
    a = p.parse_args()

    with open(a.config) as f:
        cfg = EasyDict(yaml.safe_load(f))

    if a.aux_checkpoint:
        cfg.data.aux_checkpoint = a.aux_checkpoint
    if a.batch_size:
        cfg.training.batch_size = a.batch_size
        cfg.testing.batch_size = a.batch_size
    cfg.model.n_qubits = a.qubits

    ckpt_dir = os.path.join(a.logdir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(a.logdir, 'logs'), exist_ok=True)

    print("\n" + "=" * 70)
    print(" >>> STARTING BALANCED QUANTUM-DIFFMICv2 TRAINING (512-DIM)")
    print(f" >>> Diffusion Capacity: feature_dim={cfg.model.feature_dim}, hidden_dim={cfg.model.hidden_dim}")
    print(f" >>> Quantum Conditioning Bridge: {a.qubits}-Qubit Entangled Cross-Modal Bridge")
    print(f" >>> Global Image Foundation: EfficientSAM ViT (Precomputed Cache, 1 min/epoch)")
    print(f" >>> Class Weights: [0.8 (Gr0), 2.8 (Gr1), 1.2 (Gr2), 5.0 (Gr3), 2.5 (Gr4)]")
    print(f" >>> Aux Guidance Checkpoint: {cfg.data.aux_checkpoint}")
    print(f" >>> Effective Batch Size: {cfg.training.batch_size * a.accumulate_grad_batches}")
    print(f" >>> Early Stopping Patience: {a.patience} | Min Epochs: {a.min_epochs}")
    print("=" * 70 + "\n")

    data_obj, train_dataset, test_dataset = get_dataset(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=cfg.testing.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    system = QuantumDiffMICSystem(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor='acc_f1_score',
        dirpath=ckpt_dir,
        filename='diffmic_quantum_balanced_epoch-{epoch:02d}_acc-{accuracy:.4f}_f1-{f1:.4f}',
        auto_insert_metric_name=False,
        save_top_k=1,
        mode='max'
    )
    last_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='last',
        auto_insert_metric_name=False,
        save_top_k=1,
        every_n_epochs=1
    )
    early_stop = EarlyStopping(
        monitor='acc_f1_score',
        patience=a.patience,
        min_delta=0.001,
        mode='max',
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(a.logdir, 'logs'),
        name='diffmic_quantum'
    )
    csv_logger = CSVLogger(
        save_dir=os.path.join(a.logdir, 'logs'),
        name='diffmic_quantum_csv'
    )

    trainer = pl.Trainer(
        max_epochs=a.max_epochs,
        min_epochs=a.min_epochs,
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback, last_callback, early_stop, lr_monitor],
        logger=[tb_logger, csv_logger],
        accumulate_grad_batches=a.accumulate_grad_batches,
        log_every_n_steps=cfg.training.logging_freq,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        precision='32-true'
    )

    t0 = time.time()
    trainer.fit(system, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.time() - t0
    print(f"\n[+] Training finished in {elapsed/60.0:.2f} minutes.")
    print(f"[+] Best Checkpoint: {checkpoint_callback.best_model_path}")
    print(f"[+] Best Score (Acc+F1): {checkpoint_callback.best_model_score}")

if __name__ == '__main__':
    main()
