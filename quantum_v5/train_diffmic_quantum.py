"""Train Quantum-Enhanced DiffMICv2 on APTOS 2019 dataset.
- MobileNetV3-Small encoder + 4-Qubit Quantum Cross-Conditioning Bridge.
- Pretrained QuantumDCG auxiliary guidance.
- Validation after EVERY epoch.
- Proper early stopping and top-1 checkpoint saving.
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

from quantum_v5.quantum_diffmic import QuantumDiffMICSystem
from utils import get_dataset
from torch.utils.data import DataLoader

def main():
    p = argparse.ArgumentParser(description="Train Quantum-Enhanced DiffMICv2")
    p.add_argument('--config', default='quantum_v5/config_quantum.yml')
    p.add_argument('--logdir', default='quantum_v5')
    p.add_argument('--aux-checkpoint', default='quantum_v5/ckpts/aptos_quantum_dcg_best.pth')
    p.add_argument('--max-epochs', type=int, default=1000)
    p.add_argument('--min-epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--patience', type=int, default=100)
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

    print("\n" + "=" * 65)
    print(" >>> STARTING QUANTUM-DIFFMICv2 TRAINING")
    print(f" >>> CNN Backbone: MobileNetV3-Small (Compact, Fast)")
    print(f" >>> Quantum Conditioning Bridge: {a.qubits}-Qubit Entangled Cross-Modal Bridge")
    print(f" >>> Aux Guidance Checkpoint: {cfg.data.aux_checkpoint}")
    print(f" >>> Batch Size: {cfg.training.batch_size} (Grad Accum: {a.accumulate_grad_batches})")
    print(f" >>> Validation: After EVERY epoch")
    print(f" >>> Early Stopping Patience: {a.patience}")
    print("=" * 65 + "\n")

    # Data Loaders
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

    # Initialize Quantum-Enhanced System
    system = QuantumDiffMICSystem(cfg)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='acc_f1_score',
        dirpath=ckpt_dir,
        filename='diffmic_quantum_best_epoch-{epoch:02d}_acc-{accuracy:.4f}_f1-{f1:.4f}',
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
        mode='max',
        min_delta=0.002,
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    tb_logger = TensorBoardLogger(save_dir=os.path.join(a.logdir, 'logs'), name='diffmic_quantum')
    csv_logger = CSVLogger(save_dir=os.path.join(a.logdir, 'logs'), name='diffmic_quantum_csv')

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=a.max_epochs,
        min_epochs=a.min_epochs,
        accumulate_grad_batches=a.accumulate_grad_batches,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, last_callback, early_stop, lr_monitor],
        logger=[tb_logger, csv_logger],
        precision='32-true',
        gradient_clip_val=cfg.optim.grad_clip,
        log_every_n_steps=cfg.training.logging_freq
    )

    trainer.fit(system, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\n" + "=" * 65)
    print(" >>> QUANTUM-DIFFMICv2 TRAINING COMPLETE!")
    print(f" >>> Best Checkpoint: {checkpoint_callback.best_model_path}")
    print(f" >>> Best Score (F1): {checkpoint_callback.best_model_score}")
    print("=" * 65 + "\n")

if __name__ == '__main__':
    main()
