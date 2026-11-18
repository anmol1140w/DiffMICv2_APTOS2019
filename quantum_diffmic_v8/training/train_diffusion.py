"""Fold-Aware Conditional Diffusion Trainer for Unified Quantum-DiffMICv2.
Trains UnifiedQuantumDiffMICSystem with effective batch size 64 (physical 32 x 2 accumulation steps).
Maintains strict top-1 atop checkpointing to conserve disk space.
"""
import os, sys, json, time, argparse, yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_diffmic_v8.models.diffusion import UnifiedQuantumDiffMICSystem
from aptos_research.data import AptosImageDataset

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_records, val_records, batch_size=32, num_workers=4):
        super().__init__()
        self.train_records = train_records
        self.val_records = val_records
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        ds = AptosImageDataset(self.train_records, 'train', size=224, training=True)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        ds = AptosImageDataset(self.val_records, 'validation', size=224, training=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True)

def train_one_fold_diffusion(fold_idx: int, splits_path: str, config_path: str,
                            aux_ckpt: str, max_epochs: int = 50, min_epochs: int = 20,
                            batch_size: int = 32, accum_batches: int = 2,
                            patience: int = 20, qubits: int = 4, classical_mode: bool = False,
                            output_dir: str = None, init_from: str = None,
                            device_str: str = 'cuda:0') -> str:
    print("=" * 70)
    print(f" >>> [STAGE 2] TRAINING CONDITIONAL DIFFUSION FOR FOLD {fold_idx}")
    print(f" >>> Effective BS: {batch_size * accum_batches} (Physical {batch_size} x {accum_batches})")
    print(f" >>> Guidance DCG: {aux_ckpt}")
    print(f" >>> Qubits: {qubits} | Classical Mode: {classical_mode}")
    print("=" * 70)

    with open(splits_path, 'r') as f:
        splits_manifest = json.load(f)
    with open(config_path, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))

    # Update dynamic settings in config
    cfg.training.n_epochs = max_epochs
    cfg.model.n_qubits = qubits
    cfg.model.classical_mode = classical_mode
    cfg.data.aux_checkpoint = aux_ckpt

    fold_data = splits_manifest['folds'][f'fold_{fold_idx}']
    dm = CustomDataModule(fold_data['train'], fold_data['val'], batch_size=batch_size, num_workers=4)

    system = UnifiedQuantumDiffMICSystem(config=cfg)

    # Initialize from pretrained diffusion backbone if provided
    if init_from and os.path.exists(init_from):
        try:
            ckpt_payload = torch.load(init_from, map_location='cpu', weights_only=False)
            sd = ckpt_payload.get('state_dict', ckpt_payload)
            den_sd = {k: v for k, v in sd.items() if not k.startswith('aux_model.')}
            msg = system.load_state_dict(den_sd, strict=False)
            print(f"[*] Initialized Diffusion System from {init_from}: {msg}")
        except Exception as e:
            print(f"[!] Warning: Could not initialize diffusion from {init_from}: {e}")

    if output_dir is None:
        output_dir = f"quantum_diffmic_v8/ckpts/fold_{fold_idx}"
    os.makedirs(output_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"diffmic_fold_{fold_idx}_best_epoch-{{epoch:02d}}_acc-{{accuracy:.4f}}_f1-{{f1:.4f}}",
        monitor='acc_f1_score',
        mode='max',
        save_top_k=1,
        save_last=True
    )
    early_cb = EarlyStopping(
        monitor='acc_f1_score',
        patience=patience,
        mode='max',
        min_delta=0.001
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        accelerator='gpu',
        devices=1,
        accumulate_grad_batches=accum_batches,
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb, early_cb],
        default_root_dir=output_dir,
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    trainer.fit(system, datamodule=dm)
    best_ckpt = ckpt_cb.best_model_path or os.path.join(output_dir, 'last.ckpt')
    print(f"\n[+] Diffusion Fold {fold_idx} Completed! Best Checkpoint -> {best_ckpt}")
    return best_ckpt

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--splits', default='quantum_diffmic_v8/configs/5fold_splits.json')
    p.add_argument('--config', default='quantum_diffmic_v8/configs/config_5fold.yml')
    p.add_argument('--aux-checkpoint', required=True)
    p.add_argument('--max-epochs', type=int, default=50)
    p.add_argument('--min-epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--accumulate-grad-batches', type=int, default=2)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--qubits', type=int, default=4)
    p.add_argument('--classical', action='store_true')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--init-from', default=None)
    a = p.parse_args()

    train_one_fold_diffusion(
        fold_idx=a.fold, splits_path=a.splits, config_path=a.config,
        aux_ckpt=a.aux_checkpoint, max_epochs=a.max_epochs, min_epochs=a.min_epochs,
        batch_size=a.batch_size, accum_batches=a.accumulate_grad_batches,
        patience=a.patience, qubits=a.qubits, classical_mode=a.classical,
        output_dir=a.output_dir, init_from=a.init_from
    )
