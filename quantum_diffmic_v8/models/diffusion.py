"""Unified Quantum-Assisted Conditional Diffusion Model for Medical Image Classification.
- Operates on continuous spatial class-score maps (not raw retinal pixels).
- Multi-scale visual conditioning fusing global context and local lesion ROI patches.
- Differentiable Quantum Conditioning Bridge modulating diffusion features via FiLM.
- Matched classical conditioning fallback for rigorous scientific ablation.
- Numerical stability guards with automatic NaN/Inf interception.
"""
import os, sys, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from easydict import EasyDict
from typing import Tuple, List, Optional

from quantum_diffmic_v8.models.quantum_layers import QuantumConditioningBridge
from quantum_diffmic_v8.models.dcg import ResNet18FeatureExtractor, EfficientNetFeatureExtractor, UnifiedQuantumDCG
import pipeline
from utils import cast_label_to_one_hot_and_prototype, compute_isic_metrics, get_optimizer

class SamCachedEncoder(nn.Module):
    """EfficientSAM foundation model feature encoder utilizing disk/memory cached embeddings."""
    def __init__(self, feature_dim: int = 512, cache_path: Optional[str] = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.proj_linear = nn.Linear(256, feature_dim)
        self.cached_embeddings = {}
        
        candidates = [
            cache_path,
            'quantum_v7_balanced/outputs/efficient_sam_aptos_cache.pt',
            'quantum_v5/outputs/efficient_sam_aptos_cache.pt',
            '/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019/quantum_v5/outputs/efficient_sam_aptos_cache.pt'
        ]
        for c in candidates:
            if c and os.path.exists(c):
                try:
                    self.cached_embeddings = torch.load(c, map_location='cpu')
                    print(f"[*] SamCachedEncoder: Loaded {len(self.cached_embeddings)} cached embeddings from {c}")
                    break
                except Exception as e:
                    pass

    def forward(self, x: torch.Tensor, image_ids: Optional[List[str]] = None) -> torch.Tensor:
        B = x.shape[0]
        device = x.device
        if image_ids is not None and all(i in self.cached_embeddings for i in image_ids):
            embeds = [self.cached_embeddings[i].to(device) for i in image_ids]
            feat = torch.stack(embeds, dim=0)
        else:
            feat = torch.zeros(B, 256, device=device)
        return self.proj_linear(feat)


class UnifiedQuantumConditionalModel(nn.Module):
    """Conditional Diffusion Denoising Network with Multi-scale Visual Conditioning
    and Quantum Cross-Subspace FiLM Modulation Bridge.
    """
    def __init__(self, config=None, num_classes: int = 5, feature_dim: int = 512,
                 n_steps: int = 1001, n_qubits: int = 4, classical_mode: bool = False,
                 guidance: bool = True):
        super().__init__()
        self.num_classes = num_classes if config is None else getattr(config.data, 'num_classes', num_classes)
        self.feature_dim = feature_dim if config is None else getattr(config.model, 'feature_dim', feature_dim)
        n_steps = n_steps if config is None else (config.diffusion.timesteps + 1)
        self.n_qubits = n_qubits
        self.classical_mode = classical_mode
        self.guidance = guidance
        
        # 1. Global image feature extractor
        use_sam = config is not None and getattr(config.model, 'encoder', None) == 'sam'
        self.use_sam = use_sam
        if use_sam:
            self.encoder_x = SamCachedEncoder(feature_dim=self.feature_dim)
        else:
            self.encoder_x = ResNet18FeatureExtractor(pretrained=True)
            self.pool_x = nn.AdaptiveAvgPool2d((1, 1))
            self.proj_x = nn.Linear(512, self.feature_dim)
        self.norm_x = nn.BatchNorm1d(self.feature_dim)
        
        # 2. Local patch feature extractor (EfficientNet-B3 or ResNet-18)
        arch = 'b3' if config is None else getattr(config.model, 'arch', 'b3')
        if 'resnet' in str(arch).lower():
            self.encoder_l = ResNet18FeatureExtractor(pretrained=True)
        else:
            variant = 'b2' if 'b2' in str(arch).lower() else 'b3'
            self.encoder_l = EfficientNetFeatureExtractor(variant=variant, pretrained=True)
        self.pool_l = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_l = nn.Linear(512, self.feature_dim)
        self.norm_l = nn.BatchNorm1d(self.feature_dim)
        
        # 3. Timestep embedding & initial projection
        self.embed_t = nn.Embedding(n_steps, self.feature_dim)
        in_channels = self.num_classes * 2 if guidance else self.num_classes
        self.lin1 = nn.Conv2d(in_channels, self.feature_dim, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(self.feature_dim)
        
        # 4. Multi-scale condition weighting
        num_patches = 6 if config is None else getattr(config.model, 'num_k', 6)
        self.cond_weight = nn.Parameter(torch.randn((1, self.feature_dim, 1 + num_patches)) * 0.1)
        
        # 5. Quantum Cross-Conditioning Bridge (with classical fallback)
        self.quantum_bridge = QuantumConditioningBridge(
            feature_dim=self.feature_dim, 
            n_qubits=n_qubits, 
            classical_mode=classical_mode
        )
        
        # 6. Denoising refinement convolutions
        self.lin2 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(self.feature_dim)
        self.lin3 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(self.feature_dim)
        self.lin4 = nn.Conv2d(self.feature_dim, self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, 
                x_l: torch.Tensor, attn: Optional[torch.Tensor] = None, 
                image_ids: Optional[List[str]] = None) -> torch.Tensor:
        B = x.shape[0]
        K = x_l.shape[1]
        
        # Extract global visual features
        if self.use_sam:
            f_x = self.encoder_x(x, image_ids=image_ids)
        else:
            f_x = self.proj_x(self.pool_x(self.encoder_x(x)).view(B, -1))
        f_x = self.norm_x(f_x) # (B, feature_dim)
        
        # Extract local lesion features on ROI crops
        x_l_3ch = x_l.view(B * K, 1, x_l.shape[2], x_l.shape[3]).expand(-1, 3, -1, -1)
        f_l = self.proj_l(self.pool_l(self.encoder_l(x_l_3ch)).view(B * K, -1))
        f_l = self.norm_l(f_l).view(B, K, self.feature_dim).permute(0, 2, 1) # (B, feature_dim, K)
        
        # Multi-scale condition fusion
        f_all = torch.cat([f_x.unsqueeze(-1), f_l], dim=-1) # (B, feature_dim, 1 + K)
        w = torch.softmax(self.cond_weight, dim=2)
        x_weight = torch.sum(f_all * w, dim=-1) # (B, feature_dim)
        
        # Timestep modulation embedding
        B, _, H, W = y.shape
        gamma_t = self.embed_t(t)
        if gamma_t.numel() == B * H * W * self.feature_dim:
            gamma_t = gamma_t.reshape(B, H * W, self.feature_dim).permute(0, 2, 1).reshape(B, self.feature_dim, H, W)
        elif gamma_t.numel() == B * self.feature_dim:
            gamma_t = gamma_t.view(B, self.feature_dim, 1, 1)
        else:
            gamma_t = gamma_t.reshape(B, -1, self.feature_dim).mean(dim=1).view(B, self.feature_dim, 1, 1)
            
        # Denoising Block 1
        y = self.norm1(self.lin1(y)) * gamma_t
        y = F.softplus(y)
        skip = y
        
        # Quantum Conditioning Bridge Modulation (FiLM scale & shift)
        y = self.quantum_bridge(y, x_weight)
        
        # Denoising Blocks 2 & 3
        y = self.norm2(self.lin2(y)) * gamma_t
        y = F.softplus(y)
        y = self.norm3(self.lin3(y)) * gamma_t
        y = F.softplus(y) + skip
        y = self.lin4(y)
        return y


def guided_prob_map(y0_g: torch.Tensor, y0_l: torch.Tensor, bz: int, nc: int, np_val: int) -> torch.Tensor:
    distance_to_diag = torch.tensor([[abs(i - j) for j in range(np_val)] for i in range(np_val)], dtype=y0_g.dtype, device=y0_g.device)
    weight_g = 1.0 - distance_to_diag / (np_val - 1)
    weight_l = distance_to_diag / (np_val - 1)
    interpolated = weight_l.unsqueeze(0).unsqueeze(0) * y0_l.unsqueeze(-1).unsqueeze(-1) + \
                   weight_g.unsqueeze(0).unsqueeze(0) * y0_g.unsqueeze(-1).unsqueeze(-1)
    diag_indices = torch.arange(np_val, device=y0_g.device)
    prob_map = interpolated.clone()
    for i in range(bz):
        for j in range(nc):
            prob_map[i, j, diag_indices, diag_indices] = y0_g[i, j].to(dtype=prob_map.dtype)
            prob_map[i, j, np_val - 1, 0] = y0_l[i, j].to(dtype=prob_map.dtype)
            prob_map[i, j, 0, np_val - 1] = y0_l[i, j].to(dtype=prob_map.dtype)
    return prob_map


class UnifiedQuantumDiffMICSystem(pl.LightningModule):
    """PyTorch Lightning System for Unified Quantum-Enhanced DiffMICv2."""
    def __init__(self, config):
        super().__init__()
        self.params = config
        self.epochs = config.training.n_epochs
        self.initlr = config.optim.lr
        
        # Load DDIM scheduler config
        ddim_config_path = r'option/diff_DDIM.yaml'
        if not os.path.exists(ddim_config_path):
            ddim_config_path = '/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019/option/diff_DDIM.yaml'
        with open(ddim_config_path, 'r') as f:
            self.diff_opt = EasyDict(yaml.safe_load(f))
            
        n_qubits = getattr(config.model, 'n_qubits', 4)
        feature_dim = getattr(config.model, 'feature_dim', 512)
        classical_mode = getattr(config.model, 'classical_mode', False)
        
        # Initialize Unified Quantum Diffusion Model and DCG Guidance Model
        self.model = UnifiedQuantumConditionalModel(
            config=self.params, 
            feature_dim=feature_dim, 
            n_qubits=n_qubits, 
            classical_mode=classical_mode,
            guidance=self.params.diffusion.include_guidance
        )
        self.aux_model = UnifiedQuantumDCG(
            config=self.params, 
            n_qubits=n_qubits, 
            classical_mode=classical_mode,
            pretrained=False
        )
        
        aux_ckpt = getattr(self.params.data, 'aux_checkpoint', None)
        if aux_ckpt and os.path.isfile(aux_ckpt):
            self.init_aux_weights(aux_ckpt)
        self.aux_model.eval()
        
        self.save_hyperparameters()
        self.gts = []
        self.preds = []
        
        self.DiffSampler = pipeline.SR3Sampler(
            model=self.model,
            scheduler=pipeline.create_SR3scheduler(self.diff_opt['scheduler'], 'train'),
        )
        test_steps = getattr(self.params.diffusion, 'test_timesteps', 10)
        self.DiffSampler.scheduler.set_timesteps(test_steps)
        self.DiffSampler.scheduler.diff_chns = self.params.data.num_classes

    def init_aux_weights(self, ckpt_path: str):
        """Loads weights into DCG model with flexible prefix handling."""
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, list) and len(ckpt) > 0:
            ckpt = ckpt[0]
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        ckpt = {k.replace('aux_model.', ''): v for k, v in ckpt.items()}
        msg = self.aux_model.load_state_dict(ckpt, strict=False)
        print(f"[*] Loaded DCG weights from {ckpt_path}: {msg}")

    def configure_optimizers(self):
        optimizer = get_optimizer(self.params.optim, filter(lambda p: p.requires_grad, self.model.parameters()))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01)
        return [optimizer], [scheduler]

    def diffusion_focal_loss(self, prior, targets, noise, noise_gt, gamma=1.5, alpha=10.0):
        probs = F.softmax(prior, dim=1)
        probs = (probs * targets).sum(dim=1)
        weights = 1.0 + alpha * (1.0 - probs) ** gamma
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Class-frequency rebalancing for APTOS [0.8, 2.5, 1.2, 4.5, 2.2]
        cls_weights = torch.tensor([0.8, 2.5, 1.2, 4.5, 2.2], dtype=targets.dtype, device=targets.device)
        sample_cls_weights = (targets * cls_weights).sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        return (sample_cls_weights.to(noise.dtype) * weights.to(noise.dtype) * (noise - noise_gt).square()).mean()

    def guided_prob_map(self, y0_g, y0_l, bz, nc, np_val):
        return guided_prob_map(y0_g, y0_l, bz, nc, np_val)

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.aux_model.eval()
        
        x_batch, y_batch = batch[:2]
        image_ids = batch[2] if len(batch) > 2 else None
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.to(self.device)
        x_batch = x_batch.to(self.device)
        
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)
            
        bz, nc, H, W = attn_map.size()
        bz, np_val = attns.size()
        
        y_map = y_batch.unsqueeze(1).expand(-1, np_val * np_val, -1).reshape(bz * np_val * np_val, nc)
        noise = torch.randn_like(y_map).to(self.device)
        timesteps = torch.randint(0, self.DiffSampler.scheduler.config.num_train_timesteps, (bz * np_val * np_val,), device=self.device).long()
        
        noisy_y = self.DiffSampler.scheduler.add_noise(y_map, timesteps=timesteps, noise=noise)
        noisy_y = noisy_y.view(bz, np_val * np_val, -1).permute(0, 2, 1).reshape(bz, nc, np_val, np_val)
        
        y0_cond = self.guided_prob_map(y0_aux_global, y0_aux_local, bz, nc, np_val)
        y_fusion = torch.cat([y0_cond, noisy_y], dim=1)
        
        attns_mat = attns.unsqueeze(-1)
        attns_mat = (attns_mat * attns_mat.transpose(1, 2)).unsqueeze(1)
        
        noise_pred = self.model(x_batch, y_fusion, timesteps, patches, attns_mat, image_ids=image_ids)
        noise = noise.view(bz, np_val * np_val, -1).permute(0, 2, 1).reshape(bz, nc, np_val, np_val)
        
        loss = self.diffusion_focal_loss(y0_aux, y_batch, noise_pred, noise)
        
        # Guard against NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[!] Warning: NaN/Inf detected in loss! Clamping to finite value.")
            loss = torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=0.0)
            
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self.aux_model.eval()
        
        x_batch, y_batch = batch[:2]
        image_ids = batch[2] if len(batch) > 2 else None
        y_batch, _ = cast_label_to_one_hot_and_prototype(y_batch, self.params)
        y_batch = y_batch.to(self.device)
        x_batch = x_batch.to(self.device)
        
        with torch.no_grad():
            y0_aux, y0_aux_global, y0_aux_local, patches, attns, attn_map = self.aux_model(x_batch)
            
        bz, nc, H, W = attn_map.size()
        bz, np_val = attns.size()
        
        y0_cond = self.guided_prob_map(y0_aux_global, y0_aux_local, bz, nc, np_val)
        yT = self.guided_prob_map(torch.rand_like(y0_aux_global), torch.rand_like(y0_aux_local), bz, nc, np_val)
        
        attns_mat = attns.unsqueeze(-1)
        attns_mat = (attns_mat * attns_mat.transpose(1, 2)).unsqueeze(1)
        
        y_pred = self.DiffSampler.sample_high_res(x_batch, yT, conditions=[y0_cond, patches, attns_mat, image_ids])
        y_pred = y_pred.reshape(bz, nc, np_val * np_val).mean(2)
        
        self.preds.append(y_pred.detach())
        self.gts.append(y_batch.detach())

    def on_validation_epoch_end(self):
        if not self.preds or not self.gts:
            return
        gt = torch.cat(self.gts)
        pred = torch.cat(self.preds)
        ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa = compute_isic_metrics(gt, pred)
        acc_f1_score = F1 + ACC
        if np.isnan(AUC_ovo):
            AUC_ovo = 0.5
            
        self.log('accuracy', ACC, prog_bar=True)
        self.log('f1', F1, prog_bar=True)
        self.log('acc_f1_score', acc_f1_score, prog_bar=True)
        self.log('cohen_kappa', kappa, prog_bar=True)
        self.log('AUC', AUC_ovo)
        
        print(f"\n[VAL REPORT] Epoch {self.current_epoch:03d} -> ACC: {ACC*100:.2f}% | F1: {F1:.4f} | Acc+F1: {acc_f1_score:.4f} | Kappa: {kappa:.4f}")
        self.gts.clear()
        self.preds.clear()
        torch.cuda.empty_cache()
