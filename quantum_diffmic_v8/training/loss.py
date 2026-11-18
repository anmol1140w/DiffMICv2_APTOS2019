"""Loss Functions for Unified Quantum-DiffMICv2.
- Multi-branch class-balanced focal loss for DCG (Global, Local, and Fusion heads).
- Prior-guided focal weighted MSE for Conditional Diffusion.
- Ordinal penalty loss for Cohen's Kappa optimization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiBranchFocalLoss(nn.Module):
    """Class-Balanced Multi-Branch Focal Loss for DCG."""
    def __init__(self, gamma: float = 1.5, class_weights: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        if class_weights is None:
            # Class-frequency rebalancing for APTOS [0.8, 2.5, 1.2, 4.5, 2.2]
            class_weights = torch.tensor([0.8, 2.5, 1.2, 4.5, 2.2], dtype=torch.float32)
        self.register_buffer('class_weights', class_weights)

    def _focal_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes focal loss with class weighting."""
        if targets.dim() == 1:
            targets = F.one_hot(targets.long(), num_classes=logits.size(1)).float()
        probs = F.softmax(logits, dim=1)
        pt = (probs * targets).sum(dim=1).clamp(min=1e-7, max=1.0)
        focal_weight = (1.0 - pt) ** self.gamma
        cls_w = (targets * self.class_weights.to(targets.device)).sum(dim=1)
        ce = -(targets * torch.log(probs.clamp(min=1e-7))).sum(dim=1)
        return (cls_w * focal_weight * ce).mean()

    def forward(self, y_fusion: torch.Tensor, y_global: torch.Tensor, 
                y_local: torch.Tensor, targets: torch.Tensor, return_dict: bool = False):
        """Combines fusion, global, and local branches with multi-task weights."""
        if targets.dim() == 1:
            targets = F.one_hot(targets.long(), num_classes=y_fusion.size(1)).float()
        loss_f = self._focal_ce(y_fusion, targets)
        loss_g = self._focal_ce(y_global, targets)
        loss_l = self._focal_ce(y_local, targets)
        total = loss_f + 0.5 * loss_g + 0.5 * loss_l
        if return_dict:
            return total, {
                'total_loss': total.item(),
                'aux_loss': loss_f.item(),
                'global_loss': loss_g.item(),
                'local_loss': loss_l.item()
            }
        return total


class DiffusionFocalLoss(nn.Module):
    """Prior-guided weighted MSE loss for conditional diffusion denoising."""
    def __init__(self, gamma: float = 1.5, alpha: float = 10.0, class_weights: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if class_weights is None:
            class_weights = torch.tensor([0.8, 2.5, 1.2, 4.5, 2.2], dtype=torch.float32)
        self.register_buffer('class_weights', class_weights)

    def forward(self, prior: torch.Tensor, targets: torch.Tensor, 
                noise_pred: torch.Tensor, noise_gt: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1:
            targets = F.one_hot(targets.long(), num_classes=prior.size(1)).float()
        probs = F.softmax(prior, dim=1)
        pt = (probs * targets).sum(dim=1).clamp(min=1e-7, max=1.0)
        focal_weight = (1.0 - pt) ** self.gamma
        cls_w = (targets * self.class_weights.to(targets.device)).sum(dim=1)
        
        # Spatial MSE over class-score maps
        mse = F.mse_loss(noise_pred, noise_gt, reduction='none').mean(dim=[1, 2, 3])
        return ((1.0 + self.alpha * focal_weight * cls_w) * mse).mean()
