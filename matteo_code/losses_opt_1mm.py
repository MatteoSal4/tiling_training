import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class GammaIndexLoss(nn.Module):
    """
    Differentiable Gamma Index Loss based on Martinot et al. (IPMI 2023).
    
    Optimized for 1mm³ isotropic target resolution.
    """
    
    def __init__(
        self,
        dose_percent: float = 3.0,
        dta_mm: float = 3.0,
        voxel_size_mm: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        dose_cutoff: float = 0.1,
        beta_init: float = 0.02,
        max_gamma: float = 10.0,
    ):
        super().__init__()
        
        self.dose_threshold = dose_percent / 100.0
        self.dta_mm = dta_mm
        self.voxel_size = voxel_size_mm if isinstance(voxel_size_mm, (list, tuple)) else [voxel_size_mm] * 3
        self.dose_cutoff = dose_cutoff
        self.max_gamma = max_gamma
        
        self.register_buffer('beta', torch.tensor(beta_init, dtype=torch.float32))
        
        # Search radius for 1mm voxels
        self.search_radius = int(np.ceil(dta_mm))
        
        # Precompute everything
        self._precompute_kernels()
    
    def _precompute_kernels(self):
        """Precompute search kernels and DTA² values for 1mm³ voxels."""
        r = self.search_radius
        k = 2 * r + 1
        dta_sq_threshold = self.dta_mm ** 2
        
        offsets = []
        dta_sq_values = []
        
        for oz in range(-r, r + 1):
            for oy in range(-r, r + 1):
                for ox in range(-r, r + 1):
                    dist_sq = oz * oz + oy * oy + ox * ox
                    
                    if dist_sq <= dta_sq_threshold:
                        offsets.append((oz, oy, ox))
                        dta_sq_values.append(dist_sq / dta_sq_threshold)
        
        n_offsets = len(offsets)
        
        # Build identity kernels
        kernels = torch.zeros(n_offsets, 1, k, k, k, dtype=torch.float32)
        for idx, (oz, oy, ox) in enumerate(offsets):
            kernels[idx, 0, oz + r, oy + r, ox + r] = 1.0
        
        # Register as buffers
        self.register_buffer('kernels', kernels)
        self.register_buffer('dta_sq', torch.tensor(dta_sq_values, dtype=torch.float32))
        
        self.n_offsets = n_offsets
        self.pad_size = r
    
    def _resample_to_1mm(self, x: torch.Tensor) -> torch.Tensor:
        """Resample to 1mm³ using trilinear interpolation."""
        scale = (self.voxel_size[0], self.voxel_size[1], self.voxel_size[2])
        
        if all(abs(s - 1.0) < 1e-6 for s in scale):
            return x
        
        B, C, D, H, W = x.shape
        new_size = (int(D * scale[0]), int(H * scale[1]), int(W * scale[2]))
        
        return F.interpolate(x, size=new_size, mode='trilinear', align_corners=False)
    
    def compute_gamma(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gamma index using evaluation dose search.
        
        Γ(r) = min_{δ} sqrt( |δ|²/DTA² + (D_pred(r+δ) - D_target(r))²/Δ² )
        """
        # Ensure 5D
        if pred.dim() == 3:
            pred = pred.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 4:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        
        # Resample to 1mm³
        pred = self._resample_to_1mm(pred)
        target = self._resample_to_1mm(target)
        
        # Dose threshold
        delta = self.dose_threshold * target.max()
        
        # Valid mask
        cutoff_value = self.dose_cutoff * target.max()
        valid_mask = target >= cutoff_value
        
        # Cast kernels to match input dtype (for mixed precision training)
        kernels = self.kernels.to(dtype=pred.dtype)
        dta_sq = self.dta_sq.to(dtype=pred.dtype)
        
        # Compute gamma using precomputed kernels
        pred_neighbors = F.conv3d(pred, kernels, padding=self.pad_size)
        
        # Gamma² = (dose_diff/Δ)² + (dist/DTA)²
        dose_diff_sq = ((pred_neighbors - target) / (delta + 1e-10)) ** 2
        gamma_sq = dose_diff_sq + dta_sq.view(1, -1, 1, 1, 1)
        
        # Minimum over search neighborhood
        gamma_sq_min, _ = gamma_sq.min(dim=1, keepdim=True)
        gamma = torch.sqrt(gamma_sq_min + 1e-10)
        
        # Clamp
        gamma = torch.clamp(gamma, max=self.max_gamma)
        
        return gamma, valid_mask
    
    def parametric_sigmoid(self, gamma: torch.Tensor) -> torch.Tensor:
        """σ_β(Γ) = sigmoid(β · (1 - Γ))"""
        return torch.sigmoid(self.beta * (1.0 - gamma))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss using Equation 8:
        L = 1 - (Σ σ(β(1-Γ)) · 𝟙[D_r ≥ δ]) / (Σ 𝟙[D_r ≥ δ])
        """
        gamma, valid_mask = self.compute_gamma(pred, target)
        
        # Soft pass rate
        soft_pass = self.parametric_sigmoid(gamma)

        spatial_dims = (1, 2, 3, 4)
        
        n_valid = valid_mask.float().sum(dim=spatial_dims)
        
        soft_gpr = (soft_pass * valid_mask.float()).sum(dim=spatial_dims) / (n_valid + 1e-8)  # [B]
        per_sample_loss = 1.0 - soft_gpr
        
        # Loss = 1 - soft_GPR
        return per_sample_loss.mean()
    
    def compute_pass_rate(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute hard gamma pass rate (%)."""
        with torch.no_grad():
            gamma, valid_mask = self.compute_gamma(pred, target)
            passing = (gamma < 1.0) & valid_mask
            n_valid = valid_mask.float().sum()
            if n_valid == 0:
                return 100.0
            return (passing.float().sum() / n_valid).item() * 100
    
    def set_beta(self, value: float):
        self.beta.fill_(value)
    
    def get_beta(self) -> float:
        return self.beta.item()


class BetaScheduler:
    """Beta annealing from Martinot et al."""
    
    def __init__(
        self,
        loss_fn: GammaIndexLoss,
        beta_init: float = 0.02,
        beta_mid: float = 3.0,
        beta_max: float = 5.0,
        warmup_iters: int = 150,
        phase1_interval: int = 50,
        phase2_interval: int = 100,
        growth_rate: float = 1.05,
    ):
        self.loss_fn = loss_fn
        self.beta = beta_init
        self.beta_init = beta_init
        self.beta_mid = beta_mid
        self.beta_max = beta_max
        self.warmup_iters = warmup_iters
        self.phase1_interval = phase1_interval
        self.phase2_interval = phase2_interval
        self.growth_rate = growth_rate
        self._last_update = 0
        
        loss_fn.set_beta(beta_init)
    
    def step(self, iteration: int) -> float:
        if iteration < self.warmup_iters:
            self.beta = self.beta_init
        else:
            interval = self.phase1_interval if self.beta < self.beta_mid else self.phase2_interval
            if (iteration - self._last_update) >= interval:
                self.beta = min(self.beta * self.growth_rate, self.beta_max)
                self._last_update = iteration
        
        self.loss_fn.set_beta(self.beta)
        return self.beta
    
    def reset(self):
        self.beta = self.beta_init
        self._last_update = 0
        self.loss_fn.set_beta(self.beta_init)

class SimpleBetaScheduler:
    """Beta scheduler with two-phase linear increase."""
    
    def __init__(
        self,
        loss_fn: GammaIndexLoss,
        beta_start: float = 0.1,
        beta_mid: float = 3.0,
        beta_end: float = 5.0,
        warmup_iters: int = 20,
        phase1_end: int = 65,
        phase2_end: int = 145,
        update_interval: int = 5,
    ):
        """
        Args:
            beta_start: Initial beta (0.1)
            beta_mid: Beta at end of phase 1 (3.0 at iter 65)
            beta_end: Beta at end of phase 2 (5.0 at iter 145)
            warmup_iters: Keep beta_start for first N iters (20)
            phase1_end: End of phase 1 (65)
            phase2_end: End of phase 2 (145)
            update_interval: Update beta every N iters (5)
        """
        self.loss_fn = loss_fn
        self.beta_start = beta_start
        self.beta_mid = beta_mid
        self.beta_end = beta_end
        self.warmup_iters = warmup_iters
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.update_interval = update_interval
        
        self._last_update_iter = 0
        self._current_beta = beta_start
        
        loss_fn.set_beta(beta_start)
    
    def step(self, iteration: int) -> float:
        if iteration < self.warmup_iters:
            # Warmup: keep beta constant
            beta = self.beta_start
        
        elif iteration < self.phase1_end:
            # Phase 1: increase from beta_start to beta_mid
            # Only update every update_interval iterations
            if (iteration - self._last_update_iter) >= self.update_interval:
                progress = (iteration - self.warmup_iters) / (self.phase1_end - self.warmup_iters)
                beta = self.beta_start + (self.beta_mid - self.beta_start) * progress
                self._last_update_iter = iteration
                self._current_beta = beta
            else:
                beta = self._current_beta
        
        elif iteration < self.phase2_end:
            # Phase 2: increase from beta_mid to beta_end
            if (iteration - self._last_update_iter) >= self.update_interval:
                progress = (iteration - self.phase1_end) / (self.phase2_end - self.phase1_end)
                beta = self.beta_mid + (self.beta_end - self.beta_mid) * progress
                self._last_update_iter = iteration
                self._current_beta = beta
            else:
                beta = self._current_beta
        
        else:
            # After phase 2: keep beta_end
            beta = self.beta_end
        
        self.loss_fn.set_beta(beta)
        return beta
    
    def reset(self):
        self._last_update_iter = 0
        self._current_beta = self.beta_start
        self.loss_fn.set_beta(self.beta_start)



class WeightedMSE(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: [B, 1, D, H, W] or [B, D, H, W]
            y_true: [B, 1, D, H, W] or [B, D, H, W]
        
        Returns:
            Scalar loss
        """
        
        # Weights based on target dose (higher dose = higher weight)
        weights = torch.exp(self.alpha*torch.clamp(y_true, -10.0, 10.0))
        
        # Squared error
        sq_error = (y_pred - y_true) ** 2
        
        # Weighted squared error
        weighted_sq_error = weights * sq_error
        
        # Sum over spatial dimensions, keep batch dimension
        # For [B, 1, D, H, W]: sum over dims (1, 2, 3, 4)
        # For [B, D, H, W]: sum over dims (1, 2, 3)
        if y_pred.dim() == 5:
            spatial_dims = (1, 2, 3, 4)
        else:
            spatial_dims = (1, 2, 3)
        
        numerator = weighted_sq_error.sum(dim=spatial_dims)    # [B]
        denominator = weights.sum(dim=spatial_dims) + 1e-8     # [B]
        
        # Per-sample weighted MSE, then mean over batch
        per_sample_loss = numerator / denominator  # [B]
        
        return per_sample_loss.mean()  # Scalar

class CombinedWMSEGammaLoss(nn.Module):
    """
    Combined loss: Normalized WMSE + Gamma + Outside Penalty
    """
    
    def __init__(
        self,
        wmse_loss: nn.Module,
        gamma_loss: nn.Module,
        wmse_weight: float = 1.0,
        gamma_weight: float = 1.0,
        outside_weight: float = 0.5,
        outside_threshold: float = 0.05,
        momentum: float = 0.99,
        outside_weight_required: bool = False,
    ):
        super().__init__()
        self.wmse_loss = wmse_loss
        self.gamma_loss = gamma_loss
        self.wmse_weight = wmse_weight
        self.gamma_weight = gamma_weight
        self.outside_weight = outside_weight
        self.outside_threshold = outside_threshold
        self.momentum = momentum
        self.outside_weight_required = outside_weight_required
        
        # Running average - initialized to 0, will be set on first forward pass
        self.register_buffer('wmse_avg', torch.tensor(0.0))
    
    def outside_penalty(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        low_dose_mask = (y_true < self.outside_threshold).float()
        outside_dose = (y_pred * low_dose_mask).pow(2).mean()
        return outside_dose
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # WMSE loss
        loss_wmse = self.wmse_loss(y_pred, y_true)
        
        # Outside penalty
        if self.outside_weight_required:
            loss_outside = self.outside_penalty(y_pred, y_true)
        else:
            loss_outside = torch.tensor(0.0, device=y_pred.device)
        
        # Gamma loss
        loss_gamma = self.gamma_loss(y_pred, y_true)
        
        # Update WMSE running average
        with torch.no_grad():
            if self.wmse_avg == 0.0:
                # First forward pass: initialize to actual WMSE value
                self.wmse_avg = loss_wmse.clone()
            else:
                # Normal EMA update
                self.wmse_avg = self.momentum * self.wmse_avg + (1 - self.momentum) * loss_wmse
        
        # Normalize WMSE to ~1.0 scale
        loss_wmse_norm = loss_wmse / (self.wmse_avg + 1e-8)
        
        # Combined loss
        total_loss = (
            self.wmse_weight * loss_wmse_norm +
            self.gamma_weight * loss_gamma +
            self.outside_weight * loss_outside
        )
        
        return total_loss, {
            'wmse': loss_wmse.item(),
            'wmse_norm': loss_wmse_norm.item(),
            'gamma': loss_gamma.item(),
            'outside': loss_outside.item(),
        }
    
    def compute_pass_rate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        return self.gamma_loss.compute_pass_rate(y_pred, y_true)