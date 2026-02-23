import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eccit.utils.estimators import MLPRegressor


### CALIBRATION METRICS ###
def miscal_area(pvals, cutoff=0.2):
    """
    Trapezoid approximation of F(u) - u 
    where F is the uniform CDF of pvals

    This is identical to the Wasserstein distance.
    """
    p_sorted, _ = torch.sort(pvals)
    m = p_sorted.numel() # pytorch len() basically
    if m == 0:
        # nothing to compare so zero area
        return torch.zeros((), device=pvals.device, requires_grad=True)
    
    p_trunc = p_sorted[p_sorted <= cutoff]
    k = p_trunc.numel()

    cdf = torch.arange(1, k + 1, device=p_sorted.device) / m  # uniform CDF at each p_i
    g = cdf - p_trunc  # g(u) = F(u) – u

    # u = [0, p1, p2, ..., pk, cutoff]
    u = torch.cat([torch.zeros(1, device=pvals.device),
                   p_trunc,
                   torch.tensor([cutoff], device=pvals.device)])
    # g_pad = [0, g(p1), ..., g(pk), F(cutoff) - cutoff]
    end = (k / m) - cutoff
    g_pad = torch.cat([torch.zeros(1, device=pvals.device),
                       g,
                       torch.tensor([end], device=pvals.device)])

    # widths = u_i - u_i-1
    widths = u[1:] - u[:-1]

    # area = sum: (u_i - u_i+1) * [ g(u_i+1) + g(u_i) ] / 2  
    area = torch.sum(widths * (g_pad[:-1] + g_pad[1:]) * 0.5)
    return area




def miscal_fdp(pvals, null_mask, alpha=0.2, tau=0.02, eps=1e-8):
    """
    hard value soft gradient surrogate for FDP under BH.

    High-level idea:
      - We want FDR-BH for the value but also need gradients for learning.
      - We combine a hard BH FDP (value, no gradients) with a differentiable soft step-up BH FDP
        (for gradients)

    """
    m = pvals.numel()
    if m == 0:
        return torch.zeros((), device=pvals.device, dtype=pvals.dtype)

    device, dtype = pvals.device, pvals.dtype
    is_null = null_mask.to(dtype)

    # ------------------------------------------------------------------
    # HARD BH (value only no gradients)
    # ------------------------------------------------------------------

    # 1) Sort p to get order stats p_(i)
    p_sorted, _ = torch.sort(pvals)
    ranks = torch.arange(1, m + 1, device=device, dtype=dtype)

    # 2) Build BH line
    bh_line = alpha * ranks / m

    # 3) Find k_hard = max i such that p_(i) <= alpha * i / m (or 0)
    sat = p_sorted <= bh_line
    if sat.any():
        # largest i where condition holds (ranks start at 1, so +1)
        k_hard = int(torch.nonzero(sat, as_tuple=False)[-1, 0].item() + 1)
    else:
        k_hard = 0

    # 4) Set hard cutoff t_hard
    t_hard = alpha * k_hard / m

    # 5) Hard rejections r_hard
    reject_hard = (pvals <= t_hard).to(dtype)
    total_hard = reject_hard.sum()

    # 6) Compute FDP_hard
    if total_hard.item() == 0:
        fdp_hard = torch.zeros((), device=device, dtype=dtype)
    else:
        
        false_pos_hard = (reject_hard * is_null).sum()
        fdp_hard = false_pos_hard / (total_hard + eps)

    # Stop gradients through the hard value
    fdp_hard = fdp_hard.detach()

    # ------------------------------------------------------------------
    # SOFT BH (gradients)
    # ------------------------------------------------------------------
    
    # 1) Soft under-the-line indicators: s_i = sigmoid((alpha*i/m - p_(i)) / tau)
    #    Use the same sorted p_(i) and BH line.
    s = torch.sigmoid((bh_line - p_sorted) / tau)

    # 2) Soft discovery count: k_soft = sum_i s_i
    k_soft = s.sum()
    t_soft = alpha * k_soft / m

    # 3) Soft rejections per hypothesis: reject_soft
    reject_soft = torch.sigmoid((t_soft - pvals) / tau)
    total_soft = reject_soft.sum()
    false_pos_soft = (reject_soft * is_null).sum()

    # 4) Compute FDP_soft
    fdp_soft = false_pos_soft / (total_soft + eps)

    # ------------------------------------------------------------------
    # Value equals FDP under hard BH, gradients come from the soft path.
    # ------------------------------------------------------------------
    out = fdp_hard + (fdp_soft - fdp_soft.detach())
    return out


def miscal_type1(pvals: torch.Tensor, alpha=0.2, tau=0.02):
    """Soft surrogate for the type-I error at the provided cutoff."""
    if pvals.numel() == 0:
        return torch.zeros((), device=pvals.device, dtype=pvals.dtype)

    if torch.is_tensor(alpha):
        alpha_tensor = alpha.to(device=pvals.device, dtype=pvals.dtype)
    else:
        alpha_tensor = torch.tensor(float(alpha), device=pvals.device, dtype=pvals.dtype)

    hard_rate = (pvals <= alpha_tensor).to(pvals.dtype).mean()
    soft_rate = torch.sigmoid((alpha_tensor - pvals) / tau).mean()

    surrogate = hard_rate.detach() + (soft_rate - soft_rate.detach())
    return surrogate


def miscal_type1_multi(
    pvals: torch.Tensor,
    null_mask: torch.Tensor,
    *,
    alpha=0.2,
    tau=0.02,
) -> torch.Tensor:
    """Type-I error surrogate computed only on null features."""
    if pvals.numel() == 0:
        return torch.zeros((), device=pvals.device, dtype=pvals.dtype)
    if null_mask.numel() == 0:
        return torch.zeros((), device=pvals.device, dtype=pvals.dtype)
    null_mask = null_mask.to(dtype=torch.bool)
    if not torch.any(null_mask):
        return torch.zeros((), device=pvals.device, dtype=pvals.dtype)
    null_pvals = pvals[null_mask]
    return miscal_type1(null_pvals, alpha=alpha, tau=tau)


### GENERATORS ###
class YGenerator(nn.Module):
    """
    Generate synthetic responses Y with learnable feature masks.

    Given input X and a binary mask m:
      1) Masks X elementwise by m to select active features (Xm).
      2) Uses masked features to predict conditional mean.
      3) Samples Y ~ Normal(mean, stdev).

    Args:
        n_features: Number of input features
        order: 1 for linear model, >1 for MLP
        stdev: Fixed standard deviation for Gaussian noise
        hidden: Hidden layer size for MLP (when order > 1)
    """
    def __init__(self, n_features, order=1, stdev=1.0, hidden=64):
        super().__init__()
        self.order = order
        self.stdev = float(stdev)
        self.hidden = int(hidden)
        self.input_dim = n_features

        # Learnable mask logits (soft mask via sigmoid)
        self.mask_logits = nn.Parameter(torch.full((n_features,), 0.0))

        # Mean prediction model
        if order == 1:
            self.mean_model = nn.Linear(n_features, 1, bias=False)
        else:
            self.mean_model = MLPRegressor(n_features, hidden=hidden)

    def weight_snapshot(self):
        """Return flattened weights from first layer for diagnostics."""
        with torch.no_grad():
            if self.order == 1:
                weight = self.mean_model.weight
            else:
                if hasattr(self.mean_model, "fc1"):
                    weight = self.mean_model.fc1.weight
                else:
                    weight = getattr(self.mean_model, "net")[0].weight
            return weight.detach().cpu().numpy().reshape(-1)

    ### HELPERS ###
    @torch.no_grad()
    def mask_probs_sigmoid(self):
        """
        Return per-feature inclusion probabilities in [0,1].
        """
        return torch.sigmoid(self.mask_logits)

    @torch.no_grad()
    def deterministic_mask(self, threshold=0.5, topk=None):
        """
        Return a hard binary mask derived from current probabilities.
        """
        probs = self.mask_probs_sigmoid()  # π in [0,1]
        if topk is not None:
            k = int(topk)
            if k <= 0:
                return torch.zeros_like(probs)
            idx = torch.topk(probs, k=k, largest=True).indices
            m = torch.zeros_like(probs)
            m[idx] = 1.0
            return m
        return (probs >= threshold).float()

    def get_mu(self, features):
        """Compute mean prediction from masked features."""
        mu = self.mean_model(features)
        if self.order == 1:
            mu = mu.squeeze(1)
        return mu

    def forward(self, X):
        """
        Sample a hard mask with Gumbel-softmax and produce Y with Gaussian noise.
        Returns: (Y, mask)
        """
        # Sample the mask: P(m=1) = sigmoid(mask_logits)
        logits = torch.stack([torch.zeros_like(self.mask_logits), self.mask_logits], dim=1)
        m = F.gumbel_softmax(logits, tau=1.0, hard=True)[:, 1]

        # Apply mask to features and predict
        Xm = X * m.float()
        mu = self.get_mu(Xm)
        Y = mu + self.stdev * torch.randn_like(mu)

        return Y, m

    def forward_with_mask(self, X, m_binary):
        """Generate Y using a provided binary mask."""
        Xm = X * m_binary.float()
        mu = self.get_mu(Xm)
        Y = mu + self.stdev * torch.randn_like(mu)
        return Y, m_binary


class YGeneratorSingle(nn.Module):
    """Adversarial mean model for single-feature calibration experiments."""

    def __init__(self, n_features: int, order: int = 1, stdev: float = 1.0, hidden: int = 64) -> None:
        super().__init__()
        self.order = int(order)
        self.stdev = float(stdev)
        self.input_dim = int(n_features)
        self.hidden = int(hidden)

        if self.order == 1:
            self.mean_model = nn.Linear(self.input_dim, 1, bias=False)
        else:
            self.mean_model = MLPRegressor(self.input_dim, hidden=self.hidden)

    def weight_snapshot(self) -> np.ndarray:
        with torch.no_grad():
            if self.order == 1:
                weight = self.mean_model.weight
            else:
                if hasattr(self.mean_model, "net"):
                    weight = self.mean_model.net[0].weight
                else:
                    weight = getattr(self.mean_model, "fc1").weight  # type: ignore[attr-defined]
            return weight.detach().cpu().numpy().reshape(-1)

    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mean_model(Z)
        if mu.ndim > 1:
            mu = mu.squeeze(-1)
        noise = torch.randn_like(mu)
        Y = mu + self.stdev * noise
        return Y, mu
