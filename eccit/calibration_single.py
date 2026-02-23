"""Calibration utilities for single-experiment settings."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim

from eccit.calibration import miscal_type1, YGeneratorSingle
from eccit.cits.gcm import gcm_single
from eccit.cits.hrt import hrt_single


def bootstrap_indices(n_samples: int, n_subsample: int) -> np.ndarray:
    return np.random.choice(n_samples, size=n_subsample, replace=True)


def draw_pvalues(
    adversary: YGeneratorSingle,
    Z_batch: torch.Tensor,
    X_batch: torch.Tensor,
    *,
    num_draws: int,
    test: str,
    gcm_kwargs: Dict[str, object],
    hrt_kwargs: Dict[str, object],
) -> torch.Tensor:
    values = []
    for _ in range(num_draws):
        Y_batch, _ = adversary(Z_batch)
        if test.lower() == "hrt":
            p_val = hrt_single(Z_batch, X_batch, Y_batch, **hrt_kwargs)
        else:
            p_val = gcm_single(Z_batch, X_batch, Y_batch, to_numpy=False, **gcm_kwargs)
        if isinstance(p_val, np.ndarray):
            p_val = torch.from_numpy(p_val)
        elif not torch.is_tensor(p_val):
            p_val = torch.tensor(float(p_val))
        p_val = p_val.to(dtype=Z_batch.dtype)
        values.append(p_val)
    return torch.stack(values)


def calibrate_single_experiment(
    Z,
    X,
    *,
    alpha: float = 0.2,
    tau: float = 0.02,
    num_epochs: int = 200,
    subsample_frac: float = 0.8,
    order_adv: int = 1,
    order_test: int = 1,
    use_linear: bool = True,
    weight_lr: float = 5e-4,
    draws_per_epoch: int = 10,
    bootstrap_period: int = 5,
    test: str = "gcm",
    gcm_kwargs: Optional[Dict[str, object]] = None,
    hrt_kwargs: Optional[Dict[str, object]] = None,
    freeze_adversary: bool = False,
    debug_gradients: bool = False,
    seed: Optional[int] = None,
    cutoff_ema: float = 0.2,
) -> Dict[str, object]:
    """Calibrate a conditional test by adversarially inflating type-I error."""

    dtype = torch.get_default_dtype()

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    Z_tensor = torch.as_tensor(Z, dtype=dtype)
    if Z_tensor.ndim == 1:
        Z_tensor = Z_tensor.view(-1, 1)
    X_tensor = torch.as_tensor(X, dtype=dtype).view(-1)

    n_samples = Z_tensor.shape[0]
    subsample_frac = float(subsample_frac)
    n_subsample = max(1, int(np.round(n_samples * subsample_frac)))

    gcm_kwargs = dict(gcm_kwargs or {})
    if test.lower() == "gcm" and not gcm_kwargs and order_test > 1:
        gcm_kwargs.update({"y_estimator": "mlp", "x_estimator": "mlp"})

    hrt_kwargs = dict(hrt_kwargs or {})
    if test.lower() == "hrt":
        num_epochs = max(1, num_epochs // 20)
        hrt_kwargs.setdefault("estimator_type", "linear" if use_linear or order_test == 1 else "mlp")

    adversary = YGeneratorSingle(
        n_features=Z_tensor.shape[1],
        order=order_adv,
    ).to(dtype=dtype)

    if freeze_adversary:
        for param in adversary.parameters():
            param.requires_grad_(False)

    adv_params = [p for p in adversary.parameters() if p.requires_grad]
    opt_adv = optim.Adam(adv_params, lr=weight_lr) if adv_params else None

    alpha_target = torch.tensor(float(alpha), dtype=dtype)
    cutoff_value = torch.tensor(float(alpha), dtype=dtype)
    min_cutoff = torch.tensor(1e-6, dtype=dtype)

    ema_beta = float(np.clip(float(cutoff_ema), 0.0, 1.0))

    def clamp_cutoff(value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(value, min=min_cutoff, max=alpha_target)

    idx_cache = None

    for epoch in range(num_epochs):
        if (epoch % bootstrap_period) == 0 or idx_cache is None:
            idx_cache = bootstrap_indices(n_samples, n_subsample)
        Z_batch = Z_tensor[idx_cache]
        X_batch = X_tensor[idx_cache]

        pvals_epoch = draw_pvalues(
            adversary,
            Z_batch,
            X_batch,
            num_draws=draws_per_epoch,
            test=test,
            gcm_kwargs=gcm_kwargs,
            hrt_kwargs=hrt_kwargs,
        )
        cutoff_tensor = clamp_cutoff(cutoff_value)
        miscal = miscal_type1(pvals_epoch, cutoff_tensor, tau=tau)
        loss_adv = -miscal

        if opt_adv is not None:
            opt_adv.zero_grad()
            loss_adv.backward()

            if debug_gradients:
                grad_max = 0.0
                for param in adversary.parameters():
                    if param.grad is not None:
                        grad_max = max(grad_max, float(param.grad.detach().abs().max().cpu().item()))
                print(f"Epoch {epoch + 1}: max|∇adv|={grad_max:.3e}")

            opt_adv.step()

        with torch.no_grad():
            pvals_cpu = pvals_epoch.detach().cpu().numpy()
            sorted_p = np.sort(pvals_cpu)
            m = sorted_p.shape[0]
            k = int(np.floor(alpha * m))
            if k <= 0:
                c_hat = float(min_cutoff.item())
            else:
                c_hat = float(sorted_p[k - 1])
            c_hat = min(c_hat, float(alpha_target.item()))
            c_hat_tensor = torch.tensor(c_hat)
            if ema_beta > 0.0:
                cutoff_value = (1.0 - ema_beta) * cutoff_value + ema_beta * c_hat_tensor
            else:
                cutoff_value = c_hat_tensor
            cutoff_value = clamp_cutoff(cutoff_value)

        if (epoch + 1) % 10 == 0:
            rate_value = float(miscal.detach().cpu().item())
            print(
                f"Epoch {epoch + 1} | rate {rate_value:.4f} | cutoff {float(cutoff_value.cpu().item()):.4f}"
            )

    return {
        "baseline_cutoff": float(alpha),
        "calibrated_cutoff": float(clamp_cutoff(cutoff_value).clamp(min=0.0).detach().cpu().item()),
    }
