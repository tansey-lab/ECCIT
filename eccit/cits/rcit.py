"""RCIT: Randomized Conditional Independence Test (torch implementation).

This module mirrors the per-feature interface used elsewhere in the project:
given features ``X`` and response ``Y`` it returns one p-value per feature,
testing ``X_j`` ⟂ ``Y`` | ``X_{-j}``.  The statistic follows the RCIT recipe:
random Fourier features approximate the Gaussian kernels, linear projections
remove the conditioning set, and the residual covariance energy is evaluated.
We keep all operations in PyTorch so gradients can propagate to ``Y``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


TensorLike = Union[torch.Tensor, np.ndarray]

_DEFAULT_NUM_F = 60
_DEFAULT_NUM_F2 = 10
_MAX_MEDIAN_SAMPLES = 500


def _ensure_tensor(arr: TensorLike, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        tensor = arr
    else:
        tensor = torch.as_tensor(arr)
    if device is not None:
        tensor = tensor.to(device)
    return tensor.to(dtype=torch.float64)


def _normalize(mat: torch.Tensor) -> torch.Tensor:
    mean = mat.mean(dim=0, keepdim=True)
    std = mat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-12)
    return (mat - mean) / std


def _median_pairwise_distance(data: torch.Tensor) -> torch.Tensor:
    if data.numel() == 0:
        base = torch.ones((), dtype=data.dtype, device=data.device)
        return (base + data.sum() * 0.0).clamp_min(1e-3)
    if data.ndim == 1:
        data = data.unsqueeze(1)
    if data.size(0) > _MAX_MEDIAN_SAMPLES:
        data = data[:_MAX_MEDIAN_SAMPLES]
    n = data.size(0)
    if n <= 1:
        fallback = data.abs().mean()
        return fallback.clamp_min(1e-3)
    dist_matrix = torch.cdist(data, data, p=2)
    tri = torch.triu_indices(n, n, offset=1, device=data.device)
    dists = dist_matrix[tri[0], tri[1]]
    dists_sorted, _ = torch.sort(dists)
    m = dists_sorted.numel()
    if m == 0:
        fallback = data.abs().mean()
        return fallback.clamp_min(1e-3)
    if m % 2 == 1:
        median = dists_sorted[m // 2]
    else:
        median = 0.5 * (dists_sorted[m // 2 - 1] + dists_sorted[m // 2])
    return median.clamp_min(1e-3)


def _random_fourier_features(
    X: torch.Tensor,
    num_features: int,
    sigma: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_params: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    X = X.double()
    d = X.size(1)
    sigma = sigma.clamp_min(1e-6)
    scale = 1.0 / sigma

    if params is None:
        W = torch.randn(
            d,
            num_features,
            dtype=X.dtype,
            device=X.device,
            generator=generator,
        ) * scale
        b = 2.0 * math.pi * torch.rand(
            num_features,
            dtype=X.dtype,
            device=X.device,
            generator=generator,
        )
    else:
        W, b = params
        W = W.to(dtype=X.dtype, device=X.device)
        b = b.to(dtype=X.dtype, device=X.device)

    projection = X @ W + b
    feat = math.sqrt(2.0 / num_features) * torch.cos(projection)
    if return_params:
        return feat, (W, b)
    return feat


def _poly_features(
    X: torch.Tensor,
    degree: int = 2,
    bias: float = 0.0,
) -> torch.Tensor:
    X = X.double()
    feats = [X]
    if degree >= 2 and X.size(1) > 0:
        d = X.size(1)
        tri = torch.triu_indices(d, d, device=X.device)
        prod = X[:, tri[0]] * X[:, tri[1]]
        if prod.numel() > 0:
            scale = torch.ones(tri.size(1), dtype=X.dtype, device=X.device)
            scale[tri[0] != tri[1]] = math.sqrt(2.0)
            prod = prod * scale
        feats.append(prod)
    if bias != 0.0:
        bias_col = torch.full((X.size(0), 1), float(bias), dtype=X.dtype, device=X.device)
        feats.append(bias_col)
    return torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]


def _center(mat: torch.Tensor) -> torch.Tensor:
    return mat - mat.mean(dim=0, keepdim=True)


def _covariance(a: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    if b is None:
        b = a
    a_c, b_c = _center(a), _center(b)
    n = a.size(0)
    return (a_c.T @ b_c) / (n - 1 if n > 1 else 1)


def _gamma_tail(statistic: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
    eigenvalues = torch.nan_to_num(eigenvalues)
    statistic = torch.nan_to_num(statistic).clamp_min(0.0)
    eig_vals = eigenvalues
    eig_pos = eig_vals.clamp_min(0.0)
    mean_null = eig_pos.sum()
    var_null = 2.0 * eig_pos.pow(2).sum()

    if mean_null <= 0 or var_null <= 0:
        return (torch.ones((), dtype=statistic.dtype, device=statistic.device) + statistic * 0.0).clamp(min=1e-12, max=1.0 - 1e-12)

    mean_null = mean_null.clamp_min(1e-12)
    var_null = var_null.clamp_min(1e-12)
    shape = (mean_null ** 2) / var_null
    scale = var_null / mean_null
    with torch.no_grad():
        shape = shape.clamp_min(1e-12)
        rate = scale.reciprocal().clamp_min(1e-12)
        gamma = torch.distributions.Gamma(concentration=shape, rate=rate)
        stat_det = torch.nan_to_num(statistic.detach()).clamp_min(0.0)
        tail_val = 1.0 - gamma.cdf(stat_det)
    return tail_val.clamp(min=1e-12, max=1.0 - 1e-12)


def _normal_tail(statistic: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
    eigenvalues = torch.nan_to_num(eigenvalues)
    statistic = torch.nan_to_num(statistic).clamp_min(0.0)
    eig_vals = eigenvalues
    eig_pos = eig_vals.clamp_min(0.0)
    mean_null = eig_pos.sum()
    var_null = 2.0 * eig_pos.pow(2).sum()
    if mean_null <= 0 or var_null <= 0:
        return (torch.ones((), dtype=statistic.dtype, device=statistic.device) + statistic * 0.0).clamp(min=1e-12, max=1.0 - 1e-12)
    mean_null = mean_null.clamp_min(1e-12)
    var_null = var_null.clamp_min(1e-12)
    std_null = var_null.sqrt().clamp_min(1e-12)
    z = (statistic - mean_null) / std_null
    tail = 0.5 * torch.erfc(z / math.sqrt(2.0))
    return tail.clamp(min=1e-12, max=1.0 - 1e-12)


def rcit_pvals(
    X: TensorLike,
    Y: TensorLike,
    *,
    num_f: int = _DEFAULT_NUM_F,
    num_f2: int = _DEFAULT_NUM_F2,
    approx: str = "gamma",
    seed: Optional[int] = None,
    redraw: bool = True,
    feature_cache: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    kernel: str = "rbf",
    poly_degree: int = 2,
    poly_bias: float = 0.0,
    to_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    if approx not in {"gamma", "normal"}:
        raise ValueError("approx must be 'gamma' or 'normal'")
    kernel = kernel.lower()
    if kernel not in {"rbf", "poly"}:
        raise ValueError("kernel must be 'rbf' or 'poly'")

    if isinstance(X, torch.Tensor):
        device = X.device
        out_dtype = X.dtype
    elif isinstance(Y, torch.Tensor):
        device = Y.device
        out_dtype = Y.dtype
    else:
        device = torch.device("cpu")
        out_dtype = torch.float32

    X_t = _ensure_tensor(X, device=device)
    Y_t = _ensure_tensor(Y, device=device).reshape(-1, 1)

    if X_t.ndim != 2:
        raise ValueError("X must be 2-dimensional (n_samples, n_features)")
    if Y_t.shape[0] != X_t.shape[0]:
        raise ValueError("X and Y must align on n_samples")

    n, p = X_t.shape
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    cache = feature_cache if (feature_cache is not None and not redraw and kernel == "rbf") else None

    pvals = []
    Y_norm = _normalize(Y_t)

    for j in range(p):
        x_j = _normalize(X_t[:, j : j + 1])
        if p == 1:
            z = X_t.new_zeros((n, 0))
        else:
            z = torch.cat((X_t[:, :j], X_t[:, j + 1 :]), dim=1)
        z = z[:, z.std(dim=0, unbiased=False) > 0]
        has_cond = z.numel() > 0

        z_norm = _normalize(z) if has_cond else z
        y_aug = torch.cat((Y_norm, z_norm), dim=1) if has_cond else Y_norm

        if kernel == "rbf":
            if has_cond:
                sigma_z = _median_pairwise_distance(z_norm)
            else:
                sigma_z = (torch.ones((), dtype=X_t.dtype, device=device) + X_t.sum() * 0.0).clamp_min(1e-3)
            sigma_x = _median_pairwise_distance(x_j)
            sigma_y = _median_pairwise_distance(y_aug)

            def sample_features(data: torch.Tensor, num_features: int, sigma: torch.Tensor, kind: str) -> torch.Tensor:
                if cache is None:
                    return _random_fourier_features(
                        data,
                        num_features=num_features,
                        sigma=sigma,
                        generator=generator,
                    )
                key = (kernel, kind, j, data.shape[1], num_features)
                params = cache.get(key)
                feat, params_new = _random_fourier_features(
                    data,
                    num_features=num_features,
                    sigma=sigma,
                    generator=generator,
                    params=params,
                    return_params=True,
                )
                cache[key] = params_new
                return feat
        else:
            shared_sigma = (torch.ones((), dtype=X_t.dtype, device=device) + X_t.sum() * 0.0).clamp_min(1e-3)
            sigma_z = sigma_x = sigma_y = shared_sigma

            def sample_features(data: torch.Tensor, num_features: int, sigma: torch.Tensor, kind: str) -> torch.Tensor:  # noqa: ARG001
                if data.numel() == 0:
                    return torch.zeros((data.size(0), 0), dtype=data.dtype, device=data.device)
                return _poly_features(data, degree=poly_degree, bias=poly_bias)

        if has_cond:
            f_z = sample_features(z_norm, num_features=num_f, sigma=sigma_z, kind='z')
        else:
            f_z = None
        f_x = sample_features(x_j, num_features=num_f2, sigma=sigma_x, kind='x')
        f_y = sample_features(y_aug, num_features=num_f2, sigma=sigma_y, kind='y')

        f_x = _normalize(f_x)
        f_y = _normalize(f_y)
        if has_cond:
            f_z = _normalize(f_z)

        if has_cond:
            Cxy = _covariance(f_x, f_y)
            Cxz = _covariance(f_x, f_z)
            Czy = _covariance(f_z, f_y)
            Czz = _covariance(f_z)

            ridge = 1e-6 * torch.eye(Czz.size(0), dtype=Czz.dtype, device=device)
            chol = torch.linalg.cholesky(Czz + ridge)
            Czz_inv = torch.cholesky_inverse(chol)

            Cxy_z = Cxy - Cxz @ Czz_inv @ Czy
            statistic = (float(n) * torch.sum(Cxy_z.pow(2))).clamp_min(0.0)

            z_i_Czz = _center(f_z) @ Czz_inv
            e_x_z = z_i_Czz @ Cxz.T
            e_y_z = z_i_Czz @ Czy
            res_x = _center(f_x) - e_x_z
            res_y = _center(f_y) - e_y_z
        else:
            Cxy = _covariance(f_x, f_y)
            statistic = (float(n) * torch.sum(Cxy.pow(2))).clamp_min(0.0)
            res_x = _center(f_x)
            res_y = _center(f_y)

        res_prod = (res_x[:, :, None] * res_y[:, None, :]).reshape(n, -1)
        res_prod = res_prod - res_prod.mean(dim=0, keepdim=True)
        Cov = (res_prod.T @ res_prod) / float(n)
        Cov = 0.5 * (Cov + Cov.T)
        jitter = 1e-6 * torch.eye(Cov.size(0), dtype=Cov.dtype, device=Cov.device)
        Cov = Cov + jitter

        try:
            eigvals = torch.linalg.eigvalsh(Cov)
        except RuntimeError:
            jitter = 1e-5 * torch.eye(Cov.size(0), dtype=Cov.dtype, device=Cov.device)
            eigvals = torch.linalg.eigvalsh(Cov + jitter)

        stat_tensor = torch.as_tensor(statistic, dtype=Cov.dtype, device=device).clamp_min(0.0)
        if approx == "gamma":
            p_gamma = _gamma_tail(stat_tensor, eigvals)
            p_smooth = _normal_tail(stat_tensor, eigvals)
            p_val = p_gamma + (p_smooth - p_smooth.detach())
        else:
            p_val = _normal_tail(stat_tensor, eigvals)

        pvals.append(p_val.clamp(1e-12, 1.0 - 1e-12))

    pvals_t = torch.stack(pvals).to(dtype=out_dtype)
    if to_numpy:
        return pvals_t.cpu().numpy()
    return pvals_t


__all__ = ["rcit_pvals"]
