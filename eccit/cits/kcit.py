"""KCIT: per-feature kernel conditional independence test.

Given features ``X`` and response ``Y`` we build centered RBF kernels for
``(X_j, X_{-j})`` and ``Y``, project out the conditioning set, and evaluate the
Hilbert–Schmidt trace statistic.  The implementation stays fully in PyTorch so
gradients propagate, uses a moment-matched Gaussian tail instead of the gamma
CDF for the null (PyTorch lacks `igamma` backward), and leaves the bootstrap
path optional/off by default.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch


TensorLike = Union[torch.Tensor, np.ndarray]

_RIDGE = 1e-3
_EIGEN_THRESH = 1e-5
_DEFAULT_BOOTSTRAP = 1000
_DTYPE = torch.float64


def _ensure_tensor(arr: TensorLike, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        tensor = arr.to(dtype=_DTYPE)
    else:
        tensor = torch.as_tensor(arr, dtype=_DTYPE)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _normalize_vector(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean()
    std = x.std(unbiased=False)
    if not isinstance(std, torch.Tensor):
        raise TypeError(f"std from vector expected tensor, got {type(std)!r}")
    std = std.clamp_min(1e-12)
    return (x - mean) / std


def _normalize_matrix(z: torch.Tensor) -> torch.Tensor:
    if z.numel() == 0:
        return z
    if z.ndim == 1:
        z = z.unsqueeze(1)
    mean = z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, keepdim=True, unbiased=False)
    if not isinstance(std, torch.Tensor):
        raise TypeError(f"std from matrix expected tensor, got {type(std)!r}")
    std = std.clamp_min(1e-12)
    return (z - mean) / std


def _median_pairwise_distance(data: torch.Tensor) -> torch.Tensor:
    if data.numel() == 0:
        base = torch.ones((), dtype=data.dtype, device=data.device)
        return (base + data.sum() * 0.0).clamp_min(1e-3)
    if data.ndim == 1:
        data = data.unsqueeze(1)
    if data.size(0) > 500:
        data = data[:500]

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


def _rbf_kernel(data: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    if data.ndim == 1:
        data = data.unsqueeze(1)
    dist_sq = torch.cdist(data, data, p=2) ** 2
    return torch.exp(-theta * dist_sq)


def _poly_kernel(data: torch.Tensor, degree: int, bias: float) -> torch.Tensor:
    if data.ndim == 1:
        data = data.unsqueeze(1)
    gram = data @ data.T + bias
    gram = gram.clamp(min=-1e6, max=1e6)
    return gram.pow(degree)


def _center_kernel(K: torch.Tensor) -> torch.Tensor:
    n = K.size(0)
    H = torch.eye(n, dtype=K.dtype, device=K.device) - (1.0 / n) * torch.ones((n, n), dtype=K.dtype, device=K.device)
    return H @ K @ H


def _project_operator(Kz: torch.Tensor) -> torch.Tensor:
    n = Kz.size(0)
    identity = torch.eye(n, dtype=Kz.dtype, device=Kz.device)
    regularised = Kz + _RIDGE * identity
    try:
        inv = torch.linalg.solve(regularised, identity)
    except RuntimeError:
        inv = torch.linalg.pinv(regularised)
    return identity - Kz @ inv


def _normal_tail_approx(statistic: torch.Tensor, uu_prod: torch.Tensor) -> torch.Tensor:
    uu_prod = torch.nan_to_num(uu_prod)
    statistic = torch.nan_to_num(statistic).clamp_min(0.0)
    mean_appr = torch.trace(uu_prod).clamp_min(1e-12)
    var_appr = (2.0 * torch.trace(uu_prod @ uu_prod)).clamp_min(1e-12)
    std_appr = var_appr.sqrt().clamp_min(1e-12)
    z = (statistic - mean_appr) / std_appr
    tail = 0.5 * torch.erfc(z / math.sqrt(2.0))
    return tail.clamp(min=1e-12, max=1.0 - 1e-12)


def _gamma_tail_value(statistic: torch.Tensor, uu_prod: torch.Tensor) -> torch.Tensor:
    uu_prod = torch.nan_to_num(uu_prod)
    statistic = torch.nan_to_num(statistic).clamp_min(0.0)
    trace_val = torch.trace(uu_prod)
    trace_sq = torch.trace(uu_prod @ uu_prod)
    if not torch.isfinite(trace_val):
        trace_val = torch.zeros((), dtype=uu_prod.dtype, device=uu_prod.device)
    if not torch.isfinite(trace_sq):
        trace_sq = torch.zeros((), dtype=uu_prod.dtype, device=uu_prod.device)
    mean_appr = trace_val.clamp_min(1e-12)
    var_appr = (2.0 * trace_sq).clamp_min(1e-12)
    if mean_appr <= 0 or var_appr <= 0:
        return (torch.ones((), dtype=uu_prod.dtype, device=uu_prod.device) + statistic * 0.0).clamp(min=1e-12, max=1.0 - 1e-12)
    k_appr = (mean_appr ** 2) / var_appr
    theta_appr = (var_appr / mean_appr).clamp_min(1e-12)
    with torch.no_grad():
        shape = k_appr.clamp_min(1e-12)
        rate = theta_appr.reciprocal().clamp_min(1e-12)
        gamma = torch.distributions.Gamma(concentration=shape, rate=rate)
        stat_det = torch.nan_to_num(statistic.detach()).clamp_min(0.0)
        cdf_val = gamma.cdf(stat_det)
    return (1.0 - cdf_val).clamp(min=1e-12, max=1.0 - 1e-12)


def _bootstrap_pvalue(statistic: torch.Tensor, eig_uu: torch.Tensor, num_bootstrap: int) -> torch.Tensor:
    thresh = _EIGEN_THRESH * torch.max(eig_uu)
    eig_uu = eig_uu[eig_uu > thresh]
    if eig_uu.numel() == 0:
        eig_uu = torch.as_tensor([1e-8], dtype=statistic.dtype, device=statistic.device)

    T = statistic.new_full((1,), float(num_bootstrap)).long().item()
    chi_samples = torch.randn((eig_uu.numel(), num_bootstrap), dtype=statistic.dtype, device=statistic.device) ** 2
    null_dstr = (eig_uu.unsqueeze(1) * chi_samples).sum(dim=0)
    p_val = (null_dstr > statistic).to(statistic.dtype).mean()
    return p_val.clamp(min=1e-12, max=1.0 - 1e-12)


def _kcit_core(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    bootstrap: bool,
    num_bootstrap: int,
    eigen_thresh: float,
    eigen_temp: Optional[float] = None,
    eigen_floor: float = 1e-4,
    kernel: str = "rbf",
    poly_degree: int = 2,
    poly_bias: float = 1.0,
) -> torch.Tensor:
    n = x.size(0)

    # Bandwidth heuristic
    if n <= 200:
        width = torch.tensor(1.2, dtype=x.dtype, device=x.device) + x.sum() * 0.0
    elif n > 1200:
        width = torch.tensor(0.7, dtype=x.dtype, device=x.device) + x.sum() * 0.0
    else:
        width = torch.tensor(0.4, dtype=x.dtype, device=x.device) + x.sum() * 0.0

    xy = torch.stack((x, y), dim=1)
    D = z.size(1) if z.ndim == 2 and z.numel() > 0 else 1

    if kernel == "rbf":
        width = _median_pairwise_distance(torch.cat((xy, z), dim=1)) if z.numel() > 0 else _median_pairwise_distance(xy)
        width = width.clamp_min(1e-3)
        theta = (width.pow(2) * float(D)).clamp_min(1e-6).reciprocal()
        Kx = _center_kernel(_rbf_kernel(torch.cat((x.unsqueeze(1), z / 2.0), dim=1), theta))
        Ky = _center_kernel(_rbf_kernel(y, theta))
        Kz = _center_kernel(_rbf_kernel(z, theta)) if z.numel() > 0 else torch.zeros_like(Kx)
    else:
        Kx = _center_kernel(_poly_kernel(torch.cat((x.unsqueeze(1), z / 2.0), dim=1), poly_degree, poly_bias))
        Ky = _center_kernel(_poly_kernel(y, poly_degree, poly_bias))
        Kz = _center_kernel(_poly_kernel(z, poly_degree, poly_bias)) if z.numel() > 0 else torch.zeros_like(Kx)

    Kx = torch.nan_to_num(Kx)
    Ky = torch.nan_to_num(Ky)
    Kz = torch.nan_to_num(Kz)

    P1 = _project_operator(Kz) if z.numel() > 0 else torch.eye(n, dtype=Kx.dtype, device=Kx.device)
    Kxz = P1 @ Kx @ P1.T
    Kyz = P1 @ Ky @ P1.T

    statistic = torch.trace(Kxz @ Kyz)
    statistic = torch.nan_to_num(statistic).clamp_min(0.0)

    # Symmetrise prior to eigen decomposition for numerical stability.
    Kxz_sym = 0.5 * (Kxz + Kxz.T)
    Kyz_sym = 0.5 * (Kyz + Kyz.T)
    jitter = 1e-6
    eye = torch.eye(n, dtype=Kxz.dtype, device=Kxz.device)
    success = False
    for _ in range(5):
        try:
            eig_vals_x, eig_vecs_x = torch.linalg.eigh(Kxz_sym + jitter * eye)
            eig_vals_y, eig_vecs_y = torch.linalg.eigh(Kyz_sym + jitter * eye)
            success = True
            break
        except RuntimeError:
            jitter *= 10
    if not success:
        eig_vals_x = torch.full((1,), 1e-8, dtype=x.dtype, device=x.device)
        eig_vecs_x = torch.eye(n, dtype=x.dtype, device=x.device)[:, :1]
        eig_vals_y = torch.full((1,), 1e-8, dtype=x.dtype, device=x.device)
        eig_vecs_y = torch.eye(n, dtype=x.dtype, device=x.device)[:, :1]

    if eig_vals_x.numel() == 0:
        eig_vals_x = torch.full((1,), 1e-8, dtype=x.dtype, device=x.device)
        eig_vecs_x = torch.eye(n, dtype=x.dtype, device=x.device)[:, :1]
    if eig_vals_y.numel() == 0:
        eig_vals_y = torch.full((1,), 1e-8, dtype=x.dtype, device=x.device)
        eig_vecs_y = torch.eye(n, dtype=x.dtype, device=x.device)[:, :1]

    eps = 1e-12
    max_val_x = eig_vals_x.abs().max().clamp_min(eps)
    max_val_y = eig_vals_y.abs().max().clamp_min(eps)

    if eigen_temp is not None and eigen_temp > 0:
        weights_x = torch.sigmoid(((eig_vals_x / max_val_x) - eigen_thresh) / eigen_temp)
        weights_y = torch.sigmoid(((eig_vals_y / max_val_y) - eigen_thresh) / eigen_temp)
        weights_x = weights_x * (1.0 - eigen_floor) + eigen_floor
        weights_y = weights_y * (1.0 - eigen_floor) + eigen_floor
    else:
        mask_x = eig_vals_x > max_val_x * eigen_thresh
        mask_y = eig_vals_y > max_val_y * eigen_thresh
        if not mask_x.any():
            idx = torch.argmax(eig_vals_x)
            mask_x = torch.zeros_like(eig_vals_x, dtype=torch.bool)
            mask_x[idx] = True
        if not mask_y.any():
            idx = torch.argmax(eig_vals_y)
            mask_y = torch.zeros_like(eig_vals_y, dtype=torch.bool)
            mask_y[idx] = True
        weights_x = mask_x.to(eig_vals_x.dtype) * (1.0 - eigen_floor) + eigen_floor
        weights_y = mask_y.to(eig_vals_y.dtype) * (1.0 - eigen_floor) + eigen_floor

    eig_vals_x = eig_vals_x * weights_x
    eig_vals_y = eig_vals_y * weights_y
    eig_vals_x = eig_vals_x.clamp_min(eps)
    eig_vals_y = eig_vals_y.clamp_min(eps)

    eiv_prodx = eig_vecs_x * eig_vals_x.sqrt()
    eiv_prody = eig_vecs_y * eig_vals_y.sqrt()

    uu = torch.einsum('ti,tj->tij', eiv_prodx, eiv_prody)
    uu = uu.reshape(n, -1)

    if uu.size(1) == 0:
        uu = torch.full((n, 1), 1e-8, dtype=x.dtype, device=x.device)

    uu_prod = uu.T @ uu if uu.size(1) <= n else uu @ uu.T

    if bootstrap:
        eig_uu = torch.linalg.eigvalsh(0.5 * (uu_prod + uu_prod.T))
        eig_uu = eig_uu[eig_uu > 0]
        if eig_uu.numel() == 0:
            eig_uu = torch.as_tensor([1e-8], dtype=statistic.dtype, device=statistic.device)
        return _bootstrap_pvalue(statistic, eig_uu, num_bootstrap)

    uu_prod = torch.nan_to_num(uu_prod)

    p_gamma = _gamma_tail_value(statistic, uu_prod)
    p_smooth = _normal_tail_approx(statistic, uu_prod)
    return p_gamma + (p_smooth - p_smooth.detach())


def kcit_pvals(
    X: TensorLike,
    Y: TensorLike,
    *,
    bootstrap: bool = False,
    num_bootstrap: int = _DEFAULT_BOOTSTRAP,
    eigen_thresh: float = _EIGEN_THRESH,
    eigen_temp: Optional[float] = None,
    eigen_floor: float = 1e-4,
    subsample_size: Optional[int] = None,
    seed: Optional[int] = None,
    kernel: str = "rbf",
    poly_degree: int = 2,
    poly_bias: float = 1.0,
    to_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """Compute KCIT p-values for every feature in ``X``.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        Y: Response vector of shape ``(n_samples,)`` or ``(n_samples, 1)``.
        bootstrap: Whether to estimate the null distribution via bootstrap.
        num_bootstrap: Number of bootstrap draws when ``bootstrap=True``.
        eigen_thresh: Eigenvalue threshold for numerical stability.
        to_numpy: Return a NumPy array instead of a tensor when ``True``.
    """

    if isinstance(X, torch.Tensor):
        device = X.device
        orig_dtype = X.dtype
    elif isinstance(Y, torch.Tensor):
        device = Y.device
        orig_dtype = Y.dtype
    else:
        device = torch.device('cpu')
        orig_dtype = torch.float32

    X_t = _ensure_tensor(X, device=device)
    Y_t = _ensure_tensor(Y, device=device).reshape(-1)

    if X_t.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features)")
    if Y_t.shape[0] != X_t.shape[0]:
        raise ValueError("X and Y must share the same number of rows")

    n, p = X_t.shape
    pvals = []

    kernel = kernel.lower()
    if kernel not in {"rbf", "poly"}:
        raise ValueError("kernel must be 'rbf' or 'poly'")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    kernel = kernel.lower()
    if kernel not in {"rbf", "poly"}:
        raise ValueError("kernel must be 'rbf' or 'poly'")

    for j in range(p):
        X_j = X_t
        Y_vec = Y_t
        if subsample_size is not None and subsample_size < n:
            idx = torch.randperm(n, device=device, generator=generator)[:subsample_size]
            X_j = X_j[idx]
            Y_vec = Y_vec[idx]

        Y_norm = _normalize_vector(Y_vec)
        x_j = _normalize_vector(X_j[:, j])
        if p == 1:
            z = X_j.new_zeros((X_j.size(0), 0))
        else:
            z = torch.cat((X_j[:, :j], X_j[:, j + 1 :]), dim=1)
        z_norm = _normalize_matrix(z)

        p_val = _kcit_core(
            x_j,
            Y_norm,
            z_norm,
            bootstrap=bootstrap,
            num_bootstrap=num_bootstrap,
            eigen_thresh=eigen_thresh,
            eigen_temp=eigen_temp,
            eigen_floor=eigen_floor,
            kernel=kernel,
            poly_degree=poly_degree,
            poly_bias=poly_bias,
        )
        if not torch.isfinite(p_val):
            p_val = (torch.ones((), dtype=x_j.dtype, device=x_j.device) + x_j.sum() * 0.0)
        pvals.append(p_val)

    pvals_t = torch.stack(pvals).clamp(1e-12, 1.0 - 1e-12)

    if to_numpy:
        return pvals_t.to(dtype=orig_dtype).cpu().numpy()
    return pvals_t.to(dtype=orig_dtype)


__all__ = ["kcit_pvals"]
