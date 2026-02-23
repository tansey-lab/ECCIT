"""Generalised GCM with shared estimator helpers."""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Optional

from eccit.utils.estimators import predict_regression


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def split_X(X: torch.Tensor, j: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cat([X[:, :j], X[:, j + 1 :]], dim=1), X[:, j]


def gcm(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    y_estimator: str = "linear",
    y_estimator_params: Optional[Dict[str, object]] = None,
    x_estimator: str = "linear",
    x_estimator_params: Optional[Dict[str, object]] = None,
    to_numpy: bool = False,
) -> torch.Tensor:
    device, dtype = X.device, X.dtype
    n, p = X.shape
    pvals = torch.empty(p, dtype=dtype, device=device)

    for j in range(p):
        X_mj, xj = split_X(X, j)

        mu_x = predict_regression(
            X_mj,
            xj,
            X_mj,
            estimator=x_estimator,
            params=x_estimator_params,
        )
        resid_x = xj - mu_x

        mu_y = predict_regression(
            X_mj,
            Y,
            X_mj,
            estimator=y_estimator,
            params=y_estimator_params,
        )
        resid_y = Y - mu_y

        R = resid_x * resid_y
        mean_stat = R.mean()
        var_stat = R.var()
        T = mean_stat * torch.sqrt(torch.tensor(float(n), device=device)) / torch.sqrt(var_stat + 1e-8)
        pvals[j] = 2.0 * (1.0 - normal_cdf(torch.abs(T)))

    pvals = pvals.clamp(1e-12, 1.0 - 1e-12)
    if to_numpy:
        return pvals.cpu().numpy()
    return pvals


def gcm_binary(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    max_iter: int = 25,
    tol: float = 1e-6,
    to_numpy: bool = False,
) -> torch.Tensor:
    device, dtype = X.device, X.dtype
    n, p = X.shape
    pvals = torch.empty(p, dtype=dtype, device=device)

    Y = Y.clamp(0.0, 1.0)

    for j in range(p):
        X_mj, xj = split_X(X, j)

        XtX = X_mj.T @ X_mj
        XtX = XtX + 1e-3 * torch.eye(XtX.shape[0], device=device, dtype=dtype)
        try:
            beta_x = torch.linalg.solve(XtX, X_mj.T @ xj).squeeze(-1)
        except RuntimeError:
            beta_x = torch.linalg.lstsq(X_mj, xj.unsqueeze(1)).solution.squeeze(-1)
        mu_x = X_mj @ beta_x
        resid_x = xj - mu_x

        beta = torch.zeros(X_mj.shape[1] + 1, device=device, dtype=dtype)
        X_aug = torch.cat([X_mj, torch.ones(n, 1, device=device, dtype=dtype)], dim=1)
        for _ in range(max_iter):
            logits = X_aug @ beta
            probs = torch.sigmoid(logits)
            W = (probs * (1 - probs)).clamp_min(1e-6)
            z = logits + (Y - probs) / W
            WX = X_aug * W.unsqueeze(1)
            XtWX = X_aug.T @ WX + 1e-3 * torch.eye(X_aug.shape[1], device=device, dtype=dtype)
            beta_new = torch.linalg.solve(XtWX, X_aug.T @ (W * z))
            if torch.norm(beta_new - beta) < tol:
                beta = beta_new
                break
            beta = beta_new
        mu_y = torch.sigmoid(X_aug @ beta)
        resid_y = Y - mu_y

        R = resid_x * resid_y
        mean_stat = R.mean()
        var_stat = R.var()
        T = mean_stat * torch.sqrt(torch.tensor(float(n), device=device)) / torch.sqrt(var_stat + 1e-8)
        pvals[j] = 2.0 * (1.0 - normal_cdf(torch.abs(T)))

    pvals = pvals.clamp(1e-12, 1.0 - 1e-12)
    if to_numpy:
        return pvals.cpu().numpy()
    return pvals


__all__ = ["gcm", "gcm_binary"]


def gcm_single(
    Z: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    y_estimator: str = "linear",
    y_estimator_params: Optional[Dict[str, object]] = None,
    x_estimator: str = "linear",
    x_estimator_params: Optional[Dict[str, object]] = None,
    to_numpy: bool = False,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Single-feature GCM that conditions on latent covariates ``Z``."""

    device = Y.device
    dtype = Y.dtype

    Z = Z.to(device=device, dtype=dtype)
    if Z.ndim == 1:
        Z = Z.view(-1, 1)

    if X.ndim > 1:
        x_vec = X.view(-1)
    else:
        x_vec = X.to(device=device, dtype=dtype)

    y_vec = Y.view(-1).to(device=device, dtype=dtype)

    if torch.unique(y_vec).numel() <= 2:
        return _gcm_single_binary(
            Z,
            x_vec,
            y_vec,
            to_numpy=to_numpy,
            max_iter=max_iter,
            tol=tol,
            x_estimator=x_estimator,
            x_estimator_params=x_estimator_params,
        )

    mu_x = predict_regression(
        Z,
        x_vec,
        Z,
        estimator=x_estimator,
        params=x_estimator_params,
    ).view(-1)
    resid_x = x_vec - mu_x

    mu_y = predict_regression(
        Z,
        y_vec,
        Z,
        estimator=y_estimator,
        params=y_estimator_params,
    ).view(-1)
    resid_y = y_vec - mu_y

    R = resid_x * resid_y
    n = R.shape[0]
    mean_stat = R.mean()
    var_stat = R.var(unbiased=False).clamp_min(1e-8)
    T = mean_stat * torch.sqrt(torch.tensor(float(n), device=device, dtype=dtype)) / torch.sqrt(var_stat)
    pval = 2.0 * (1.0 - normal_cdf(torch.abs(T))).clamp(1e-12, 1.0 - 1e-12)

    if to_numpy:
        return pval.detach().cpu().numpy()
    return pval


def _gcm_single_binary(
    Z: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    to_numpy: bool,
    max_iter: int,
    tol: float,
    x_estimator: str,
    x_estimator_params: Optional[Dict[str, object]],
) -> torch.Tensor:
    dtype = Z.dtype
    device = Z.device

    X = X.to(device=device, dtype=dtype)
    Y = Y.clamp(0.0, 1.0).to(device=device, dtype=dtype)

    mu_x = predict_regression(
        Z,
        X,
        Z,
        estimator=x_estimator,
        params=x_estimator_params,
    ).view(-1)
    resid_x = X - mu_x

    n = Z.shape[0]
    Z_aug = torch.cat([Z, torch.ones(n, 1, device=device, dtype=dtype)], dim=1)
    beta = torch.zeros(Z_aug.shape[1], device=device, dtype=dtype)
    for _ in range(max_iter):
        logits = Z_aug @ beta
        probs = torch.sigmoid(logits)
        W = (probs * (1 - probs)).clamp_min(1e-6)
        z = logits + (Y - probs) / W
        WX = Z_aug * W.unsqueeze(1)
        XtWX = Z_aug.T @ WX + 1e-3 * torch.eye(Z_aug.shape[1], device=device, dtype=dtype)
        beta_new = torch.linalg.solve(XtWX, Z_aug.T @ (W * z))
        if torch.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    mu_y = torch.sigmoid(Z_aug @ beta)
    resid_y = Y - mu_y

    R = resid_x * resid_y
    mean_stat = R.mean()
    var_stat = R.var(unbiased=False).clamp_min(1e-8)
    T = mean_stat * torch.sqrt(torch.tensor(float(n), device=device, dtype=dtype)) / torch.sqrt(var_stat)
    pval = 2.0 * (1.0 - normal_cdf(torch.abs(T))).clamp(1e-12, 1.0 - 1e-12)

    if to_numpy:
        return pval.detach().cpu().numpy()
    return pval


__all__ = [
    "gcm",
    "gcm_binary",
    "gcm_single",
]
