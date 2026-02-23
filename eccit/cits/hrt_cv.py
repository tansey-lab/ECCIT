"""HRT helpers built on shared estimator utilities."""

from __future__ import annotations

import math
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from eccit.utils.estimators import fit_regression_predictor, fit_classifier_predictor


TensorLike = Union[torch.Tensor, np.ndarray]


def ensure_tensor(arr: TensorLike, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        tensor = arr
    else:
        tensor = torch.as_tensor(arr)
    if device is not None:
        tensor = tensor.to(device)
    return tensor.to(dtype=torch.float64)


def split_folds(n: int, n_folds: int, generator: Optional[torch.Generator]) -> list[torch.Tensor]:
    indices = torch.randperm(n, generator=generator)
    fold_sizes = [(n + i) // n_folds for i in range(n_folds)]
    folds = []
    start = 0
    for size in fold_sizes:
        end = start + size
        folds.append(indices[start:end])
        start = end
    return folds


def normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def hrt_cv_pvals(
    X: TensorLike,
    Y: TensorLike,
    *,
    W_hat: TensorLike,
    V_hat: TensorLike,
    U_hat: TensorLike,
    n_folds: int = 4,
    n_trials: int = 30,
    seed: Optional[int] = None,
    to_numpy: bool = False,
    estimator_type: str = "mlp",
    estimator_params: Optional[Dict[str, Union[int, float, str]]] = None,
) -> Union[torch.Tensor, np.ndarray]:
    if n_folds <= 1:
        raise ValueError("n_folds must be greater than 1")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    if isinstance(X, torch.Tensor):
        device = X.device
        dtype = X.dtype
    else:
        device = torch.device("cpu")
        dtype = torch.float64

    X_t = ensure_tensor(X, device=device)
    Y_t = ensure_tensor(Y, device=device).reshape(-1)
    W_t = ensure_tensor(W_hat, device=device)
    V_t = ensure_tensor(V_hat, device=device)
    U_t = ensure_tensor(U_hat, device=device)

    n, p = X_t.shape
    if Y_t.shape[0] != n:
        raise ValueError("X and Y must have matching number of samples")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    means = W_t @ V_t.T
    scales_raw = W_t @ U_t.T
    scales = torch.nn.functional.softplus(scales_raw) + 1e-6

    if torch.unique(Y_t).numel() <= 2:
        return hrt_cv_pvals_binary(
            X_t,
            Y_t,
            W_hat=W_t,
            V_hat=V_t,
            U_hat=U_t,
            n_folds=n_folds,
            n_trials=n_trials,
            seed=seed,
            to_numpy=to_numpy,
            estimator_type=estimator_type,
            estimator_params=estimator_params,
        )

    folds = split_folds(n, n_folds, generator)
    fold_sizes = torch.tensor([len(idx) for idx in folds], dtype=dtype, device=device)
    fold_weights = fold_sizes / float(n)

    regressor = estimator_type.lower()
    params = dict(estimator_params or {})

    pvals = []
    for feature_idx in range(p):
        fold_losses = []
        fold_infos = []

        for fold_idx, test_idx in enumerate(folds):
            train_idx = torch.cat([folds[i] for i in range(n_folds) if i != fold_idx])
            X_train = X_t[train_idx]
            Y_train = Y_t[train_idx]
            X_test = X_t[test_idx]
            Y_test = Y_t[test_idx]

            predictor = fit_regression_predictor(
                X_train,
                Y_train,
                estimator=regressor,
                params=params,
            )

            with torch.no_grad():
                preds = predictor(X_test).to(dtype=dtype)
            obs_loss = torch.mean((preds - Y_test) ** 2)
            fold_losses.append(obs_loss)

            means_test = means[test_idx, feature_idx]
            scales_test = scales[test_idx, feature_idx]

            fold_infos.append({
                "weight": fold_weights[fold_idx],
                "predictor": predictor,
                "X_test": X_test.detach(),
                "Y_test": Y_test.detach(),
                "means": means_test.detach(),
                "scales": scales_test.detach(),
            })

        observed = torch.zeros((), dtype=dtype, device=device)
        for loss, info in zip(fold_losses, fold_infos):
            observed = observed + info["weight"] * loss

        null_losses = []
        for _ in range(n_trials):
            trial_loss = torch.zeros((), dtype=dtype, device=device)
            for info in fold_infos:
                noise = torch.randn_like(info["means"])
                sampled_feature = info["means"] + info["scales"] * noise
                X_rand = info["X_test"].clone()
                X_rand[:, feature_idx] = sampled_feature
                with torch.no_grad():
                    preds_rand = info["predictor"](X_rand).to(dtype=dtype)
                loss_rand = torch.mean((preds_rand - info["Y_test"]) ** 2)
                trial_loss = trial_loss + info["weight"] * loss_rand
            null_losses.append(trial_loss)

        null_losses_t = torch.stack(null_losses)
        null_mean = null_losses_t.mean()
        null_std = null_losses_t.std(unbiased=False).clamp_min(1e-6)
        z = (observed - null_mean) / null_std
        p_val = normal_cdf(z).clamp(1e-12, 1.0 - 1e-12)
        pvals.append(p_val)

    pvals_t = torch.stack(pvals).to(dtype=dtype)
    if to_numpy:
        return pvals_t.detach().cpu().numpy()
    return pvals_t


def hrt_cv_pvals_binary(
    X: TensorLike,
    Y: TensorLike,
    *,
    W_hat: TensorLike,
    V_hat: TensorLike,
    U_hat: TensorLike,
    n_folds: int = 4,
    n_trials: int = 30,
    seed: Optional[int] = None,
    to_numpy: bool = False,
    estimator_type: str = "mlp",
    estimator_params: Optional[Dict[str, Union[int, float, str]]] = None,
) -> Union[torch.Tensor, np.ndarray]:
    if n_folds <= 1:
        raise ValueError("n_folds must be greater than 1")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    if isinstance(X, torch.Tensor):
        device = X.device
        dtype = X.dtype
    else:
        device = torch.device("cpu")
        dtype = torch.float64

    X_t = ensure_tensor(X, device=device)
    Y_t = ensure_tensor(Y, device=device).reshape(-1).clamp(0.0, 1.0)
    W_t = ensure_tensor(W_hat, device=device)
    V_t = ensure_tensor(V_hat, device=device)
    U_t = ensure_tensor(U_hat, device=device)

    n, p = X_t.shape
    if Y_t.shape[0] != n:
        raise ValueError("X and Y must have matching number of samples")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    means = W_t @ V_t.T
    scales_raw = W_t @ U_t.T
    scales = torch.nn.functional.softplus(scales_raw) + 1e-6

    folds = split_folds(n, n_folds, generator)
    fold_sizes = torch.tensor([len(idx) for idx in folds], dtype=dtype, device=device)
    fold_weights = fold_sizes / float(n)

    clf_type = estimator_type.lower()
    params = dict(estimator_params or {})

    pvals = []
    for feature_idx in range(p):
        fold_losses = []
        fold_infos = []

        for fold_idx, test_idx in enumerate(folds):
            train_idx = torch.cat([folds[i] for i in range(n_folds) if i != fold_idx])
            X_train = X_t[train_idx]
            Y_train = Y_t[train_idx]
            X_test = X_t[test_idx]
            Y_test = Y_t[test_idx]

            predictor = fit_classifier_predictor(
                X_train,
                Y_train,
                estimator=clf_type,
                params=params,
            )

            with torch.no_grad():
                probs = predictor(X_test).to(dtype=torch.float32)
            obs_loss = F.binary_cross_entropy(
                probs,
                Y_test.to(dtype=torch.float32),
            )
            fold_losses.append(obs_loss)

            means_test = means[test_idx, feature_idx]
            scales_test = scales[test_idx, feature_idx]

            fold_infos.append({
                "weight": fold_weights[fold_idx],
                "predictor": predictor,
                "X_test": X_test.detach(),
                "Y_test": Y_test.detach(),
                "means": means_test.detach(),
                "scales": scales_test.detach(),
            })

        observed = torch.zeros((), dtype=dtype, device=device)
        for loss, info in zip(fold_losses, fold_infos):
            observed = observed + info["weight"] * loss

        null_losses = []
        for _ in range(n_trials):
            trial_loss = torch.zeros((), dtype=dtype, device=device)
            for info in fold_infos:
                noise = torch.randn_like(info["means"])
                sampled_feature = info["means"] + info["scales"] * noise
                X_rand = info["X_test"].clone()
                X_rand[:, feature_idx] = sampled_feature
                with torch.no_grad():
                    probs_rand = info["predictor"](X_rand).to(dtype=torch.float32)
                loss_rand = F.binary_cross_entropy(probs_rand, info["Y_test"].to(dtype=torch.float32))
                trial_loss = trial_loss + info["weight"] * loss_rand
            null_losses.append(trial_loss)

        null_losses_t = torch.stack(null_losses)
        null_mean = null_losses_t.mean()
        null_std = null_losses_t.std(unbiased=False).clamp_min(1e-6)
        z = (observed - null_mean) / null_std
        p_val = normal_cdf(z).clamp(1e-12, 1.0 - 1e-12)
        pvals.append(p_val)

    pvals_t = torch.stack(pvals).to(dtype=dtype)
    if to_numpy:
        return pvals_t.detach().cpu().numpy()
    return pvals_t


__all__ = ["hrt_cv_pvals", "hrt_cv_pvals_binary"]
