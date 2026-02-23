"""Single-pass HRT implementation aligned with AMI-CRT defaults."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from eccit.utils.estimators import fit_classifier_predictor, fit_regression_predictor


TensorLike = Union[torch.Tensor, np.ndarray]


def ensure_tensor(arr: TensorLike, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        tensor = arr
    else:
        tensor = torch.as_tensor(arr)
    if device is not None:
        tensor = tensor.to(device)
    return tensor.to(dtype=torch.float64)


def normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _feature_context(X: torch.Tensor, feature_idx: int) -> torch.Tensor:
    left = X[:, :feature_idx]
    right = X[:, feature_idx + 1:]
    if left.numel() == 0:
        return right
    if right.numel() == 0:
        return left
    return torch.cat((left, right), dim=1)


@dataclass
class _RandomForestConditional:
    is_binary: bool
    dtype: torch.dtype
    device: torch.device
    rf_kwargs: Dict[str, object]

    def __post_init__(self) -> None:
        self.model = None
        self.residuals = None
        self.classes = None
        self._prob_index = None

    def fit(self, X_context: torch.Tensor, target: torch.Tensor, seed: Optional[int]) -> None:
        kwargs = dict(self.rf_kwargs)
        if seed is not None:
            kwargs.setdefault("random_state", seed)

        X_np = X_context.detach().cpu().numpy()
        y_np = target.detach().cpu().numpy()

        if self.is_binary:
            model = RandomForestClassifier(**kwargs)
            model.fit(X_np, y_np)
            self.model = model

            classes_np = model.classes_.astype(np.float64)
            self.classes = torch.from_numpy(classes_np).to(device=self.device, dtype=self.dtype)

            if self.classes.numel() > 1:
                positive_value = self.classes[-1].item()
                self._prob_index = int(np.where(model.classes_ == positive_value)[0][0])
            else:
                self._prob_index = 0
        else:
            model = RandomForestRegressor(**kwargs)
            model.fit(X_np, y_np)
            self.model = model

            preds_np = model.predict(X_np)
            preds = torch.from_numpy(preds_np).to(device=self.device, dtype=self.dtype)
            self.residuals = (target.to(device=self.device, dtype=self.dtype) - preds).detach()

    def sample(self, X_context: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        assert self.model is not None, "Conditional model must be fit before sampling"

        X_np = X_context.detach().cpu().numpy()
        if self.is_binary:
            if self.classes.numel() == 1:
                return torch.full((X_context.shape[0],), self.classes[0], dtype=self.dtype, device=X_context.device)

            probs_np = self.model.predict_proba(X_np)[:, self._prob_index]
            probs = torch.from_numpy(probs_np).to(device=X_context.device, dtype=self.dtype)

            rand_kwargs = {"device": X_context.device, "dtype": self.dtype}
            if generator is not None:
                rand = torch.rand(probs.shape, generator=generator, **rand_kwargs)
            else:
                rand = torch.rand(probs.shape, **rand_kwargs)

            draws = (rand < probs).to(dtype=self.dtype)
            values = self.classes.to(device=X_context.device)
            low, high = values[0], values[-1]
            return torch.where(draws > 0.5, high, low)

        preds_np = self.model.predict(X_np)
        preds = torch.from_numpy(preds_np).to(device=X_context.device, dtype=self.dtype)

        if self.residuals is None or self.residuals.numel() == 0:
            noise = torch.zeros_like(preds)
        else:
            residuals = self.residuals.to(device=X_context.device)
            idx_kwargs = {"device": X_context.device}
            if generator is not None:
                indices = torch.randint(0, residuals.shape[0], (X_context.shape[0],), generator=generator, **idx_kwargs)
            else:
                indices = torch.randint(0, residuals.shape[0], (X_context.shape[0],), **idx_kwargs)
            noise = residuals[indices]

        return preds + noise.to(dtype=self.dtype)


@dataclass
class _LinearConditional:
    is_binary: bool
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        self.beta = None
        self.sigma = None

    def fit(self, X_context: torch.Tensor, target: torch.Tensor, seed: Optional[int]) -> None:
        X_context = X_context.to(device=self.device, dtype=self.dtype)
        target = target.to(device=self.device, dtype=self.dtype)
        if self.is_binary:
            X_aug = torch.cat([X_context, torch.ones(X_context.shape[0], 1, device=self.device, dtype=self.dtype)], dim=1)
            beta = torch.linalg.lstsq(X_aug, target.unsqueeze(1).to(device=self.device, dtype=self.dtype)).solution.squeeze()
            self.beta = beta
        else:
            X_aug = torch.cat([X_context, torch.ones(X_context.shape[0], 1, device=self.device, dtype=self.dtype)], dim=1)
            beta = torch.linalg.lstsq(X_aug, target.unsqueeze(1).to(device=self.device, dtype=self.dtype)).solution.squeeze()
            self.beta = beta
            preds = X_aug @ beta
            resid = target.to(device=self.device, dtype=self.dtype) - preds
            self.sigma = resid.std(unbiased=True).clamp_min(1e-6)

    def sample(self, X_context: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        dtype = self.dtype
        device = self.device
        X_context = X_context.to(device=device, dtype=dtype)
        X_aug = torch.cat([X_context, torch.ones(X_context.shape[0], 1, device=device, dtype=dtype)], dim=1)

        preds = X_aug @ self.beta
        if self.is_binary:
            probs = torch.sigmoid(preds)
            rand_kwargs = {"device": device, "dtype": dtype}
            if generator is not None:
                rand = torch.rand(probs.shape, generator=generator, **rand_kwargs)
            else:
                rand = torch.rand(probs.shape, **rand_kwargs)
            return (rand < probs).to(dtype=dtype)

        if self.sigma is None:
            noise = torch.zeros_like(preds)
        else:
            if generator is not None:
                noise = torch.randn(preds.shape, generator=generator, device=device, dtype=dtype) * self.sigma
            else:
                noise = torch.randn(preds.shape, device=device, dtype=dtype) * self.sigma
        return preds + noise


@dataclass
class _RidgeConditional:
    is_binary: bool
    dtype: torch.dtype
    device: torch.device
    lam: float

    def __post_init__(self) -> None:
        self.beta = None
        self.sigma = None

    def fit(self, X_context: torch.Tensor, target: torch.Tensor, seed: Optional[int]) -> None:
        X_context = X_context.to(device=self.device, dtype=self.dtype)
        target = target.to(device=self.device, dtype=self.dtype)
        X_aug = torch.cat([X_context, torch.ones(X_context.shape[0], 1, device=self.device, dtype=self.dtype)], dim=1)
        XtX = X_aug.T @ X_aug
        ridge = torch.eye(XtX.shape[0], device=self.device, dtype=self.dtype) * self.lam
        ridge[-1, -1] = 0.0
        XtX = XtX + ridge
        XtY = X_aug.T @ target
        try:
            beta = torch.linalg.solve(XtX, XtY).squeeze(-1)
        except RuntimeError:
            beta = torch.linalg.lstsq(XtX, XtY.unsqueeze(1)).solution.squeeze(-1)
        self.beta = beta
        if not self.is_binary:
            preds = X_aug @ beta
            resid = target - preds
            self.sigma = resid.std(unbiased=True).clamp_min(1e-6)

    def sample(self, X_context: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        dtype = self.dtype
        device = self.device
        X_context = X_context.to(device=device, dtype=dtype)
        X_aug = torch.cat([X_context, torch.ones(X_context.shape[0], 1, device=device, dtype=dtype)], dim=1)

        preds = X_aug @ self.beta
        if self.is_binary:
            probs = torch.sigmoid(preds)
            rand_kwargs = {"device": device, "dtype": dtype}
            if generator is not None:
                rand = torch.rand(probs.shape, generator=generator, **rand_kwargs)
            else:
                rand = torch.rand(probs.shape, **rand_kwargs)
            return (rand < probs).to(dtype=dtype)

        if self.sigma is None:
            noise = torch.zeros_like(preds)
        else:
            if generator is not None:
                noise = torch.randn(preds.shape, generator=generator, device=device, dtype=dtype) * self.sigma
            else:
                noise = torch.randn(preds.shape, device=device, dtype=dtype) * self.sigma
        return preds + noise


@dataclass
class _Poly2Conditional:
    is_binary: bool
    dtype: torch.dtype
    device: torch.device
    lam: float

    def __post_init__(self) -> None:
        self.beta = None
        self.sigma = None

    def _poly2_features(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.shape
        if d == 0:
            return X
        idx = torch.combinations(torch.arange(d, device=X.device), r=2, with_replacement=True)
        quad = X[:, idx[:, 0]] * X[:, idx[:, 1]]
        return torch.cat([X, quad], dim=1)

    def fit(self, X_context: torch.Tensor, target: torch.Tensor, seed: Optional[int]) -> None:
        X_context = X_context.to(device=self.device, dtype=self.dtype)
        target = target.to(device=self.device, dtype=self.dtype)
        X_poly = self._poly2_features(X_context)
        X_aug = torch.cat([X_poly, torch.ones(X_poly.shape[0], 1, device=self.device, dtype=self.dtype)], dim=1)
        XtX = X_aug.T @ X_aug
        ridge = torch.eye(XtX.shape[0], device=self.device, dtype=self.dtype) * self.lam
        ridge[-1, -1] = 0.0
        XtX = XtX + ridge
        XtY = X_aug.T @ target
        try:
            beta = torch.linalg.solve(XtX, XtY).squeeze(-1)
        except RuntimeError:
            beta = torch.linalg.lstsq(XtX, XtY.unsqueeze(1)).solution.squeeze(-1)
        self.beta = beta
        if not self.is_binary:
            preds = X_aug @ beta
            resid = target - preds
            self.sigma = resid.std(unbiased=True).clamp_min(1e-6)

    def sample(self, X_context: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        dtype = self.dtype
        device = self.device
        X_context = X_context.to(device=device, dtype=dtype)
        X_poly = self._poly2_features(X_context)
        X_aug = torch.cat([X_poly, torch.ones(X_poly.shape[0], 1, device=device, dtype=dtype)], dim=1)

        preds = X_aug @ self.beta
        if self.is_binary:
            probs = torch.sigmoid(preds)
            rand_kwargs = {"device": device, "dtype": dtype}
            if generator is not None:
                rand = torch.rand(probs.shape, generator=generator, **rand_kwargs)
            else:
                rand = torch.rand(probs.shape, **rand_kwargs)
            return (rand < probs).to(dtype=dtype)

        if self.sigma is None:
            noise = torch.zeros_like(preds)
        else:
            if generator is not None:
                noise = torch.randn(preds.shape, generator=generator, device=device, dtype=dtype) * self.sigma
            else:
                noise = torch.randn(preds.shape, device=device, dtype=dtype) * self.sigma
        return preds + noise


def _prepare_conditionals(
    X: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: Optional[int],
    rf_kwargs: Optional[Dict[str, object]] = None,
    conditional_type: str = "linear",
) -> list[Union[_RandomForestConditional, _LinearConditional]]:
    cond_kwargs = dict(rf_kwargs or {})
    cond_type = conditional_type.lower()
    if cond_type in {"rf", "random_forest"}:
        cond_kwargs.setdefault("n_estimators", 10)
    elif cond_type in {"ridge", "poly2"}:
        cond_kwargs.setdefault("lambda", 1e-1)

    p = X.shape[1]
    conditionals: list[Union[_RandomForestConditional, _LinearConditional, _RidgeConditional]] = []

    for feature_idx in range(p):
        feature = X[:, feature_idx]
        values = torch.unique(feature)
        is_binary = values.numel() <= 2

        if cond_type == "linear":
            cond = _LinearConditional(
                is_binary=is_binary,
                dtype=dtype,
                device=device,
            )
        elif cond_type == "ridge":
            lam = float(cond_kwargs.get("lambda", cond_kwargs.get("alpha", 1e-1)))
            cond = _RidgeConditional(
                is_binary=is_binary,
                dtype=dtype,
                device=device,
                lam=lam,
            )
        elif cond_type == "poly2":
            lam = float(cond_kwargs.get("lambda", cond_kwargs.get("alpha", 1e-1)))
            cond = _Poly2Conditional(
                is_binary=is_binary,
                dtype=dtype,
                device=device,
                lam=lam,
            )
        else:
            cond = _RandomForestConditional(
                is_binary=is_binary,
                dtype=dtype,
                device=device,
                rf_kwargs=cond_kwargs,
            )

        context = _feature_context(X, feature_idx)
        cond.fit(context, feature, seed)
        conditionals.append(cond)

    return conditionals


def _empirical_gaussian_pval(observed: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    null_mean = samples.mean()
    null_std = samples.std(unbiased=False).clamp_min(1e-6)
    z = (observed - null_mean) / null_std
    return normal_cdf(z).clamp(1e-12, 1.0 - 1e-12)


def hrt_pvals(
    X: TensorLike,
    Y: TensorLike,
    *,
    W_hat: Optional[TensorLike] = None,
    V_hat: Optional[TensorLike] = None,
    U_hat: Optional[TensorLike] = None,
    n_folds: int = 4,
    n_trials: int = 30,
    seed: Optional[int] = None,
    to_numpy: bool = False,
    estimator_type: str = "mlp",
    estimator_params: Optional[Dict[str, Union[int, float, str]]] = None,
    conditional_type: str = "linear",
    conditional_kwargs: Optional[Dict[str, object]] = None,
) -> Union[torch.Tensor, np.ndarray]:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    if isinstance(X, torch.Tensor):
        device = X.device
        dtype = X.dtype
    else:
        device = torch.device("cpu")
        dtype = torch.float64

    _ = (W_hat, V_hat, U_hat)

    X_t = ensure_tensor(X, device=device)
    Y_t = ensure_tensor(Y, device=device).reshape(-1)

    if torch.unique(Y_t).numel() <= 2:
        return hrt_pvals_binary(
            X_t,
            Y_t,
            W_hat=W_hat,
            V_hat=V_hat,
            U_hat=U_hat,
            n_folds=n_folds,
            n_trials=n_trials,
            seed=seed,
            to_numpy=to_numpy,
            estimator_type=estimator_type,
            estimator_params=estimator_params,
        )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    params = dict(estimator_params or {})
    regressor = estimator_type.lower()

    predictor = fit_regression_predictor(
        X_t,
        Y_t,
        estimator=regressor,
        params=params,
    )

    with torch.no_grad():
        preds_obs = predictor(X_t).to(dtype=dtype)
    observed_loss = torch.mean((preds_obs - Y_t) ** 2)

    conditionals = _prepare_conditionals(
        X_t,
        dtype=dtype,
        device=device,
        seed=seed,
        rf_kwargs=conditional_kwargs,
        conditional_type=conditional_type,
    )

    p = X_t.shape[1]
    pvals = []

    for feature_idx in range(p):
        context = _feature_context(X_t, feature_idx)
        cond = conditionals[feature_idx]

        null_losses = []
        for _ in range(n_trials):
            X_rand = X_t.clone().detach()
            samples = cond.sample(context, generator)
            X_rand[:, feature_idx] = samples.to(dtype=dtype)

            with torch.no_grad():
                preds_rand = predictor(X_rand).to(dtype=dtype)
            loss_rand = torch.mean((preds_rand - Y_t) ** 2)
            null_losses.append(loss_rand)

        null_tensor = torch.stack(null_losses)
        p_val = _empirical_gaussian_pval(observed_loss, null_tensor)
        pvals.append(p_val)

    result = torch.stack(pvals).to(dtype=dtype)
    if to_numpy:
        return result.detach().cpu().numpy()
    return result


def hrt_pvals_binary(
    X: TensorLike,
    Y: TensorLike,
    *,
    W_hat: Optional[TensorLike] = None,
    V_hat: Optional[TensorLike] = None,
    U_hat: Optional[TensorLike] = None,
    n_folds: int = 4,
    n_trials: int = 30,
    seed: Optional[int] = None,
    to_numpy: bool = False,
    estimator_type: str = "mlp",
    estimator_params: Optional[Dict[str, Union[int, float, str]]] = None,
    conditional_type: str = "linear",
    conditional_kwargs: Optional[Dict[str, object]] = None,
) -> Union[torch.Tensor, np.ndarray]:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    if isinstance(X, torch.Tensor):
        device = X.device
    else:
        device = torch.device("cpu")

    _ = (W_hat, V_hat, U_hat)

    X_t = ensure_tensor(X, device=device)
    Y_t = ensure_tensor(Y, device=device).reshape(-1).clamp(0.0, 1.0)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    params = dict(estimator_params or {})
    clf_type = estimator_type.lower()

    predictor = fit_classifier_predictor(
        X_t,
        Y_t,
        estimator=clf_type,
        params=params,
    )

    with torch.no_grad():
        probs_obs = predictor(X_t).to(dtype=torch.float32)
    observed_loss = F.binary_cross_entropy(
        probs_obs,
        Y_t.to(dtype=torch.float32),
    )

    conditionals = _prepare_conditionals(
        X_t,
        dtype=X_t.dtype,
        device=device,
        seed=seed,
        rf_kwargs=conditional_kwargs,
        conditional_type=conditional_type,
    )

    p = X_t.shape[1]
    pvals = []

    for feature_idx in range(p):
        context = _feature_context(X_t, feature_idx)
        cond = conditionals[feature_idx]

        null_losses = []
        for _ in range(n_trials):
            X_rand = X_t.clone().detach()
            samples = cond.sample(context, generator)
            X_rand[:, feature_idx] = samples.to(dtype=X_t.dtype)

            with torch.no_grad():
                probs_rand = predictor(X_rand).to(dtype=torch.float32)
            loss_rand = F.binary_cross_entropy(
                probs_rand,
                Y_t.to(dtype=torch.float32),
            )
            null_losses.append(loss_rand.to(dtype=X_t.dtype))

        null_tensor = torch.stack(null_losses)
        p_val = _empirical_gaussian_pval(observed_loss.to(dtype=X_t.dtype), null_tensor)
        pvals.append(p_val)

    result = torch.stack(pvals).to(dtype=X_t.dtype)
    if to_numpy:
        return result.detach().cpu().numpy()
    return result


def hrt_single(
    Z: TensorLike,
    X: TensorLike,
    Y: TensorLike,
    *,
    n_trials: int = 30,
    seed: Optional[int] = None,
    estimator_type: str = "linear",
    estimator_params: Optional[Dict[str, Union[int, float, str]]] = None,
    to_numpy: bool = False,
    conditional_type: str = "linear",
    conditional_kwargs: Optional[Dict[str, object]] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """Single-feature HRT conditioned on ``Z`` with a direct null sampler."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    Z_t = ensure_tensor(Z)
    if Z_t.ndim == 1:
        Z_t = Z_t.view(-1, 1)

    X_t = ensure_tensor(X)
    if X_t.ndim == 1:
        X_t = X_t.view(-1, 1)
    X_col = X_t[:, 0]

    Y_t = ensure_tensor(Y).reshape(-1)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=Z_t.device)
        generator.manual_seed(seed)

    params = dict(estimator_params or {})
    estimator = estimator_type.lower()

    if torch.unique(Y_t).numel() <= 2:
        predictor = fit_classifier_predictor(
            torch.cat([X_t, Z_t], dim=1),
            Y_t,
            estimator=estimator,
            params=params,
        )
        with torch.no_grad():
            probs_obs = predictor(torch.cat([X_t, Z_t], dim=1)).to(dtype=torch.float32)
        observed_loss = F.binary_cross_entropy(
            probs_obs,
            Y_t.to(dtype=torch.float32),
        )
        cond = _prepare_conditionals(
            torch.cat([X_col.view(-1, 1), Z_t], dim=1),
            dtype=Z_t.dtype,
            device=Z_t.device,
            seed=seed,
            rf_kwargs=conditional_kwargs,
            conditional_type=conditional_type,
        )[0]

        null_losses = []
        for _ in range(n_trials):
            X_rand_col = cond.sample(Z_t, generator)
            X_rand = torch.cat([X_rand_col.view(-1, 1), Z_t], dim=1)
            with torch.no_grad():
                probs_rand = predictor(X_rand).to(dtype=torch.float32)
            loss_rand = F.binary_cross_entropy(
                probs_rand,
                Y_t.to(dtype=torch.float32),
            )
            null_losses.append(loss_rand.to(dtype=Z_t.dtype))

        null_tensor = torch.stack(null_losses)
        p_val = _empirical_gaussian_pval(
            observed_loss.to(dtype=Z_t.dtype),
            null_tensor,
        )
    else:
        predictor = fit_regression_predictor(
            torch.cat([X_t, Z_t], dim=1),
            Y_t,
            estimator=estimator,
            params=params,
        )
        with torch.no_grad():
            preds_obs = predictor(torch.cat([X_t, Z_t], dim=1)).to(dtype=Z_t.dtype)
        observed_loss = torch.mean((preds_obs - Y_t) ** 2)

        cond = _prepare_conditionals(
            torch.cat([X_col.view(-1, 1), Z_t], dim=1),
            dtype=Z_t.dtype,
            device=Z_t.device,
            seed=seed,
            rf_kwargs=conditional_kwargs,
            conditional_type=conditional_type,
        )[0]

        null_losses = []
        for _ in range(n_trials):
            X_rand_col = cond.sample(Z_t, generator)
            X_rand = torch.cat([X_rand_col.view(-1, 1), Z_t], dim=1)
            with torch.no_grad():
                preds_rand = predictor(X_rand).to(dtype=Z_t.dtype)
            loss_rand = torch.mean((preds_rand - Y_t) ** 2)
            null_losses.append(loss_rand)

        null_tensor = torch.stack(null_losses)
        p_val = _empirical_gaussian_pval(observed_loss, null_tensor)

    p_val = p_val.clamp(1e-12, 1.0 - 1e-12)
    if to_numpy:
        return p_val.detach().cpu().numpy()
    return p_val


__all__ = ["hrt_pvals", "hrt_pvals_binary", "hrt_single"]
