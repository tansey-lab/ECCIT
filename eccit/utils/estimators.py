"""Shared estimator utilities for regression and classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor, XGBClassifier


def ensure_float(y: torch.Tensor) -> torch.Tensor:
    if y.dtype in (torch.float16, torch.float32, torch.float64):
        return y
    return y.to(dtype=torch.float32)


def _poly2_features(X: torch.Tensor) -> torch.Tensor:
    """Build degree-2 polynomial features (including squares) for regression."""
    n, d = X.shape
    if d == 0:
        return X
    idx = torch.combinations(torch.arange(d, device=X.device), r=2, with_replacement=True)
    quad = X[:, idx[:, 0]] * X[:, idx[:, 1]]
    return torch.cat([X, quad], dim=1)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp_regressor(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> MLPRegressor:
    X_det = X.detach()
    y_det = y.detach()
    model = MLPRegressor(X.shape[1], hidden=hidden).to(device=X.device, dtype=X_det.dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(X_det, y_det.to(dtype=X_det.dtype, device=X_det.device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    with torch.enable_grad():
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device=X.device, dtype=X_det.dtype)
                yb = yb.to(device=X.device, dtype=X_det.dtype)
                opt.zero_grad()
                preds = model(xb)
                loss = F.mse_loss(preds, yb)
                loss.backward()
                opt.step()

    model.eval()
    return model


def train_mlp_classifier(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> MLPClassifier:
    X_det = X.detach()
    y_det = y.detach().to(dtype=X_det.dtype, device=X_det.device)
    model = MLPClassifier(X.shape[1], hidden=hidden).to(device=X.device, dtype=X_det.dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(X_det, y_det)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    with torch.enable_grad():
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device=X.device, dtype=X_det.dtype)
                yb = yb.to(device=X.device, dtype=X_det.dtype)
                opt.zero_grad()
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                loss.backward()
                opt.step()

    model.eval()
    return model


def fit_regression_predictor(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    estimator: str,
    params: Optional[Dict[str, object]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    est = (estimator or "ridge").lower()
    cfg = dict(params or {})
    y_dtype = y_train.dtype
    device = X_train.device

    if est == "linear":
        lam = float(cfg.get("lambda", 1e-3))
        if lam < 0.0:
            raise ValueError("lambda must be nonnegative for linear estimator")
        X_det = X_train.detach()
        y_det = y_train.detach()
        XtX = X_det.T @ X_det
        if lam > 0.0:
            XtX = XtX + lam * torch.eye(XtX.shape[0], device=device, dtype=X_det.dtype)
        XtY = X_det.T @ y_det
        try:
            beta = torch.linalg.solve(XtX, XtY)
        except RuntimeError:
            beta = torch.linalg.lstsq(X_det, y_det.unsqueeze(1)).solution.squeeze(-1)

        def predict(X_eval: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return (X_eval.detach() @ beta).to(dtype=y_dtype)

        return predict

    if est in {"poly", "poly2", "polynomial"}:
        degree = int(cfg.get("degree", 2))
        if degree != 2:
            raise ValueError("Only degree=2 is supported for polynomial estimator")
        lam = float(cfg.get("lambda", 1e-1))
        if lam < 0.0:
            raise ValueError("lambda must be nonnegative for polynomial estimator")
        X_det = _poly2_features(X_train.detach())
        y_det = y_train.detach()
        XtX = X_det.T @ X_det
        XtX = XtX + lam * torch.eye(XtX.shape[0], device=device, dtype=X_det.dtype)
        XtY = X_det.T @ y_det
        try:
            beta = torch.linalg.solve(XtX, XtY)
        except RuntimeError:
            beta = torch.linalg.lstsq(XtX, XtY.unsqueeze(1)).solution.squeeze(-1)

        def predict(X_eval: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                X_poly = _poly2_features(X_eval.detach())
                return (X_poly @ beta).to(dtype=y_dtype)

        return predict

    if est == "ridge":
        lam = float(cfg.get("lambda", 1e-1))
        if lam < 0.0:
            raise ValueError("lambda must be nonnegative for ridge estimator")
        X_det = X_train.detach()
        y_det = y_train.detach()
        XtX = X_det.T @ X_det
        XtX = XtX + lam * torch.eye(XtX.shape[0], device=device, dtype=X_det.dtype)
        XtY = X_det.T @ y_det
        try:
            beta = torch.linalg.solve(XtX, XtY)
        except RuntimeError:
            beta = torch.linalg.lstsq(X_det, y_det.unsqueeze(1)).solution.squeeze(-1)

        def predict(X_eval: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return (X_eval.detach() @ beta).to(dtype=y_dtype)

        return predict

    if est in {"rf", "random_forest"}:
        cfg.setdefault("n_estimators", 10)
        model = RandomForestRegressor(**cfg)
        model.fit(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())

        def predict(X_eval: torch.Tensor) -> torch.Tensor:
            preds = model.predict(X_eval.detach().cpu().numpy())
            return torch.from_numpy(preds).to(device=X_eval.device, dtype=y_dtype)

        return predict

    if est in {"xgb", "xgboost"}:
        model = XGBRegressor(**cfg)
        model.fit(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())

        def predict(X_eval: torch.Tensor) -> torch.Tensor:
            preds = model.predict(X_eval.detach().cpu().numpy())
            return torch.from_numpy(preds).to(device=X_eval.device, dtype=y_dtype)

        return predict

    if est in {"mlp", "nn"}:
        hidden = int(cfg.get("hidden", 64))
        epochs = int(cfg.get("epochs", 100))
        lr = float(cfg.get("lr", 1e-3))
        weight_decay = float(cfg.get("weight_decay", 1e-4))
        batch_size = int(cfg.get("batch_size", min(512, max(16, X_train.shape[0]))))
        model = train_mlp_regressor(
            X_train,
            ensure_float(y_train),
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )

        def predict(X_eval: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return model(X_eval.detach()).to(dtype=y_dtype)

        return predict

    raise ValueError(f"Unsupported regression estimator '{estimator}'")


def predict_regression(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_eval: torch.Tensor,
    *,
    estimator: str,
    params: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    predictor = fit_regression_predictor(X_train, y_train, estimator=estimator, params=params)
    return predictor(X_eval)


def fit_classifier_predictor(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    estimator: str,
    params: Optional[Dict[str, object]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    est = (estimator or "logistic").lower()
    cfg = dict(params or {})
    device = X_train.device

    y_float = ensure_float(y_train)

    if est in {"linear", "logistic", "logit"}:
        clf = LogisticRegression(
            penalty=cfg.get("penalty", "l2"),
            solver=cfg.get("solver", "lbfgs"),
            fit_intercept=True,
            max_iter=int(cfg.get("max_iter", 1000)),
        )
        clf.fit(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())

        def predict(X_eval: torch.Tensor) -> torch.Tensor:
            probs = clf.predict_proba(X_eval.detach().cpu().numpy())[:, 1]
            return torch.from_numpy(probs).to(device=X_eval.device, dtype=torch.float32)

        return predict

    if est in {"rf", "random_forest"}:
        cfg.setdefault("n_estimators", 10)
        model = RandomForestClassifier(**cfg)
        model.fit(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())

        def predict(X_eval: torch.Tensor) -> torch.Tensor:
            probs = model.predict_proba(X_eval.detach().cpu().numpy())[:, 1]
            return torch.from_numpy(probs).to(device=X_eval.device, dtype=torch.float32)

        return predict

    if est in {"xgb", "xgboost"}:
        model = XGBClassifier(**cfg)
        model.fit(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())

        def predict(X_eval: torch.Tensor) -> torch.Tensor:
            probs = model.predict_proba(X_eval.detach().cpu().numpy())[:, 1]
            return torch.from_numpy(probs).to(device=X_eval.device, dtype=torch.float32)

        return predict

    if est in {"mlp", "nn"}:
        hidden = int(cfg.get("hidden", 64))
        epochs = int(cfg.get("epochs", 100))
        lr = float(cfg.get("lr", 1e-3))
        weight_decay = float(cfg.get("weight_decay", 1e-4))
        batch_size = int(cfg.get("batch_size", min(512, max(16, X_train.shape[0]))))

        model = train_mlp_classifier(
            X_train,
            y_float,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )

        def predict(X_eval: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                logits = model(X_eval.detach())
                return torch.sigmoid(logits).to(dtype=torch.float32)

        return predict

    raise ValueError(f"Unsupported classifier estimator '{estimator}'")
