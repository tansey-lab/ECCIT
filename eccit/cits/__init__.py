"""Test utilities and registry."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .gcm import gcm, gcm_binary, gcm_single
from .kcit import kcit_pvals
from .rcit import rcit_pvals
from .hrt import hrt_pvals, hrt_pvals_binary, hrt_single
from .hrt_cv import hrt_cv_pvals, hrt_cv_pvals_binary


def compute_conditional_pvals(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    test: str = "gcm",
    order: int = 1,
    use_linear: bool = True,
    to_numpy: bool = False,
    kcit_kwargs: Optional[Dict[str, object]] = None,
    rcit_kwargs: Optional[Dict[str, object]] = None,
    hrt_kwargs: Optional[Dict[str, object]] = None,
    gcm_kwargs: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    """Compute per-feature p-values for the requested conditional test."""
    test = test.lower()

    if test == "kcit":
        kwargs = dict(kcit_kwargs or {})
        kwargs.setdefault("to_numpy", to_numpy)
        return kcit_pvals(X, Y, **kwargs)

    if test == "rcit":
        kwargs = dict(rcit_kwargs or {})
        kwargs.setdefault("to_numpy", to_numpy)
        return rcit_pvals(X, Y, **kwargs)

    if test == "hrt":
        kwargs = dict(hrt_kwargs or {})
        kwargs.setdefault("to_numpy", to_numpy)
        use_cv = bool(kwargs.pop("use_cv", False))
        if use_cv:
            kwargs.pop("conditional_type", None)
            kwargs.pop("conditional_kwargs", None)
            if not all(key in kwargs for key in ("W_hat", "V_hat", "U_hat")):
                raise ValueError("hrt_cv_pvals requires W_hat, V_hat, and U_hat in hrt_kwargs")
            if torch.unique(Y).numel() <= 2:
                return hrt_cv_pvals_binary(X, Y, **kwargs)
            return hrt_cv_pvals(X, Y, **kwargs)
        if torch.unique(Y).numel() <= 2:
            return hrt_pvals_binary(X, Y, **kwargs)
        return hrt_pvals(X, Y, **kwargs)

    if test == "gcm":
        if torch.unique(Y).numel() <= 2:
            return gcm_binary(X, Y, to_numpy=to_numpy)

        kwargs = dict(gcm_kwargs or {})
        if not kwargs and order > 1:
            kwargs = {
                "y_estimator": "mlp",
                "x_estimator": "mlp",
            }
        return gcm(X, Y, to_numpy=to_numpy, **kwargs)



__all__ = [
    "gcm",
    "gcm_binary",
    "gcm_single",
    "kcit_pvals",
    "rcit_pvals",
    "hrt_pvals",
    "hrt_pvals_binary",
    "hrt_single",
    "compute_conditional_pvals",
]
