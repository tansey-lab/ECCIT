"""Smoke tests covering estimator choices for GCM and HRT."""

from __future__ import annotations

import numpy as np
import torch

from eccit.calibration_runner import calibrate_run
from eccit.cits import compute_conditional_pvals
from eccit.utils.helpers import make_X


ESTIMATORS = ["linear", "rf", "mlp"]


def make_tiny_tensor(n: int = 48, p: int = 6) -> torch.Tensor:
    X_np = make_X(n, p, distribution="correlated")
    return torch.from_numpy(X_np).float()


def run_gcm_case(estimator: str) -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    X = make_tiny_tensor()
    out = calibrate_run(
        X,
        metric="area",
        num_epochs=1,
        test="gcm",
        use_linear=True,
        gcm_kwargs={
            "y_estimator": estimator,
            "x_estimator": estimator,
        },
        debug_gradients=True,
    )

    calibrator = out[0]
    Y = torch.randn(X.size(0))
    pvals = compute_conditional_pvals(
        X,
        Y,
        test="gcm",
        gcm_kwargs={
            "y_estimator": estimator,
            "x_estimator": estimator,
        },
        to_numpy=True,
    )
    _ = calibrator(pvals)


def run_hrt_case(estimator: str) -> None:
    torch.manual_seed(1)
    np.random.seed(1)

    X = make_tiny_tensor()
    out = calibrate_run(
        X,
        metric="area",
        num_epochs=1,
        test="hrt",
        use_linear=True,
        hrt_n_components=2,
        hrt_n_steps=100,
        hrt_kwargs={
            "estimator_type": estimator,
        },
        debug_gradients=True,
    )
    calibrator, *_extras, diagnostics = out
    hrt_eval_kwargs = {
        "estimator_type": estimator,
        "W_hat": diagnostics.get("W_hat"),
        "V_hat": diagnostics.get("V_hat"),
        "U_hat": diagnostics.get("U_hat"),
    }
    if any(val is None for val in (hrt_eval_kwargs["W_hat"], hrt_eval_kwargs["V_hat"], hrt_eval_kwargs["U_hat"])):
        raise RuntimeError("HRT diagnostics missing factor model components")
    Y = (torch.randn(X.size(0)) > 0).float()
    pvals = compute_conditional_pvals(
        X,
        Y,
        test="hrt",
        hrt_kwargs=hrt_eval_kwargs,
        to_numpy=True,
    )
    _ = calibrator(pvals)


def main() -> None:
    for est in ESTIMATORS:
        run_gcm_case(est)
        run_hrt_case(est)


if __name__ == "__main__":
    main()
