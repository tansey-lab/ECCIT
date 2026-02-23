"""Benchmark utility for comparing GCM variants, HRT, and CONTRA.

This version decouples the feature generator (``x_distribution``) from the
response mechanism (``response``).  It supports the new MLP-based HRT
implementation for nonlinear binary data while reverting to a logistic
classifier for the linear case.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from eccit.contra import amicrt as contra
import pandas as pd
from eccit.cits.gcm import gcm, gcm_binary
from eccit.cits.hrt import hrt_pvals, hrt_pvals_binary
from eccit.utils.helpers import make_X, compute_power_stats, make_Y
from eccit.utils.sgd_factor_model import factorize
from eccit.calibration_runner import calibrate_run


HRT_DEFAULT_COMPONENTS = 10
HRT_DEFAULT_STEPS = 1000
ALPHA_TABLE = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=float)


@dataclass(frozen=True)
class BenchmarkConfig:
    n: int = 400
    m: int = 10
    active_features: int = 4
    gamma: float = 0.5
    x_distribution: str = "normal"
    response: str = "linear"


@dataclass
class BenchmarkInstance:
    X: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    beta: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None



def _summarize_curves(
    pvals: np.ndarray,
    truth_mask: np.ndarray,
    alphas: np.ndarray,
    *,
    alpha_adjust: Optional[Callable[[float], float]] = None,
) -> dict[str, np.ndarray]:
    power, valid, fdr = [], [], []
    for alpha in alphas:
        valid_power, power_val, fdr_val = compute_power_stats(
            pvals,
            truth_mask,
            alpha=alpha,
            alpha_adjust=alpha_adjust,
        )
        power.append(power_val)
        valid.append(valid_power)
        fdr.append(fdr_val)
    return {
        "power": np.asarray(power, dtype=float),
        "valid": np.asarray(valid, dtype=float),
        "fdr": np.asarray(fdr, dtype=float),
    }




def _select_gcm_estimator(response: str) -> str:
    resp = response.lower()
    if resp in {"linear", "linear_continuous"}:
        return "linear"
    # return "mlp"
    # return "xgb"
    return "linear"


def _select_hrt_estimator(response: str) -> str:
    resp = response.lower()
    if resp == "linear":
        return "logistic"
    if resp == "linear_continuous":
        return "linear"
    # return "mlp"
    # return "xgb"
    return "linear"

def _make_gcm_kwargs(estimator: str) -> Optional[Dict[str, object]]:
    est = (estimator or "linear").lower()
    if est in {"linear", "ridge"}:
        return None
    if est in {"mlp", "nn"}:
        return {
            "y_estimator": "mlp",
            "y_estimator_params": {
                "hidden": 64,
                "epochs": 30,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 256,
            },
            "x_estimator": "linear",
        }
    if est in {"rf", "random_forest"}:
        params = {"n_estimators": 200, "max_depth": None, "random_state": 0}
        return {
            "y_estimator": "rf",
            "y_estimator_params": params,
            "x_estimator": "rf",
            "x_estimator_params": params,
        }
    if est in {"xgb", "xgboost"}:
        params = {"n_estimators": 10}
        return {
            "y_estimator": "xgb",
            "y_estimator_params": dict(params),
            "x_estimator": "xgb",
            "x_estimator_params": dict(params),
        }
    raise ValueError(f"Unsupported GCM estimator '{estimator}'")


def _response_caption(response: str) -> str:
    resp = response.lower()
    if resp == "linear":
        return "(i) Linear Response."
    if resp == "linear_continuous":
        return "(i) Linear Continuous Response."
    if resp == "nonlinear":
        return "(ii) Nonlinear Response."
    return f"{response.replace('_', ' ').title()} Response."




def sample_instance(config: BenchmarkConfig, seed: int) -> BenchmarkInstance:
    rng = np.random.default_rng(seed)

    # Set global numpy seed for make_X
    np.random.seed(seed)

    X = make_X(
        config.n,
        config.m,
        distribution=config.x_distribution,
        gamma=config.gamma
    )

    response = config.response.lower()
    mask = np.zeros(config.m, dtype=bool)
    beta = None
    sigma = None

    if response == "linear":
        m = X.shape[1]
        beta = np.zeros(m, dtype=float)
        mask = np.zeros(m, dtype=bool)
        if config.active_features > 0:
            active = rng.choice(m, size=config.active_features, replace=False)
            beta[active] = rng.normal(loc=1.0, scale=0.5, size=config.active_features)
            mask[active] = True
        logits = X @ beta
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = rng.binomial(1, prob)
        sigma = np.full((m, m), 0.0, dtype=float)
        np.fill_diagonal(sigma, 1.0)
        return BenchmarkInstance(X=X, y=y, mask=mask, beta=beta, sigma=sigma)
    
    if response == "linear_continuous":
        m = X.shape[1]
        beta = np.zeros(m, dtype=float)
        mask = np.zeros(m, dtype=bool)
        active = np.array([], dtype=int)
        if config.active_features > 0:
            active = rng.choice(m, size=config.active_features, replace=False)
            beta[active] = rng.normal(loc=0.0, scale=1.0, size=config.active_features)
            mask[active] = True

        noise = rng.normal(loc=0.0, scale=1.0, size=config.n)
        y = X @ beta + noise

        sigma = np.eye(m, dtype=float)
        return BenchmarkInstance(X=X, y=y, mask=mask, beta=beta, sigma=sigma)

    if response == "orange":
        mask[: config.active_features] = True
        if config.active_features == 0:
            y = rng.integers(0, 2, size=config.n)
        else:
            logits = np.sum(X[:, : config.active_features] ** 2, axis=-1) - config.active_features
            probs = 1.0 / (1.0 + np.exp(-logits))
            y = rng.binomial(1, probs)
        return BenchmarkInstance(X=X, y=y, mask=mask)

    if response == "nonlinear":
        y, active_idx = make_Y(X, config.active_features, order=2)
        mask = np.zeros(config.m, dtype=bool)
        mask[active_idx] = True
        return BenchmarkInstance(X=X, y=y, mask=mask)

    raise ValueError(f"Unknown response '{config.response}'.")


def gcm_pvalues(
    instance: BenchmarkInstance,
    seed: Optional[int] = None,
    *,
    estimator_type: str = "linear",
    y_estimator_params: Optional[dict] = None,
    x_estimator: str = "linear",
    x_estimator_params: Optional[dict] = None,
) -> np.ndarray:
    """Compute GCM p-values using the estimator-backed implementation."""

    X_t = torch.from_numpy(instance.X).float()
    y_t = torch.from_numpy(instance.y).float()
    is_binary = np.unique(instance.y).size <= 2

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if is_binary:
        return gcm_binary(X_t, y_t, to_numpy=True)

    kwargs = _make_gcm_kwargs(estimator_type)
    if kwargs is None:
        return gcm(X_t, y_t, to_numpy=True)

    if y_estimator_params is not None:
        kwargs = dict(kwargs)
        params = dict(kwargs.get("y_estimator_params") or {})
        params.update(y_estimator_params)
        kwargs["y_estimator_params"] = params

    if x_estimator_params is not None and kwargs.get("x_estimator"):
        kwargs = dict(kwargs)
        params = dict(kwargs.get("x_estimator_params") or {})
        params.update(x_estimator_params)
        kwargs["x_estimator_params"] = params

    if x_estimator is not None:
        kwargs["x_estimator"] = x_estimator
    return gcm(X_t, y_t, to_numpy=True, **kwargs)


def hrt_pvalues(
    instance: BenchmarkInstance,
    seed: Optional[int] = None,
    *,
    n_components: int = HRT_DEFAULT_COMPONENTS,
    n_steps: int = HRT_DEFAULT_STEPS,
    estimator_type: str = "mlp",
) -> np.ndarray:
    is_binary = np.unique(instance.y).size <= 2

    X_np = np.asarray(instance.X, dtype=float)
    k = min(n_components, X_np.shape[1])

    if seed is not None:
        seed32 = int(seed % (2**32 - 1))
        np.random.seed(seed32)
        torch.manual_seed(seed32)

    _, _, W_hat, V_hat, U_hat, _ = factorize(
        X_np,
        n_components=k,
        n_steps=n_steps,
    )

    if is_binary:
        return hrt_pvals_binary(
            instance.X,
            instance.y,
            W_hat=W_hat,
            V_hat=V_hat,
            U_hat=U_hat,
            n_trials=100,
            n_folds=5,
            seed=seed,
            to_numpy=True,
            estimator_type=estimator_type,
        )

    return hrt_pvals(
        instance.X,
        instance.y,
        W_hat=W_hat,
        V_hat=V_hat,
        U_hat=U_hat,
        n_trials=100,
        n_folds=5,
        seed=seed,
        to_numpy=True,
        estimator_type=estimator_type,
    )


def build_calibrator(
    config: BenchmarkConfig,
    *,
    test: str,
    metric: str,
    alpha: float,
    seed: int,
    num_epochs: int,
    estimator_type: str = "mlp",
    hrt_kwargs: Optional[dict] = None,
    order_adv_override: Optional[int] = None,
    order_test_override: Optional[int] = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], Optional[Callable[[float], float]]]:
    inst = sample_instance(config, seed)
    X_tensor = torch.from_numpy(inst.X).float()

    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    np.random.seed(seed)

    gcm_kwargs = _make_gcm_kwargs(estimator_type) if test == "gcm" else None

    nonlinear_responses = {"orange", "nonlinear"}
    adversary_order = order_adv_override if order_adv_override is not None else (
        2 if config.response.lower() in nonlinear_responses else 1
    )
    order_test = order_test_override if order_test_override is not None else (
        2 if config.response.lower() in nonlinear_responses else 1
    )

    order_test = order_test_override if order_test_override is not None else (
        2 if config.response.lower() in nonlinear_responses else 1
    )

    calibrator, _, _, _, _, _, _, diagnostics = calibrate_run(
        X_tensor,
        metric=metric,
        use_linear=(test == "gcm" and estimator_type == "linear"),
        order_adv=adversary_order,
        order_test=order_test,
        alpha_train=alpha,
        test=test,
        num_epochs=num_epochs,
        hrt_n_components=HRT_DEFAULT_COMPONENTS,
        hrt_n_steps=HRT_DEFAULT_STEPS,
        gcm_kwargs=gcm_kwargs,
        hrt_kwargs=hrt_kwargs,
    )

    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)

    if metric in ("area", "type1"):
        cal_fn = calibrator
        alpha_adjust = diagnostics.get("alpha_adjust")
    else:
        cal_fn = lambda p: np.asarray(p, dtype=float)
        alpha_adjust = diagnostics.get("alpha_adjust_fdp")

    if alpha_adjust is None:
        alpha_adjust = lambda a: a

    return cal_fn, alpha_adjust


def contra_pvalues(
    instance: BenchmarkInstance,
    *,
    method: str,
    seed: int,
    response_type: str = "linear",
) -> np.ndarray:
    X_train, X_test, y_train, y_test = train_test_split(
        instance.X, instance.y, test_size=0.3, random_state=seed
    )

    method = method.lower()
    response_type = response_type.lower()
    is_binary = np.unique(instance.y).size <= 2
    common_kwargs: Dict[str, object] = dict(tqdm=lambda it, **_: it)
    ordering = '>'

    if is_binary and response_type == "linear":
        stat_kwargs = dict(
            TestStatisticClass=contra.statistics.ModelBasedStatistic,
            TestStatisticArgs=dict(
                modelClass=LogisticRegression,
                modelArgs=dict(max_iter=1000, solver="lbfgs"),
                fn=contra.utils.monteCarloEntropy,
            ),
        )
    elif not is_binary:
        def mse_stat(model, x, y):
            preds = model.predict(x)
            return float(np.mean((preds - y) ** 2))

        stat_kwargs = dict(
            TestStatisticClass=contra.statistics.ModelBasedRegressor,
            TestStatisticArgs=dict(
                modelClass=RandomForestRegressor,
                modelArgs=dict(n_estimators=10, random_state=seed),
                fn=mse_stat,
            ),
        )
        ordering = '<'
        # Use ContinuousConditional for continuous features to avoid discretization
        common_kwargs["CompleteConditionalClass"] = contra.conditionals.ContinuousConditional
        common_kwargs["CompleteConditionalArgs"] = dict(
            modelClass=RandomForestRegressor,
            modelArgs=dict(n_estimators=10, random_state=seed)
        )
    else:
        stat_kwargs = dict(
            TestStatisticClass=contra.statistics.ModelBasedStatistic,
            TestStatisticArgs=dict(fn=contra.utils.monteCarloEntropy),
        )

    common_kwargs.update(stat_kwargs)
    common_kwargs["ordering"] = ordering

    if method == "fastcrt":
        crt = contra.FastCRT(**common_kwargs)
    else:
        crt = contra.CRT(refitStatistic=(method != "hrt"), **common_kwargs)

    crt.initialize(X_train, y_train)
    pvals = crt.fit_evaluate(X_train, y_train, X_test, y_test)
    return np.array([pvals[j] for j in sorted(pvals)], dtype=float)


def run_single_experiment(args_tuple):
    run_idx, config, args = args_tuple

    include_gcm = "gcm" in args.tests
    include_hrt = "hrt" in args.tests

    gcm_estimator = _select_gcm_estimator(config.response)
    hrt_estimator = _select_hrt_estimator(config.response)

    calibrators: Dict[str, tuple] = {}

    cal_seed = args.seed + 10_000

    order_adv_override = getattr(args, 'order_adv', None)
    order_test_override = getattr(args, 'order_test', None)

    if include_gcm:
        gcm_calibrator, gcm_alpha_adjustor = build_calibrator(
            config,
            test="gcm",
            metric=args.metric,
            alpha=args.alpha,
            seed=cal_seed,
            num_epochs=args.num_epochs,
            estimator_type=gcm_estimator,
            order_adv_override=order_adv_override,
            order_test_override=order_test_override,
        )
        calibrators["gcm"] = (gcm_calibrator, gcm_alpha_adjustor)

    if include_hrt:
        hrt_calibrator, hrt_alpha_adjustor = build_calibrator(
            config,
            test="hrt",
            metric=args.metric,
            alpha=args.alpha,
            seed=cal_seed + 5_000,
            num_epochs=args.num_epochs,
            hrt_kwargs={"estimator_type": hrt_estimator},
            order_adv_override=order_adv_override,
            order_test_override=order_test_override,
        )
        calibrators["hrt"] = (hrt_calibrator, hrt_alpha_adjustor)

    inst = sample_instance(config, args.seed + run_idx)

    alpha_grid = np.asarray(getattr(args, "alpha_levels", ALPHA_TABLE), dtype=float)
    mask = inst.mask

    results: Dict[str, dict] = {"alphas": alpha_grid}

    def summarize_method(
        pvals: np.ndarray,
        *,
        calibrator_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        alpha_adjust: Optional[Callable[[float], float]] = None,
    ) -> dict[str, np.ndarray]:
        eval_p = np.asarray(pvals, dtype=float)
        if calibrator_fn is not None:
            eval_p = np.asarray(calibrator_fn(eval_p), dtype=float)
        return _summarize_curves(eval_p, mask, alpha_grid, alpha_adjust=alpha_adjust)

    if include_gcm:
        gcm_p = gcm_pvalues(
            inst,
            seed=args.seed + run_idx,
            estimator_type=gcm_estimator,
        )
        results["gcm"] = summarize_method(gcm_p)

        _, alpha_adjustor = calibrators["gcm"]
        results["gcm_cal"] = summarize_method(gcm_p, alpha_adjust=alpha_adjustor)

    if include_hrt:
        hrt_p = hrt_pvalues(
            inst,
            seed=args.seed + run_idx,
            n_components=HRT_DEFAULT_COMPONENTS,
            n_steps=HRT_DEFAULT_STEPS,
            estimator_type=hrt_estimator,
        )
        results["hrt"] = summarize_method(hrt_p)

        _, alpha_adjustor = calibrators["hrt"]
        results["hrt_cal"] = summarize_method(hrt_p, alpha_adjust=alpha_adjustor)

    results["contra"] = {}
    for method in args.contra_methods:
        try:
            pvals_contra = contra_pvalues(
                inst,
                method=method,
                seed=args.seed + run_idx,
                response_type=config.response,
            )
            results["contra"][method] = summarize_method(pvals_contra)
        except Exception as exc:
            print(f"Warning: CONTRA {method} failed on run {run_idx}: {exc}")
            zero_curve = {
                "power": np.zeros_like(alpha_grid),
                "valid": np.zeros_like(alpha_grid),
                "fdr": np.ones_like(alpha_grid),
            }
            results["contra"][method] = zero_curve

    return run_idx, results


def run_experiment(args: argparse.Namespace) -> None:
    selected_tests = set(args.tests or [])
    args.tests = list(selected_tests)

    config = BenchmarkConfig(
        n=args.n,
        m=args.m,
        active_features=args.active_features,
        gamma=args.gamma,
        x_distribution=args.x_dist,
        response=args.response
    )

    include_gcm = "gcm" in selected_tests
    include_hrt = "hrt" in selected_tests

    worker_args = [
        (run_idx, config, args)
        for run_idx in range(args.runs)
    ]

    if args.no_parallel:
        print(f"Running {args.runs} experiments sequentially...")
        all_results = [run_single_experiment(worker_arg) for worker_arg in worker_args]
    else:
        print(f"Running {args.runs} experiments in parallel...")
        all_results = Parallel(n_jobs=-1)(
            delayed(run_single_experiment)(worker_arg)
            for worker_arg in worker_args
        )

    payloads = [res for _, res in all_results]
    alpha_grid, summary = summarize_benchmark_runs(
        payloads,
        include_gcm,
        include_hrt,
        args.contra_methods,
    )

    print(
        "Configuration: x_dist={x}, response={resp}, tests={tests}, n={n}, m={m}, active={active}, gamma={gamma}".format(
            x=args.x_dist,
            resp=args.response,
            tests=",".join(sorted(selected_tests)) if selected_tests else "none",
            n=config.n,
            m=config.m,
            active=config.active_features,
            gamma=config.gamma
        )
    )
    print(f"\nResults averaged over {len(payloads)} runs:")

    method_order = [
        "GCM",
        "Calibrated GCM",
        "HRT",
        "Calibrated HRT",
        "CONTRA-HRT",
        "CONTRA-FASTCRT",
    ]
    if alpha_grid is not None:
        for idx, alpha in enumerate(alpha_grid):
            print(f"\nalpha={alpha:.2f}")
            for label in method_order:
                stats = summary.get(label)
                if not stats:
                    continue
                valid_mean = stats["valid_mean"][idx]
                power_mean = stats["power_mean"][idx]
                fdr_mean = stats["fdr_mean"][idx]
                count_arr = stats.get("count")
                count_val = int(count_arr[idx]) if count_arr is not None and idx < len(count_arr) else 0
                if np.isnan(valid_mean):
                    continue
                print(
                    f"  {label:<18s} | n={count_val:3d} valid={valid_mean:.3f} power={power_mean:.3f} fdr={fdr_mean:.3f}"
                )

    export_benchmark_tables(args, config, alpha_grid, summary)


def summarize_benchmark_runs(
    result_payloads: list[Dict[str, dict]],
    include_gcm: bool,
    include_hrt: bool,
    contra_methods: list[str],
) -> tuple[Optional[np.ndarray], Dict[str, dict]]:
    if not result_payloads:
        return None, {}

    alpha_values: set[float] = set()
    metrics = ("power", "valid", "fdr")
    store: Dict[str, Dict[float, Dict[str, list[float]]]] = {}

    def _register(label: str, curves: dict, alphas: np.ndarray) -> None:
        if label not in store:
            store[label] = {}
        for idx, alpha in enumerate(alphas):
            alpha_val = float(alpha)
            alpha_values.add(alpha_val)
            bucket = store[label].setdefault(alpha_val, {m: [] for m in metrics})
            for metric in metrics:
                bucket[metric].append(float(curves[metric][idx]))

    for payload in result_payloads:
        alphas = np.asarray(payload.get("alphas"), dtype=float)
        if alphas.ndim == 0:
            alphas = np.atleast_1d(alphas)
        if alphas.size == 0:
            continue

        if include_gcm and "gcm" in payload:
            _register("GCM", payload["gcm"], alphas)
            if "gcm_cal" in payload:
                _register("Calibrated GCM", payload["gcm_cal"], alphas)

        if include_hrt and "hrt" in payload:
            _register("HRT", payload["hrt"], alphas)
            if "hrt_cal" in payload:
                _register("Calibrated HRT", payload["hrt_cal"], alphas)

        contra_payload = payload.get("contra", {})
        for method in contra_methods:
            if method in contra_payload:
                label = f"CONTRA-{method.upper()}"
                _register(label, contra_payload[method], alphas)

    if not alpha_values:
        return None, {}

    alpha_grid = np.array(sorted(alpha_values), dtype=float)
    summary: Dict[str, dict] = {}
    for label, buckets in store.items():
        summary[label] = {
            "count": np.zeros_like(alpha_grid),
        }
        for metric in metrics:
            summary[label][f"{metric}_mean"] = np.full_like(alpha_grid, np.nan, dtype=float)
            summary[label][f"{metric}_std"] = np.full_like(alpha_grid, np.nan, dtype=float)

        for idx, alpha in enumerate(alpha_grid):
            bucket = buckets.get(float(alpha))
            if not bucket:
                continue
            runs = len(bucket["power"])
            if runs == 0:
                continue
            summary[label]["count"][idx] = runs
            for metric in metrics:
                arr = np.asarray(bucket[metric], dtype=float)
                summary[label][f"{metric}_mean"][idx] = float(arr.mean())
                if arr.size > 1:
                    summary[label][f"{metric}_std"][idx] = float(arr.std(ddof=1))
                else:
                    summary[label][f"{metric}_std"][idx] = 0.0

    return alpha_grid, summary


def export_benchmark_tables(
    args: argparse.Namespace,
    config: BenchmarkConfig,
    alpha_grid: Optional[np.ndarray],
    method_summary: Dict[str, dict],
) -> None:
    if alpha_grid is None or not method_summary:
        return

    tables_dir = Path("outputs") / "benchmarks" / f"{args.x_dist}_{args.response}_{args.metric}"
    tables_dir.mkdir(parents=True, exist_ok=True)

    alpha_cols = [f"alpha={a:.2f}" for a in alpha_grid]
    method_order = [
        "GCM",
        "Calibrated GCM",
        "HRT",
        "Calibrated HRT",
        "CONTRA-HRT",
        "CONTRA-FASTCRT",
    ]

    def _build_rows(metric_key: str, highlight_fn) -> tuple[list[list[str]], list[list[str]]]:
        csv_rows: list[list[str]] = []
        latex_rows: list[list[str]] = []
        for label in method_order:
            stats = method_summary.get(label)
            if not stats:
                continue
            counts = np.asarray(stats.get("count"))
            means = np.asarray(stats.get(f"{metric_key}_mean"))
            stds = np.asarray(stats.get(f"{metric_key}_std"))
            csv_vals = [label]
            latex_vals = [label]
            for idx, alpha in enumerate(alpha_grid):
                mean_val = means[idx] if idx < means.size else np.nan
                std_val = stds[idx] if idx < stds.size else np.nan
                count_val = counts[idx] if counts is not None and idx < counts.size else 0.0
                if np.isnan(mean_val) or count_val <= 0:
                    entry = "--"
                    latex_entry = entry
                else:
                    se = std_val / np.sqrt(max(count_val, 1.0))
                    entry = f"{mean_val:.2f}±{se:.3f}"
                    latex_entry = entry
                    if highlight_fn(label, idx, mean_val, se, alpha):
                        latex_entry = f"\\textbf{{{entry}}}"
                csv_vals.append(entry)
                latex_vals.append(latex_entry)
            csv_rows.append(csv_vals)
            latex_rows.append(latex_vals)
        return csv_rows, latex_rows

    valid_best = []
    for idx in range(len(alpha_grid)):
        best = -np.inf
        for label in method_order:
            stats = method_summary.get(label)
            if not stats:
                continue
            val = stats["valid_mean"][idx]
            if np.isnan(val):
                continue
            best = max(best, val)
        valid_best.append(best)

    def _highlight_valid(label: str, idx: int, mean: float, _se: float, _alpha: float) -> bool:
        if np.isnan(mean) or np.isnan(valid_best[idx]):
            return False
        return np.isclose(mean, valid_best[idx])

    def _highlight_fdr(_label: str, idx: int, mean: float, _se: float, alpha: float) -> bool:
        if np.isnan(mean):
            return False
        return mean <= alpha

    valid_csv, valid_latex = _build_rows("valid", _highlight_valid)
    fdr_csv, fdr_latex = _build_rows("fdr", _highlight_fdr)

    if valid_csv:
        valid_df = pd.DataFrame(valid_csv, columns=["Method"] + alpha_cols)
        valid_df.to_csv(tables_dir / "valid_power_table.csv", index=False)
        valid_latex_df = pd.DataFrame(valid_latex, columns=["Method"] + alpha_cols)
        response_caption = _response_caption(args.response)
        caption = (
            f"\\textbf{{{response_caption}}} Valid Power across $\\alpha$ levels. "
            f"$n={config.n}$, $m={config.m}$. Bold = best valid power per $\\alpha$."
        )
        valid_latex_df.to_latex(
            tables_dir / "valid_power_table.tex",
            index=False,
            escape=False,
            caption=caption,
            label=f"tab:benchmark_valid_{args.x_dist}_{args.response}_{args.metric}",
            column_format='l' + 'c' * len(alpha_cols)
        )

    if fdr_csv:
        fdr_df = pd.DataFrame(fdr_csv, columns=["Method"] + alpha_cols)
        fdr_df.to_csv(tables_dir / "fdr_table.csv", index=False)
        fdr_latex_df = pd.DataFrame(fdr_latex, columns=["Method"] + alpha_cols)
        caption = (
            f"\\textbf{{FDR at Target $\\alpha$}} for {args.response.title()} response, $n={config.n}$, $m={config.m}$. "
            "Bold entries satisfy $\\widehat{\\mathrm{FDR}} \leq \alpha$."
        )
        fdr_latex_df.to_latex(
            tables_dir / "fdr_table.tex",
            index=False,
            escape=False,
            caption=caption,
            label=f"tab:benchmark_fdr_{args.x_dist}_{args.response}_{args.metric}",
            column_format='l' + 'c' * len(alpha_cols)
        )

    print(f"Benchmark tables saved to {tables_dir}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GCM variants against CONTRA.")
    parser.add_argument("--runs", type=int, default=8, help="Number of independent datasets to evaluate.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Target FDR level for reporting.")
    parser.add_argument(
        "--alpha-levels",
        type=float,
        nargs="*",
        default=None,
        help="Explicit alpha levels to evaluate (defaults to 0.05, 0.10, 0.15, 0.20).",
    )
    parser.add_argument(
        "--metric",
        choices=("type1", "fdp", "area"),
        default="type1",
        help="Calibration metric.",
    )
    parser.add_argument(
        "--contra-methods",
        nargs="*",
        choices=("crt", "hrt", "fastcrt"),
        default=["crt", "hrt", "fastcrt"],
        help="Which CONTRA variants to run.",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        choices=("gcm", "hrt"),
        default=["gcm", "hrt"],
        help="Which main benchmark tests to run (GCM, HRT). Use empty list to run only CONTRA methods.",
    )
    parser.add_argument(
        "--x-dist",
        choices=("normal", "correlated", "laplace", "gdsc", "nonlinear"),
        default="normal",
        help="Feature distribution passed to make_X.",
    )
    parser.add_argument(
        "--response",
        choices=("linear", "linear_continuous", "orange", "nonlinear"),
        default="nonlinear",
        help="Response type / signal structure.",
    )
    parser.add_argument("--n", type=int, default=50, help="Number of samples per dataset draw.")
    parser.add_argument("--m", type=int, default=10, help="Total feature count.")
    parser.add_argument("--active-features", type=int, default=4, help="Number of informative features.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Correlation parameter for Gaussian features.")
    parser.add_argument("--num-epochs", type=int, default=10, help="Epochs used in calibration training.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--no-parallel", action="store_true", help="Disable joblib parallelism.")
    args = parser.parse_args(argv)
    if args.alpha_levels is None or len(args.alpha_levels) == 0:
        args.alpha_levels = ALPHA_TABLE.copy()
    else:
        args.alpha_levels = np.array(args.alpha_levels, dtype=float)
    if args.metric == "area":
        args.metric = "type1"
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
