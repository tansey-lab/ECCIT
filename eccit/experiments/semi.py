"""Semi-supervised setting based on GDSC feature matrices."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from eccit.contra import amicrt as contra
from sklearn.ensemble import RandomForestRegressor
from eccit.calibration_runner import calibrate_run
from eccit.cits import compute_conditional_pvals
from eccit.utils.helpers import make_Y, summarize


_DEFAULT_GDSC_PATH = Path("gdsc_all_features.csv")
_ALPHA_GRID = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=float)


def read_gdsc(
    path: Optional[Union[Path, str]] = None,
    *,
    feature_prefix: str = "EXP:",
    n_features: int = None,
    dropna: bool = True,
) -> np.ndarray:
    """Load and standardise the GDSC expression features.

    Parameters
    ----------
    path:
        Path to the CSV file containing the full GDSC feature matrix.
        Defaults to ``gdsc_all_features.csv``.
    feature_prefix:
        Column prefix identifying continuous features to include.  The
        default selects the expression panels (``EXP:``).
    n_features:
        Optional cap on the number of features retained.  When provided,
        the function keeps the highest-variance columns.
    dropna:
        If ``True`` (default) rows containing missing or infinite values
        are removed prior to scaling.  Otherwise missing entries are
        imputed with zero.

    Returns
    -------
    np.ndarray
        A centred and unit-variance matrix ``(n_samples, n_features)``.
    """

    data_path = Path(path or _DEFAULT_GDSC_PATH)
    if not data_path.exists():
        raise FileNotFoundError(
            f"GDSC feature file not found at {data_path}.",
        )

    df = pd.read_csv(data_path, index_col=0)
    feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    if not feature_cols:
        raise ValueError(
            f"No columns with prefix '{feature_prefix}' in {data_path.name}.",
        )

    df_feat = df[feature_cols].select_dtypes(include=[np.number])
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

    if dropna:
        df_feat = df_feat.dropna(axis=0, how="any")
    else:
        df_feat = df_feat.fillna(0.0)

    if n_features is not None and n_features < df_feat.shape[1]:
        variances = df_feat.var(axis=0)
        top_cols = variances.nlargest(n_features).index
        df_feat = df_feat[top_cols]

    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat.values)
    return X.astype(np.float64)


def _sample_matrix(
    X: np.ndarray,
    n: int,
    m: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample rows/columns from GDSC feature matrix."""
    n_total, m_total = X.shape
    row_idx = rng.choice(n_total, size=n, replace=n > n_total)
    col_idx = rng.choice(m_total, size=min(m, m_total), replace=False)
    X_sub = X[row_idx][:, col_idx]
    if m > m_total:
        repeats = math.ceil(m / m_total)
        tiled = np.tile(X[row_idx], (1, repeats))
        X_sub = tiled[:, :m]
    return X_sub


def _order_from_response(response: str) -> int:
    resp = response.lower()
    if resp not in {"linear", "nonlinear"}:
        raise ValueError(f"Unsupported response type '{response}'.")
    return 1 if resp == "linear" else 2


def _compute_contra_pvals(X: np.ndarray, y: np.ndarray, method: str, seed: int) -> np.ndarray:
    """Compute CONTRA p-values for semi experiment."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )

    method = method.lower()
    is_binary = np.unique(y).size <= 2
    common_kwargs = dict(tqdm=lambda it, **_: it)
    ordering = '>'

    if not is_binary:
        # Continuous response (most common for semi)
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
        common_kwargs["CompleteConditionalClass"] = contra.conditionals.ContinuousConditional
        common_kwargs["CompleteConditionalArgs"] = dict(
            modelClass=RandomForestRegressor,
            modelArgs=dict(n_estimators=10, random_state=seed)
        )
    else:
        # Binary response (rare for semi)
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


def run_semi_experiment(
    *,
    n: int,
    m: int,
    test: str = "gcm",
    metric: str = "fdp",
    response: str = "linear",
    num_responses: int = 100,
    feat_size: Optional[int] = None,
    seed: Optional[int] = None,
    gdsc_path: Optional[Union[Path, str]] = None,
    alpha_train: float = 0.2,
    gcm_kwargs: Optional[dict] = None,
    kcit_kwargs: Optional[dict] = None,
    rcit_kwargs: Optional[dict] = None,
    hrt_kwargs: Optional[dict] = None,
    hrt_n_components: int = 3,
    hrt_n_steps: int = 300,
    hrt_likelihood: str = "gaussian",
    contra_methods: Optional[List[str]] = None,
) -> Dict[str, Union[np.ndarray, float, int, str]]:
    """Run a single semi-supervised calibration experiment on GDSC data."""

    if metric == "area":
        metric = "type1"

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X_full = read_gdsc(gdsc_path)
    X_sub = _sample_matrix(X_full, n=n, m=m, rng=rng)
    X_t = torch.from_numpy(X_sub).float()

    order = _order_from_response(response)
    feat_k = feat_size or int(0.1 * m)
    if m == 20:
        feat_k = 8

    use_linear = test.lower() == "gcm" and order == 1

    local_gcm_kwargs = dict(gcm_kwargs or {})
    local_kcit_kwargs = dict(kcit_kwargs or {})
    local_rcit_kwargs = dict(rcit_kwargs or {})
    local_hrt_kwargs = dict(hrt_kwargs or {})
    if test.lower() == "gcm" and order > 1 and not local_gcm_kwargs:
        local_gcm_kwargs = {
            "y_estimator": "linear",
            "x_estimator": "linear",
        }
    if test.lower() == "hrt":
        if order > 1:
            local_hrt_kwargs.setdefault("use_cv", True)
            local_hrt_kwargs.setdefault("n_folds", 2)
            local_hrt_kwargs.pop("conditional_type", None)
            local_hrt_kwargs.pop("conditional_kwargs", None)
        else:
            local_hrt_kwargs.setdefault("conditional_type", "linear")

    calibrator, *_extras, diagnostics = calibrate_run(
        X_t,
        metric=metric,
        order_adv=order,
        order_test=order,
        alpha_train=alpha_train,
        use_linear=use_linear,
        test=test,
        gcm_kwargs=local_gcm_kwargs or None,
        kcit_kwargs=local_kcit_kwargs or None,
        rcit_kwargs=local_rcit_kwargs or None,
        hrt_kwargs=local_hrt_kwargs or None,
        hrt_n_components=hrt_n_components,
        hrt_n_steps=hrt_n_steps,
        hrt_likelihood=hrt_likelihood,
    )

    alpha_adjust = diagnostics.get(
        "alpha_adjust_fdp" if metric == "fdp" else "alpha_adjust",
    )

    # Extract HRT components if test is HRT
    W_hat = diagnostics.get('W_hat')
    V_hat = diagnostics.get('V_hat')
    U_hat = diagnostics.get('U_hat')

    # Add HRT components to hrt_kwargs if using HRT test
    if test.lower() == 'hrt' and W_hat is not None:
        local_hrt_kwargs = dict(local_hrt_kwargs or {})
        local_hrt_kwargs['W_hat'] = W_hat
        local_hrt_kwargs['V_hat'] = V_hat
        local_hrt_kwargs['U_hat'] = U_hat
        # Set classifier type based on order
        if 'estimator_type' not in local_hrt_kwargs:
            local_hrt_kwargs['estimator_type'] = "linear"

    raw_list: List[np.ndarray] = []
    sel_list: List[Sequence[int]] = []

    # Initialize CONTRA results
    contra_methods = contra_methods or []
    contra_pvals: Dict[str, List[np.ndarray]] = {method: [] for method in contra_methods}

    for _ in range(num_responses):
        y_np, sel = make_Y(X_sub, feat_k, order=order)
        Y_t = torch.from_numpy(y_np).float()
        p_raw = compute_conditional_pvals(
            X_t,
            Y_t,
            test=test,
            order=order,
            use_linear=use_linear,
            gcm_kwargs=local_gcm_kwargs or None,
            kcit_kwargs=local_kcit_kwargs or None,
            rcit_kwargs=local_rcit_kwargs or None,
            hrt_kwargs=local_hrt_kwargs or None,
            to_numpy=True,
        )
        raw_list.append(p_raw)
        sel_list.append(sel)

        # Run CONTRA tests
        for method in contra_methods:
            pvals_contra = _compute_contra_pvals(X_sub, y_np, method, seed=(seed or 0) + _)
            contra_pvals[method].append(pvals_contra)

    alphas = _ALPHA_GRID.copy()
    fdr_r, se_r, pow_r, se_pow_r, vpw_r, se_vpw_r = summarize(
        raw_list,
        sel_list,
        alphas,
        num_responses,
        alpha_adjust=None,
    )
    fdr_c, se_c, pow_c, se_pow_c, vpw_c, se_vpw_c = summarize(
        raw_list,
        sel_list,
        alphas,
        num_responses,
        alpha_adjust=alpha_adjust,
    )

    vpw_gain = np.array(vpw_c) - np.array(vpw_r)
    vpw_gain_se = np.sqrt(np.array(se_vpw_c) ** 2 + np.array(se_vpw_r) ** 2)

    # Summarize CONTRA results
    contra_results = {}
    for method in contra_methods:
        fdr_contra, se_contra, pow_contra, se_pow_contra, vpw_contra, se_vpw_contra = summarize(
            contra_pvals[method],
            sel_list,
            alphas,
            num_responses,
            alpha_adjust=None,
        )
        contra_results[method] = {
            "fdr": np.array(fdr_contra),
            "fdr_se": np.array(se_contra),
            "pow": np.array(pow_contra),
            "pow_se": np.array(se_pow_contra),
            "valid_pow": np.array(vpw_contra),
            "valid_pow_se": np.array(se_vpw_contra),
        }

    return {
        "response": response,
        "n": n,
        "m": m,
        "test": test,
        "metric": metric,
        "num_responses": num_responses,
        "alphas": alphas,
        "fdr_raw": np.array(fdr_r),
        "fdr_raw_se": np.array(se_r),
        "pow_raw": np.array(pow_r),
        "pow_raw_se": np.array(se_pow_r),
        "valid_pow_raw": np.array(vpw_r),
        "valid_pow_raw_se": np.array(se_vpw_r),
        "fdr_cal": np.array(fdr_c),
        "fdr_cal_se": np.array(se_c),
        "pow_cal": np.array(pow_c),
        "pow_cal_se": np.array(se_pow_c),
        "valid_pow_cal": np.array(vpw_c),
        "valid_pow_cal_se": np.array(se_vpw_c),
        "valid_pow_gain": vpw_gain,
        "valid_pow_gain_se": vpw_gain_se,
        "contra": contra_results,
        "calibrator": calibrator,
        "diagnostics": diagnostics,
    }


def aggregate_semi_runs(run_results: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Aggregate multiple semi-experiment runs for a single response type."""
    if not run_results:
        return {}

    fields = [
        "fdr_raw",
        "fdr_cal",
        "pow_raw",
        "pow_cal",
        "valid_pow_raw",
        "valid_pow_cal",
        "valid_pow_gain",
    ]

    aggregated: Dict[str, np.ndarray] = {
        "alphas": run_results[0]["alphas"],
        "num_runs": len(run_results),
        "response": run_results[0]["response"],
        "test": run_results[0]["test"],
        "runs": run_results,
    }

    for field in fields:
        stack = np.stack([np.asarray(run[field]) for run in run_results], axis=0)
        aggregated[f"{field}_mean"] = stack.mean(axis=0)
        aggregated[f"{field}_std"] = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(stack[0])

    # Aggregate CONTRA results if present
    if "contra" in run_results[0] and run_results[0]["contra"]:
        contra_methods = list(run_results[0]["contra"].keys())
        contra_aggregated = {}
        for method in contra_methods:
            contra_aggregated[method] = {}
            for metric in ["fdr", "pow", "valid_pow"]:
                stack = np.stack([np.asarray(run["contra"][method][metric]) for run in run_results], axis=0)
                contra_aggregated[method][f"{metric}_mean"] = stack.mean(axis=0)
                contra_aggregated[method][f"{metric}_std"] = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(stack[0])
        aggregated["contra"] = contra_aggregated

    return aggregated


__all__ = ["read_gdsc", "run_semi_experiment", "aggregate_semi_runs"]
