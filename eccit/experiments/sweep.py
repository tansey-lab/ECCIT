from collections import defaultdict
import numpy as np
import torch
from itertools import product
from joblib import Parallel, delayed

from eccit.calibration_runner import calibrate_step
from eccit.cits import compute_conditional_pvals
from eccit.utils.helpers import make_X, make_Y, summarize, make_alpha_adjuster, make_alpha_adjuster_from_fdp, eval_performance
from eccit.utils.sgd_factor_model import factorize


def sweep_calibration(n_list=tuple(range(25, 501, 25)),
                      m_list=(10,25,50),
                      dist_list=("normal", "correlated", "laplace"),
                      num_runs=10,
                      use_linear=True,
                      metric="fdp",
                      test="gcm",
                      gcm_kwargs=None,
                      kcit_kwargs=None,
                      rcit_kwargs=None,
                      hrt_kwargs=None):
    """
    Run calibration sweep across different data configurations.

    Tests calibration performance across different sample sizes, feature counts,
    and data distributions.
    """
    # if test == "hrt":
    #     n_list = (500, 1000)

    if metric == "area":
        metric = "type1"

    tasks = [
        (distribution, n, m)
        for distribution, n, m in product(dist_list, n_list, m_list)
        if m < n
        for _ in range(num_runs)
    ]

    # Run calibration jobs in parallel
    raw_outs = Parallel(n_jobs=-1, verbose=10)(
        delayed(calibrate_step)(
            n,
            m,
            distribution,
            use_linear,
            metric,
            test,
            gcm_kwargs,
            kcit_kwargs,
            rcit_kwargs,
            hrt_kwargs,
        )
        for distribution, n, m in tasks
    )

    # Group results by (distribution, n, m)
    grouped = defaultdict(list)
    for key, val in raw_outs:
        grouped[key].append(val)

    results = {}
    for (distribution, n, m), runs in grouped.items():
        if metric == "fdp":
            # Aggregate FDP curves into an alpha adjustor; p-value calibrator is identity
            diag0 = runs[0][7]
            alpha_grid = diag0.get('alpha_grid_fdp', np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]))
            fdp_curves = np.vstack([r[7]['fdp_curve'] for r in runs])
            fdp_mean = np.maximum.accumulate(np.clip(fdp_curves.mean(axis=0), 0.0, 1.0))
            alpha_adjustor = make_alpha_adjuster_from_fdp(alpha_grid, fdp_mean)
            calibrator = lambda p: p
            results[(distribution, n, m)] = dict(
                alpha_grid=alpha_grid,
                fdp_mean=fdp_mean,
                all_nulls=np.concatenate([r[4] for r in runs]),
                calibrator=calibrator,
                alpha_adjustor=alpha_adjustor,
                max_area=max(r[3] for r in runs),
            )
        else:
            # CDF-based path (type-I calibration)
            curves = np.vstack([r[5] for r in runs])
            joint_cdf = np.clip(np.maximum.accumulate(curves.max(axis=0)), 0, 1)
            grid = runs[0][6]
            if metric == "type1":
                calibrator = lambda p: np.asarray(p, dtype=float)
            else:
                def calibrator(p_raw):
                    p_cal = np.interp(p_raw, grid, joint_cdf, 0.0, 1.0)
                    return p_cal
            alpha_adjustor = make_alpha_adjuster(grid, joint_cdf)
            results[(distribution, n, m)] = dict(
                grid=grid,
                joint_cdf=joint_cdf,
                mean_cdf=curves.mean(axis=0),
                all_nulls=np.concatenate([r[4] for r in runs]),
                calibrator=calibrator,
                alpha_adjustor=alpha_adjustor,
                max_type1=max(r[3] for r in runs),
            )

    return results


def sweep_performance(results,
                      alphas=(0.0,0.05,0.1,0.15,0.2,0.25,0.3),
                      n_responses=100,
                      use_linear=True,
                      response_order=2,
                      metric="fdp",
                      test="gcm",
                      gcm_kwargs=None,
                      kcit_kwargs=None,
                      rcit_kwargs=None,
                      hrt_kwargs=None):
    """
    Evaluate performance of calibrated p-values from sweep results.

    Tests FDR control and statistical power across different alpha levels.

    Parameters:
    -----------
    response_order : int
        Order of the response function (1=linear, 2=nonlinear)
    """
    if metric == "area":
        metric = "type1"
    perf = {}
    features_to_construct = {10: 5, 25: 8, 50: 10}

    for (distribution, n, m), val in results.items():
        calibrator = val['calibrator']
        alpha_adjustor = val.get('alpha_adjustor', None)

        raw_list, cal_list, sel_list = [], [], []
        # Feature size based on m
        feat_size = features_to_construct.get(m, max(3, int(0.1*m)))

        for _ in range(n_responses):
            # Generate test data
            X_np = make_X(n, m, distribution=distribution)
            X_t = torch.from_numpy(X_np).float()

            # Generate response with known active features using make_Y
            Y_sim, sel = make_Y(X_np, feat_size, order=response_order, noise_scale=1.0)
            Y_t = torch.from_numpy(Y_sim).float()

            # Fit factor model if using HRT test
            local_hrt_kwargs = dict(hrt_kwargs or {})
            if test == "hrt":
                # Set classifier type based on whether we're using linear test
                if "estimator_type" not in local_hrt_kwargs:
                    local_hrt_kwargs["estimator_type"] = "linear" if use_linear else "mlp"

                if "W_hat" not in local_hrt_kwargs:
                    print(f"Fitting factor model for HRT performance evaluation with {m} features...")
                    _, _, W_hat, V_hat, U_hat, _ = factorize(
                        X_np,
                        n_components=local_hrt_kwargs.get('hrt_n_components', 10),
                        n_steps=local_hrt_kwargs.get('hrt_n_steps', 1000),
                        likelihood=local_hrt_kwargs.get('hrt_likelihood', 'gaussian')
                    )
                    local_hrt_kwargs.update({
                        "W_hat": W_hat,
                        "V_hat": V_hat,
                        "U_hat": U_hat,
                    })

            # Compute p-values
            p_raw = compute_conditional_pvals(
            X_t,
            Y_t,
            test=test,
            order=1,
            use_linear=(use_linear if test == "gcm" else True),
            kcit_kwargs=kcit_kwargs,
            rcit_kwargs=rcit_kwargs,
            hrt_kwargs=local_hrt_kwargs,
            gcm_kwargs=gcm_kwargs,
            to_numpy=True,
        )

            p_cal = calibrator(p_raw)
            raw_list.append(p_raw)
            cal_list.append(p_cal)
            sel_list.append(sel)

        # Summarize performance
        alphas = np.asarray(alphas)
        fdr_r, se_r, pw_r, se_pw_r, vpw_r, se_vpw_r = summarize(raw_list, sel_list, alphas, n_responses, alpha_adjust=None)
        fdr_c, se_c, pw_c, se_pw_c, vpw_c, se_vpw_c = summarize(raw_list, sel_list, alphas, n_responses, alpha_adjust=alpha_adjustor)

        perf[(distribution, n, m)] = dict(
            alphas=alphas,
            fdr_raw=fdr_r, fdr_raw_se=se_r,
            pow_raw=pw_r, pow_raw_se=se_pw_r,
            valid_pow_raw=vpw_r, valid_pow_raw_se=se_vpw_r,
            fdr_cal=fdr_c, fdr_cal_se=se_c,
            pow_cal=pw_c, pow_cal_se=se_pw_c,
            valid_pow_cal=vpw_c, valid_pow_cal_se=se_vpw_c,
        )

        target_alpha = 0.2
        raw_vp_vals = []
        cal_vp_vals = []
        gain_vals = []
        for pvals, sel in zip(raw_list, sel_list):
            vp_raw, _, _ = eval_performance(pvals, sel, alpha=target_alpha, alpha_adjust=None)
            vp_cal, _, _ = eval_performance(pvals, sel, alpha=target_alpha, alpha_adjust=alpha_adjustor)
            raw_vp_vals.append(vp_raw)
            cal_vp_vals.append(vp_cal)
            gain_vals.append(vp_cal - vp_raw)

        raw_vp_vals = np.asarray(raw_vp_vals, dtype=float)
        cal_vp_vals = np.asarray(cal_vp_vals, dtype=float)
        gain_vals = np.asarray(gain_vals, dtype=float)

        def _std(arr):
            if arr.size <= 1:
                return 0.0
            return float(np.std(arr, ddof=1))

        perf[(distribution, n, m)].update({
            'target_alpha': float(target_alpha),
            'valid_power_raw_mean': float(np.mean(raw_vp_vals)) if raw_vp_vals.size else 0.0,
            'valid_power_raw_std': _std(raw_vp_vals),
            'valid_power_cal_mean': float(np.mean(cal_vp_vals)) if cal_vp_vals.size else 0.0,
            'valid_power_cal_std': _std(cal_vp_vals),
            'valid_power_gain_mean': float(np.mean(gain_vals)) if gain_vals.size else 0.0,
            'valid_power_gain_std': _std(gain_vals),
        })

    return perf


def run_sweep_experiment(metric="fdp", output_dir="outputs", test="gcm",
                         use_linear=True, response_order=2, gcm_kwargs=None,
                         kcit_kwargs=None, rcit_kwargs=None, hrt_kwargs=None):
    """
    Run complete sweep experiment with specified metric.

    Main entry point for sweep experiments.

    Parameters:
    -----------
    response_order : int
        Order of the response function (1=linear, 2=nonlinear)
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create sweep-specific subdirectory
    from pathlib import Path
    sweep_dir = Path(output_dir) / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    if metric == "area":
        metric = "type1"
    print(f"Running sweep experiment with metric: {metric}")

    # Run calibration sweep
    results = sweep_calibration(
        metric=metric,
        test=test,
        use_linear=use_linear,
        gcm_kwargs=gcm_kwargs,
        kcit_kwargs=kcit_kwargs,
        rcit_kwargs=rcit_kwargs,
        hrt_kwargs=hrt_kwargs,
    )

    # Plot CDFs and calibration offsets
    from eccit.experiments.plots import plot_cdf_sweep, plot_calibration_offset, plot_perf_sweep
    plot_cdf_sweep(results, out_dir=sweep_dir)
    plot_calibration_offset(results, out_dir=sweep_dir, metric=metric)

    # Evaluate performance
    perf_results = sweep_performance(
        results,
        metric=metric,
        use_linear=use_linear,
        response_order=response_order,
        test=test,
        gcm_kwargs=gcm_kwargs,
        kcit_kwargs=kcit_kwargs,
    )

    # Plot performance
    plot_perf_sweep(perf_results, out_dir=sweep_dir, metric=metric)
    
    return results, perf_results
