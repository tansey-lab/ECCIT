from collections import defaultdict
import numpy as np
import torch
from pathlib import Path
from joblib import Parallel, delayed

from eccit.calibration_runner import calibrate_step_order
from eccit.cits import compute_conditional_pvals
from eccit.utils.helpers import make_X, make_Y, summarize, make_alpha_adjuster, make_alpha_adjuster_from_fdp


def run_second_order_calibration(n=100, m=50, distribution="correlated", 
                                 num_cal_runs=1, metric="fdp", test="gcm",
                                 gcm_kwargs=None, kcit_kwargs=None):
    """
    Run calibration for different adversarial and test orders.
    
    Tests combinations of 1st/2nd order adversarial Y generation
    and 1st/2nd order GCM testing.
    """
    if metric == "area":
        metric = "type1"
    orders = [1, 2]
    alpha_trains = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    cal_tasks = [
        (oa, ot, a)
        for oa in orders
        for ot in orders
        for a in alpha_trains
        for _ in range(num_cal_runs)
    ]

    # Run calibration jobs in parallel
    raw_cal = Parallel(n_jobs=-1, verbose=10)(
        delayed(calibrate_step_order)(
            n,
            m,
            distribution,
            oa,
            ot,
            a,
            metric,
            test,
            (
                gcm_kwargs
                if gcm_kwargs is not None
                else (
                    {"y_estimator": "linear", "x_estimator": "linear"}
                    if ot == 1
                    else {
                        "y_estimator": "poly2",
                        "y_estimator_params": {"lambda": 1e-1},
                        "x_estimator": "linear",
                    }
                )
            ) if test == "gcm" else None,
            kcit_kwargs,
            None,
            (
                {"estimator_type": "linear", "conditional_type": "linear"}
                if ot == 1
                else {
                    "estimator_type": "poly2",
                    "estimator_params": {"lambda": 1e-1},
                    "conditional_type": "poly2",
                    "conditional_kwargs": {"lambda": 1e-1},
                }
            ) if test == "hrt" else None,
        ) 
        for oa, ot, a in cal_tasks
    )

    # Group results by (order_adv, order_test)
    cal_grouped = defaultdict(list)
    for (oa, ot, a), val in raw_cal:
        cal_grouped[(oa, ot)].append(val)
        
    cal_results = {}
    for (oa, ot), runs in cal_grouped.items():
        if metric == "fdp":
            diag0 = runs[0][7]
            alpha_grid = diag0.get('alpha_grid_fdp', np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]))
            fdp_curves = np.vstack([r[7]['fdp_curve'] for r in runs])
            fdp_mean = np.maximum.accumulate(np.clip(fdp_curves.mean(axis=0), 0.0, 1.0))
            alpha_adjustor = make_alpha_adjuster_from_fdp(alpha_grid, fdp_mean)
            cal_results[(oa, ot)] = dict(
                alpha_grid=alpha_grid,
                fdp_mean=fdp_mean,
                calibrator=(lambda p: p),
                alpha_adjustor=alpha_adjustor
            )
        else:
            curves = np.vstack([r[5] for r in runs])
            joint_cdf = np.clip(np.maximum.accumulate(curves.max(axis=0)), 0, 1)
            grid = runs[0][6]
            if metric == "type1":
                calibrator = (lambda p: np.asarray(p, dtype=float))
            else:
                calibrator = (lambda g, c: (lambda p: np.interp(p, g, c, 0.0, 1.0)))(grid, joint_cdf)
            cal_results[(oa, ot)] = dict(
                grid=grid,
                joint_cdf=joint_cdf,
                calibrator=calibrator,
                alpha_adjustor=make_alpha_adjuster(grid, joint_cdf)
            )

    return cal_results


def evaluate_second_order_performance(cal_results, n=100, m=50, distribution="correlated",
                                     n_responses=100, metric="fdp", test="gcm",
                                     gcm_kwargs=None, kcit_kwargs=None,
                                     rcit_kwargs=None, hrt_kwargs=None):
    """
    Test performance across different ground truth and test orders.
    
    Evaluates how well calibration works when the true model order
    differs from the test order.
    """
    orders = [1, 2]
    feat_size = max(3, int(0.1 * m))
    perf_results = {}
    
    for oa, ot in cal_results.keys():
        calibrator = cal_results[(oa, ot)]['calibrator']
        alpha_adj = cal_results[(oa, ot)]['alpha_adjustor']

        # Test against different ground truth orders
        for truth in orders:
                raw_list, cal_list, sel_list = [], [], []
                
                for _ in range(n_responses):
                    X_np = make_X(n, m, distribution=distribution)
                    X = torch.from_numpy(X_np).float()
                    Y_sim, sel = make_Y(X_np, feat_size, truth)
                    Y_t = torch.from_numpy(Y_sim).float()

                    # Compute p-values with specified test order
                    # Set HRT classifier type and fit factor model for each new X
                    local_hrt_kwargs = dict(hrt_kwargs or {})
                    if test == "hrt":
                        if "estimator_type" not in local_hrt_kwargs:
                            local_hrt_kwargs["estimator_type"] = "linear"
                        if "conditional_type" not in local_hrt_kwargs:
                            local_hrt_kwargs["conditional_type"] = "poly2" if ot == 2 else "linear"
                        if local_hrt_kwargs["conditional_type"] == "poly2":
                            local_hrt_kwargs.setdefault("conditional_kwargs", {"lambda": 1e-1})

                        # Fit factor model for this X matrix
                        if "W_hat" not in local_hrt_kwargs:
                            from eccit.utils.sgd_factor_model import factorize
                            _, _, W_hat, V_hat, U_hat, _ = factorize(
                                X_np,
                                n_components=local_hrt_kwargs.get('hrt_n_components', 10),
                                n_steps=local_hrt_kwargs.get('hrt_n_steps', 1000),
                                likelihood=local_hrt_kwargs.get('hrt_likelihood', 'gaussian')
                            )
                            local_hrt_kwargs['W_hat'] = W_hat
                            local_hrt_kwargs['V_hat'] = V_hat
                            local_hrt_kwargs['U_hat'] = U_hat

                    p_raw = compute_conditional_pvals(
                        X,
                        Y_t,
                        test=test,
                        order=ot,
                        use_linear=(ot == 1),
                        gcm_kwargs=(
                            gcm_kwargs
                            if gcm_kwargs is not None
                            else (
                                {"y_estimator": "linear", "x_estimator": "linear"}
                                if ot == 1
                                else {
                                    "y_estimator": "poly2",
                                    "y_estimator_params": {"lambda": 1e-1},
                                    "x_estimator": "linear",
                                }
                            )
                        ) if test == "gcm" else None,
                        kcit_kwargs=kcit_kwargs,
                        rcit_kwargs=rcit_kwargs,
                        hrt_kwargs=local_hrt_kwargs,
                        to_numpy=True,
                    )
                            
                    p_cal = calibrator(p_raw)
                    raw_list.append(p_raw)
                    cal_list.append(p_cal)
                    sel_list.append(sel)

                # Summarize performance
                alphas = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
                fdr_r, se_r, pw_r, se_pw_r, vpw_r, se_vpw_r = summarize(raw_list, sel_list, alphas, n_responses, alpha_adjust=None)
                fdr_c, se_c, pw_c, se_pw_c, vpw_c, se_vpw_c = summarize(raw_list, sel_list, alphas, n_responses, alpha_adjust=alpha_adj)

                vpw_gain = np.array(vpw_c) - np.array(vpw_r)
                vpw_gain_se = np.sqrt(np.array(se_vpw_c)**2 + np.array(se_vpw_r)**2)

                perf_results[(ot, oa, truth)] = dict(
                    alphas=np.array(alphas),
                    fdr_raw=fdr_r, fdr_raw_se=se_r,
                    pow_raw=pw_r, pow_raw_se=se_pw_r,
                    valid_pow_raw=vpw_r, valid_pow_raw_se=se_vpw_r,
                    fdr_cal=fdr_c, fdr_cal_se=se_c,
                    pow_cal=pw_c, pow_cal_se=se_pw_c,
                    valid_pow_cal=vpw_c, valid_pow_cal_se=se_vpw_c,
                    valid_pow_gain=vpw_gain,
                    valid_pow_gain_se=vpw_gain_se,
                )

    return perf_results


def run_second_order_experiment(n=50, m=25, distribution=None,
                               distribution_list=None, num_cal_runs=1, n_responses=100,
                               metric="fdp", output_dir="outputs", test="gcm",
                               gcm_kwargs=None, kcit_kwargs=None):
    """
    Complete second-order experiment with specified metric.

    Main entry point for second-order experiments.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create second-order specific subdirectory
    second_order_dir = Path(output_dir) / "second_order"
    second_order_dir.mkdir(parents=True, exist_ok=True)

    if distribution_list is None:
        if distribution is None:
            distribution_list = ("normal", "correlated", "laplace")
        else:
            distribution_list = (distribution,)

    if metric == "area":
        metric = "type1"
    print(f"Running second-order experiment with metric: {metric}")
    print(f"Parameters: n={n}, m={m}, distributions={distribution_list}")

    from eccit.experiments.plots import plot_second_order_calibration, plot_second_order_performance

    all_cal_results = {}
    all_perf_results = {}

    for dist in distribution_list:
        dist_dir = second_order_dir / dist
        dist_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCalibrating for distribution: {dist}")
        cal_results = run_second_order_calibration(
            n=n,
            m=m,
            distribution=dist,
            num_cal_runs=num_cal_runs,
            metric=metric,
            test=test,
            gcm_kwargs=gcm_kwargs,
            kcit_kwargs=kcit_kwargs,
        )
        plot_second_order_calibration(cal_results, dist_dir, distribution=dist, test=test)

        print(f"Evaluating performance for distribution: {dist}")
        perf_results = evaluate_second_order_performance(
            cal_results,
            n=n,
            m=m,
            distribution=dist,
            n_responses=n_responses,
            metric=metric,
            test=test,
            gcm_kwargs=gcm_kwargs,
            kcit_kwargs=kcit_kwargs,
        )
        plot_second_order_performance(perf_results, dist_dir, metric=metric, distribution=dist, test=test)

        all_cal_results[dist] = cal_results
        all_perf_results[dist] = perf_results

    return all_cal_results, all_perf_results
