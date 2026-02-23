import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eccit.calibration_runner import calibrate_run
from eccit.cits import compute_conditional_pvals
from eccit.calibration import miscal_fdp, miscal_type1_multi
from eccit.utils.helpers import make_X
def _set_random_seed(seed: int) -> None:
    """Set numpy, random, and torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate_calibration(pvals, null_mask, calibrator, alpha):
    """Compute type-I error and FDP for (possibly) calibrated p-values."""
    pvals = np.asarray(pvals, dtype=float)
    null_mask = np.asarray(null_mask, dtype=bool)

    if calibrator is not None:
        pvals = np.asarray(calibrator(pvals), dtype=float)

    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=0.0)
    pvals = np.clip(pvals, 1e-12, 1.0)

    if null_mask.size == 0:
        return {
            'type1': 0.0,
            'fdp': 0.0,
        }

    p_tensor = torch.tensor(pvals, dtype=torch.float32)
    null_tensor = torch.tensor(null_mask, dtype=torch.bool)

    if null_tensor.any():
        type1 = miscal_type1_multi(p_tensor, null_tensor, alpha=alpha).item()
    else:
        type1 = 0.0
    fdp = miscal_fdp(p_tensor, null_tensor, alpha=alpha).item()

    return {
        'type1': float(type1),
        'fdp': float(fdp),
    }


def _run_calibration_once(X, seed, freeze_adversary=False, **calibrate_kwargs):
    """Run calibration once and extract the statistics needed for evaluation."""
    _set_random_seed(seed)
    cal_out = calibrate_run(
        X,
        freeze_adversary=freeze_adversary,
        **calibrate_kwargs,
    )

    calibrator, _, _, _, _, _, _, diagnostics = cal_out

    return {
        'calibrator': calibrator,
        'eval_pvals': diagnostics['eval_pvals'],
        'eval_null_mask': diagnostics['eval_null_mask'],
        'final_probs': diagnostics['final_probs'],
        'final_mask': diagnostics['final_mask'],
        'mask_prob_hist': diagnostics.get('mask_prob_hist'),
        'alpha_train': diagnostics.get('alpha_train', calibrate_kwargs.get('alpha_train', 0.2)),
    }


# Mask enumeration and evaluation
def all_binary_masks(p):
    """Generate all possible binary masks for p features."""
    for i in range(1 << p):
        yield np.array([(i >> b) & 1 for b in range(p)], dtype=float)


def mask_to_bitstring(mask_1d):
    """Convert binary mask to string representation."""
    return ''.join('1' if int(x) == 1 else '0' for x in mask_1d)

def compute_null_type1_for_mask(X_full, ygen, mask_bin, use_linear=True, order_test=1,
                                mask_draws=50, subsample_frac=0.8,
                                alpha_eval=0.2, metric="fdp", test="gcm",
                                gcm_kwargs=None, kcit_kwargs=None):
    """
    Evaluate miscalibration metric for a specific binary mask.
    
    Returns mean and standard error across multiple bootstrap samples.
    """
    X_np = X_full.numpy() if torch.is_tensor(X_full) else X_full
    n, p = X_np.shape
    n_sub = int(round(n * subsample_frac))

    mask_t = torch.from_numpy(mask_bin).float()
    null_mask = (mask_bin < 0.5).astype(bool)

    if metric == "area":
        metric = "type1"

    metrics = []
    for _ in range(mask_draws):
        idx = np.random.choice(n, size=n_sub)
        X_sub = torch.from_numpy(X_np[idx]).float()

        with torch.no_grad():
            Y_sub, _ = ygen.forward_with_mask(X_sub, mask_t)

        # Compute p-values
        pvals = compute_conditional_pvals(
            X_sub,
            Y_sub,
            test=test,
            order=order_test,
            use_linear=(use_linear if test == "gcm" else True),
            gcm_kwargs=gcm_kwargs,
            kcit_kwargs=kcit_kwargs,
            to_numpy=True,
        )

        # Evaluate metric
        if null_mask.sum() == 0:
            metrics.append(0.0)
        else:
            if metric == "fdp":
                fdp_val = miscal_fdp(
                    torch.tensor(pvals, dtype=torch.float32),
                    torch.tensor(null_mask, dtype=torch.bool),
                    alpha=alpha_eval
                ).item()
                metrics.append(fdp_val)
            elif metric == "type1":
                type1_val = miscal_type1_multi(
                    torch.tensor(pvals, dtype=torch.float32),
                    torch.tensor(null_mask, dtype=torch.bool),
                    alpha=alpha_eval,
                ).item()
                metrics.append(type1_val)

    metrics = np.asarray(metrics, dtype=float)
    mean = float(metrics.mean()) if metrics.size else 0.0
    sem = float(metrics.std(ddof=1) / np.sqrt(len(metrics))) if len(metrics) > 1 else 0.0
    return mean, sem


def evaluate_all_masks(X_full, ygen, use_linear=True, order_test=1, 
                       mask_draws=50, final_mask=None, alpha_eval=0.2, metric="fdp",
                       test="gcm", gcm_kwargs=None, kcit_kwargs=None):
    """
    Enumerate all possible masks and rank by performance.
    
    Returns DataFrame with mask rankings and performance metrics.
    """
    X_np = X_full.numpy() if torch.is_tensor(X_full) else X_full
    n, p = X_np.shape

    rows = []
    for mask_bin in all_binary_masks(p):
        type1_mean, type1_sem = compute_null_type1_for_mask(
            X_np,
            ygen,
            mask_bin,
            use_linear=use_linear,
            order_test=order_test,
            mask_draws=mask_draws,
            alpha_eval=alpha_eval,
            metric=metric,
            test=test,
            gcm_kwargs=gcm_kwargs,
            kcit_kwargs=kcit_kwargs,
        )
        rows.append(dict(
            mask_bits=mask_to_bitstring(mask_bin),
            k_active=int(mask_bin.sum()),
            type1_mean=type1_mean,
            type1_sem=type1_sem
        ))

    df = pd.DataFrame(rows)
    df.sort_values(['type1_mean', 'k_active'], ascending=[False, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['rank'] = np.arange(1, len(df) + 1)

    if final_mask is not None:
        final_bits = mask_to_bitstring(final_mask.astype(float))
        df['is_final'] = (df['mask_bits'] == final_bits).astype(int)
    else:
        df['is_final'] = 0

    return df


def sample_masks_from_probs(probs, S=5000):
    """Sample binary masks from learned probabilities."""
    probs = np.asarray(probs, dtype=float)
    return (np.random.rand(S, probs.shape[0]) < probs).astype(float)


def evaluate_mask_sampling_from_table(df_masks, probs, S=5000, topK=20, eps_frac=0.01):
    """
    Evaluate how well the learned mask probabilities sample good masks.
    
    Returns summary statistics of mask sampling performance.
    """
    if 'type1_mean' not in df_masks.columns and 'area_mean' in df_masks.columns:
        df_masks = df_masks.rename(columns={'area_mean': 'type1_mean'})
    # Create lookup tables
    rank_lookup = dict(zip(df_masks['mask_bits'].values, df_masks['rank'].values))
    type1_lookup = dict(zip(df_masks['mask_bits'].values, df_masks['type1_mean'].values))
    best_metric = float(df_masks['type1_mean'].iloc[0])

    # Sample masks and convert to strings
    masks = sample_masks_from_probs(probs, S=S)
    bits = [mask_to_bitstring(m) for m in masks]

    # Aggregate statistics
    uniq, counts = np.unique(bits, return_counts=True)
    ranks = np.array([rank_lookup[b] for b in uniq], dtype=float)
    type1_vals = np.array([type1_lookup[b] for b in uniq], dtype=float)

    total = counts.sum()
    prob_est = counts / total
    expected_type1 = float((prob_est * type1_vals).sum())

    # Compute median rank and top-K hit rate
    order = np.argsort(ranks)
    cdf = np.cumsum(prob_est[order])
    median_rank = float(ranks[order][np.searchsorted(cdf, 0.5)])
    hit_topK_rate = float(prob_est[ranks <= topK].sum())

    samples_df = pd.DataFrame(dict(
        mask_bits=uniq,
        count=counts,
        prob_est=prob_est,
        rank=ranks,
        type1_mean=type1_vals
    )).sort_values('rank').reset_index(drop=True)

    summary = dict(
        S=int(S),
        topK=int(topK),
        eps_frac=float(eps_frac),
        best_metric=best_metric,
        expected_metric=expected_type1,
        median_rank=median_rank,
        hit_topK_rate=hit_topK_rate,
    )
    
    return summary, samples_df


def run_mask_experiment(n=20, p=10, distribution="normal", num_epochs=50,
                       order_adv=2, order_test=2, use_linear=True, mask_draws=100,
                       top_k=50, S_samples=5000, eps_frac=0.01,
                       metric="type1", output_dir="outputs",
                       mask_lr=1e-2, weight_lr=1e-3,
                       test="gcm", gcm_kwargs=None, kcit_kwargs=None,
                       rcit_kwargs=None, hrt_kwargs=None,
                       n_runs=5, toy_seed=0):
    """
    Compare calibration quality when the adversarial mean model is trained vs frozen.

    The new experiment keeps the mask training intact but optionally freezes the
    adversarial MLP weights.  We report calibration metrics on a toy setup and
    aggregate statistics over multiple random runs for three modes:

      * raw p-values (no calibration)
      * calibration learned with frozen adversary
      * calibration learned with fully trained adversary
    """

    mask_dir = Path(output_dir) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    if metric == "area":
        metric = "type1"
    print(f"Running mask experiment (freeze comparison) with metric: {metric}")
    print(f"Parameters: n={n}, p={p}, epochs={num_epochs}, runs={n_runs}")

    calibrate_kwargs = dict(
        metric=metric,
        use_linear=use_linear,
        num_epochs=num_epochs,
        order_adv=order_adv,
        order_test=order_test,
        mask_lr=mask_lr,
        weight_lr=weight_lr,
        test=test,
        gcm_kwargs=gcm_kwargs,
        kcit_kwargs=kcit_kwargs,
        rcit_kwargs=rcit_kwargs,
        hrt_kwargs=hrt_kwargs,
    )

    # --- Toy example -----------------------------------------------------
    _set_random_seed(toy_seed)
    toy_X_np = make_X(n, p, distribution=distribution)
    toy_X = torch.from_numpy(toy_X_np).float()

    toy_trained = _run_calibration_once(toy_X, toy_seed, False, **calibrate_kwargs)
    toy_frozen = _run_calibration_once(toy_X, toy_seed, True, **calibrate_kwargs)

    toy_pvals = toy_trained['eval_pvals']
    toy_null_mask = toy_trained['eval_null_mask']
    toy_alpha = toy_trained['alpha_train']

    toy_raw_metrics = _evaluate_calibration(toy_pvals, toy_null_mask, None, toy_alpha)
    toy_trained_metrics = _evaluate_calibration(toy_pvals, toy_null_mask, toy_trained['calibrator'], toy_alpha)
    toy_frozen_metrics = _evaluate_calibration(toy_pvals, toy_null_mask, toy_frozen['calibrator'], toy_alpha)

    toy_mask_df = pd.DataFrame(
        {
            'trained_adversary': toy_trained['final_probs'],
            'frozen_adversary': toy_frozen['final_probs'],
        }
    )
    toy_mask_path = mask_dir / "toy_mask_probabilities.csv"
    toy_mask_df.to_csv(toy_mask_path, index_label="feature")

    print(f"\n--- Toy Example (seed {toy_seed}) ---")
    print(
        f"Raw: type-I={toy_raw_metrics['type1']:.4f}, FDP={toy_raw_metrics['fdp']:.4f}"
    )
    print(
        f"Calibrated (frozen adversary): type-I={toy_frozen_metrics['type1']:.4f}, "
        f"FDP={toy_frozen_metrics['fdp']:.4f}"
    )
    print(
        f"Calibrated (trained adversary): type-I={toy_trained_metrics['type1']:.4f}, "
        f"FDP={toy_trained_metrics['fdp']:.4f}"
    )

    # Highlight average mask mass to contrast the two settings.
    toy_trained_mass = float(np.sum(toy_trained['final_probs']))
    toy_frozen_mass = float(np.sum(toy_frozen['final_probs']))
    print(
        f"Mask probability mass (trained vs frozen): "
        f"{toy_trained_mass:.2f} vs {toy_frozen_mass:.2f}"
    )

    # --- Multiple runs ---------------------------------------------------
    records = []
    mask_records = []

    for run_idx in range(n_runs):
        run_seed = toy_seed + run_idx + 1
        _set_random_seed(run_seed)
        X_np = make_X(n, p, distribution=distribution)
        X_tensor = torch.from_numpy(X_np).float()

        trained_run = _run_calibration_once(X_tensor, run_seed, False, **calibrate_kwargs)
        frozen_run = _run_calibration_once(X_tensor, run_seed, True, **calibrate_kwargs)

        eval_pvals = trained_run['eval_pvals']
        null_mask = trained_run['eval_null_mask']
        alpha = trained_run['alpha_train']

        metrics_raw = _evaluate_calibration(eval_pvals, null_mask, None, alpha)
        metrics_frozen = _evaluate_calibration(eval_pvals, null_mask, frozen_run['calibrator'], alpha)
        metrics_trained = _evaluate_calibration(eval_pvals, null_mask, trained_run['calibrator'], alpha)

        records.extend(
            [
                {
                    'run': run_idx,
                    'mode': 'raw',
                    'miscal_type1': metrics_raw['type1'],
                    'miscal_fdp': metrics_raw['fdp'],
                },
                {
                    'run': run_idx,
                    'mode': 'calibrated_frozen',
                    'miscal_type1': metrics_frozen['type1'],
                    'miscal_fdp': metrics_frozen['fdp'],
                },
                {
                    'run': run_idx,
                    'mode': 'calibrated_trained',
                    'miscal_type1': metrics_trained['type1'],
                    'miscal_fdp': metrics_trained['fdp'],
                },
            ]
        )

        mask_records.extend(
            [
                {
                    'run': run_idx,
                    'mode': 'frozen',
                    'prob_sum': float(np.sum(frozen_run['final_probs'])),
                    'prob_mean': float(np.mean(frozen_run['final_probs'])),
                },
                {
                    'run': run_idx,
                    'mode': 'trained',
                    'prob_sum': float(np.sum(trained_run['final_probs'])),
                    'prob_mean': float(np.mean(trained_run['final_probs'])),
                },
            ]
        )

    results_df = pd.DataFrame(records)
    mask_df = pd.DataFrame(mask_records)

    summary_df = results_df.groupby('mode').agg(
        miscal_type1_mean=('miscal_type1', 'mean'),
        miscal_type1_std=('miscal_type1', 'std'),
        miscal_fdp_mean=('miscal_fdp', 'mean'),
        miscal_fdp_std=('miscal_fdp', 'std'),
    ).fillna(0.0)

    mask_summary_df = mask_df.groupby('mode').agg(
        prob_sum_mean=('prob_sum', 'mean'),
        prob_sum_std=('prob_sum', 'std'),
        prob_mean_mean=('prob_mean', 'mean'),
        prob_mean_std=('prob_mean', 'std'),
    ).fillna(0.0)

    ci_factor = 1.0 / 10.0
    for col in summary_df.columns:
        if col.endswith('_std'):
            summary_df[col] *= ci_factor
    for col in mask_summary_df.columns:
        if col.endswith('_std'):
            mask_summary_df[col] *= ci_factor

    # Ensure consistent ordering for reporting
    summary_order = [mode for mode in ['raw', 'calibrated_frozen', 'calibrated_trained'] if mode in summary_df.index]
    if summary_order:
        summary_df = summary_df.loc[summary_order]

    mask_order = [mode for mode in ['frozen', 'trained'] if mode in mask_summary_df.index]
    if mask_order:
        mask_summary_df = mask_summary_df.loc[mask_order]

    results_path = mask_dir / "frozen_vs_trained_runs.csv"
    summary_path = mask_dir / "frozen_vs_trained_summary.csv"
    mask_summary_path = mask_dir / "mask_probability_summary.csv"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path)
    mask_summary_df.to_csv(mask_summary_path)

    print(f"\n--- Aggregate across {n_runs} runs ---")
    print(
        summary_df.to_string(
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print("\nMask probability mass (mean ± std):")
    print(
        mask_summary_df.to_string(
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return {
        'toy': {
            'seed': toy_seed,
            'raw': toy_raw_metrics,
            'calibrated_frozen': toy_frozen_metrics,
            'calibrated_trained': toy_trained_metrics,
            'mask_probabilities_csv': str(toy_mask_path),
        },
        'runs': results_df,
        'summary': summary_df,
        'mask_summary': mask_summary_df,
        'runs_csv': str(results_path),
        'summary_csv': str(summary_path),
        'mask_summary_csv': str(mask_summary_path),
    }
