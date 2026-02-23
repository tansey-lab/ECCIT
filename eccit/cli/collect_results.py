#!/usr/bin/env python3
"""
Collect and aggregate results from cluster experiments.

Usage:
    eccit-collect-results sweep fdp --test gcm
    eccit-collect-results second_order type1 --test hrt
"""

import argparse
from collections import defaultdict
from pathlib import Path
import dill as pickle
import numpy as np
import pandas as pd

from eccit.utils.job_utils import load_and_aggregate_results
from eccit.experiments.benchmarks import BenchmarkConfig, export_benchmark_tables, summarize_benchmark_runs
from types import SimpleNamespace


def aggregate_sweep_results(aggregated_results, metric):
    calibration = {}
    performance = {}
    runtime_entries = []

    for result_list in aggregated_results.values():
        for result, metadata in result_list:
            if not isinstance(result, dict) or 'calibration' not in result or 'performance' not in result:
                raise ValueError("Sweep results must contain precomputed calibration/performance payloads")
            calibration.update(result['calibration'])
            performance.update(result['performance'])
            runtime = metadata.get('runtime_seconds') if metadata else None
            if runtime is not None:
                runtime_entries.append((runtime, metadata))

    runtime_summary = {}
    if runtime_entries:
        max_runtime, meta = max(runtime_entries, key=lambda x: x[0])
        runtime_summary = {
            'max_runtime_seconds': max_runtime,
            'job_metadata': meta
        }

    return {
        'calibration': calibration,
        'performance': performance,
        'runtime_summary': runtime_summary
    }


def aggregate_second_order_results(aggregated_results, metric):
    calibration = defaultdict(dict)
    performance = defaultdict(dict)
    runtime_entries = []

    for result_list in aggregated_results.values():
        for result, metadata in result_list:
            if not isinstance(result, dict) or 'calibration' not in result or 'performance' not in result:
                raise ValueError("Second-order results must contain precomputed calibration/performance payloads")

            for (dist, oa, ot), cal_payload in result['calibration'].items():
                calibration[dist][(oa, ot)] = cal_payload

            for (dist, ot, oa, truth), perf_payload in result['performance'].items():
                performance[dist][(ot, oa, truth)] = perf_payload

            runtime = metadata.get('runtime_seconds') if metadata else None
            if runtime is not None:
                runtime_entries.append((runtime, metadata))

    runtime_summary = {}
    if runtime_entries:
        max_runtime, meta = max(runtime_entries, key=lambda x: x[0])
        runtime_summary = {
            'max_runtime_seconds': max_runtime,
            'job_metadata': meta
        }

    return {
        'calibration': dict(calibration),
        'performance': dict(performance),
        'runtime_summary': runtime_summary
    }


def aggregate_mask_results(aggregated_results):
    """Process aggregated mask results."""
    final_results = {}
    runtime_entries = []

    for key, result_list in aggregated_results.items():
        n, p, distribution = key

        # For masks, we just collect all individual results
        # since each is a complete experiment
        final_results[key] = {
            'individual_results': result_list,
            'num_runs': len(result_list)
        }

        for _, metadata in result_list:
            runtime = metadata.get('runtime_seconds') if metadata else None
            if runtime is not None:
                runtime_entries.append((runtime, metadata))

    runtime_summary = {}
    if runtime_entries:
        max_runtime, meta = max(runtime_entries, key=lambda x: x[0])
        runtime_summary = {
            'max_runtime_seconds': max_runtime,
            'job_metadata': meta
        }

    return {
        'results': final_results,
        'runtime_summary': runtime_summary
    }


def export_semi_tables(performance, metric):
    """Export LaTeX tables for semi experiment results (at alpha=0.2)."""
    alpha_target = 0.2

    for (response, test), payload in sorted(performance.items()):
        alphas = np.asarray(payload.get('alphas', []))
        if alphas.size == 0:
            continue

        # Find index for alpha=0.2
        alpha_idx = int(np.argmin(np.abs(alphas - alpha_target)))

        rows = []

        def _extract_at_alpha(field_base):
            """Extract mean and std at alpha=0.2."""
            mean_vals = np.asarray(payload.get(f'{field_base}_mean', []))
            std_vals = np.asarray(payload.get(f'{field_base}_std', []))
            if mean_vals.size > alpha_idx:
                mean = float(mean_vals[alpha_idx])
                std = float(std_vals[alpha_idx]) if std_vals.size > alpha_idx else 0.0
                se = 2 * std  # SE
                return mean, se
            return None, None

        # Add uncalibrated test
        power_mean, power_se = _extract_at_alpha('pow_raw')
        valid_mean, valid_se = _extract_at_alpha('valid_pow_raw')
        if power_mean is not None:
            rows.append({
                'Method': f'{test.upper()}',
                'Power': power_mean,
                'Power SE': power_se,
                'Valid Power': valid_mean,
                'Valid Power SE': valid_se,
            })

        # Add calibrated test
        power_mean, power_se = _extract_at_alpha('pow_cal')
        valid_mean, valid_se = _extract_at_alpha('valid_pow_cal')
        if power_mean is not None:
            rows.append({
                'Method': f'Calibrated {test.upper()}',
                'Power': power_mean,
                'Power SE': power_se,
                'Valid Power': valid_mean,
                'Valid Power SE': valid_se,
            })

        # Add CONTRA methods
        if "contra" in payload:
            for method in ['crt', 'hrt', 'fastcrt']:
                if method in payload["contra"]:
                    contra_data = payload["contra"][method]
                    contra_alphas = np.asarray(contra_data.get('alphas', []))
                    if contra_alphas.size == 0:
                        continue
                    contra_alpha_idx = int(np.argmin(np.abs(contra_alphas - alpha_target)))

                    pow_mean_arr = np.asarray(contra_data.get('pow_mean', []))
                    pow_std_arr = np.asarray(contra_data.get('pow_std', []))
                    vpow_mean_arr = np.asarray(contra_data.get('valid_pow_mean', []))
                    vpow_std_arr = np.asarray(contra_data.get('valid_pow_std', []))

                    if pow_mean_arr.size > contra_alpha_idx:
                        rows.append({
                            'Method': f'CONTRA-{method.upper()}',
                            'Power': float(pow_mean_arr[contra_alpha_idx]),
                            'Power SE': 2 * float(pow_std_arr[contra_alpha_idx]) if pow_std_arr.size > contra_alpha_idx else 0.0,
                            'Valid Power': float(vpow_mean_arr[contra_alpha_idx]) if vpow_mean_arr.size > contra_alpha_idx else 0.0,
                            'Valid Power SE': 2 * float(vpow_std_arr[contra_alpha_idx]) if vpow_std_arr.size > contra_alpha_idx else 0.0,
                        })

        if not rows:
            continue

        df = pd.DataFrame(rows)
        tables_dir = Path("outputs") / "semi" / f"{response}_{test}_{metric}"
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = tables_dir / "semi_summary.csv"
        df.to_csv(csv_path, index=False)

        # Create LaTeX table
        best_valid = df['Valid Power'].max()
        latex_df = df.copy()
        latex_df['Power'] = [
            f"{mean:.2f}±{se:.3f}"
            for mean, se in zip(df['Power'].values, df['Power SE'].values)
        ]
        valid_col = []
        for mean, se in zip(df['Valid Power'].values, df['Valid Power SE'].values):
            entry = f"{mean:.2f}±{se:.3f}"
            if np.isclose(mean, best_valid):
                entry = f"\\textbf{{{entry}}}"
            valid_col.append(entry)
        latex_df['Valid Power'] = valid_col
        latex_df = latex_df[['Method', 'Power', 'Valid Power']]

        response_label = response.capitalize()
        metric_label = metric.upper()
        num_jobs = payload.get('num_jobs', '?')
        caption = (
            f"\\textbf{{GDSC Semi-synthetic ({response_label}, {test.upper()})}} "
            f"Calibrated with {metric_label} metric. "
            f"$n=125$, $m=25$, $\\alpha=0.2$. "
            f"Aggregated over {num_jobs} independent GDSC subsamples. ± SE."
        )
        label = f"tab:semi_{response}_{test}_{metric}"
        latex_path = tables_dir / "semi_summary.tex"
        latex_df.to_latex(
            latex_path,
            index=False,
            escape=False,
            caption=caption,
            label=label,
            column_format='lcc'
        )

        print(f"Semi tables saved to {tables_dir}")


def aggregate_semi_results(aggregated_results):
    """Aggregate semi-supervised experiment outputs across jobs."""
    performance = defaultdict(list)
    runtime_entries = []

    for result_list in aggregated_results.values():
        for result, metadata in result_list:
            if isinstance(result, dict):
                perf_payload = result.get('performance', {})
                for response, payload in perf_payload.items():
                    # Extract test type from payload
                    test = payload.get('test', 'gcm')
                    key = (response, test)
                    performance[key].append(payload)
            runtime = metadata.get('runtime_seconds') if metadata else None
            if runtime is not None:
                runtime_entries.append((runtime, metadata))

    runtime_summary = {}
    if runtime_entries:
        max_runtime, meta = max(runtime_entries, key=lambda x: x[0])
        runtime_summary = {
            'max_runtime_seconds': max_runtime,
            'job_metadata': meta,
        }

    combined = {}
    metric_fields = [
        'fdr_raw_mean',
        'fdr_cal_mean',
        'pow_raw_mean',
        'pow_cal_mean',
        'valid_pow_raw_mean',
        'valid_pow_cal_mean',
        'valid_pow_gain_mean',
    ]

    for (response, test), payloads in performance.items():
        if not payloads:
            continue
        alphas = np.asarray(payloads[0]['alphas'])
        summary = {
            'alphas': alphas,
            'num_jobs': len(payloads),
            'test': test,
            'response': response,
        }
        for field in metric_fields:
            stack = np.stack([np.asarray(p[field]) for p in payloads], axis=0)
            summary[field] = stack.mean(axis=0)
            std_name = field.replace('_mean', '_std')
            summary[std_name] = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(stack[0])

        # Aggregate CONTRA results if present
        if "contra" in payloads[0] and payloads[0]["contra"]:
            contra_methods = list(payloads[0]["contra"].keys())
            contra_summary = {}
            for method in contra_methods:
                contra_summary[method] = {}
                contra_summary[method]['alphas'] = alphas
                for metric in ["fdr", "pow", "valid_pow"]:
                    stack = np.stack([np.asarray(p["contra"][method][f"{metric}_mean"]) for p in payloads], axis=0)
                    contra_summary[method][f"{metric}_mean"] = stack.mean(axis=0)
                    contra_summary[method][f"{metric}_std"] = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(stack[0])
            summary["contra"] = contra_summary

        combined[(response, test)] = summary

    return {
        'performance': combined,
        'runtime_summary': runtime_summary,
    }


def summarize_mask_freeze_results(final_results):
    """Print aggregate summary for mask-freeze experiments."""
    results = final_results.get('results', {})
    if not results:
        print("No mask-freeze results to summarize.")
        return

    for key, payload in results.items():
        combined_runs = []
        combined_mask_summaries = []

        for result, metadata in payload.get('individual_results', []):
            runs_df = result.get('runs') if isinstance(result, dict) else None
            mask_summary_df = result.get('mask_summary') if isinstance(result, dict) else None

            if isinstance(runs_df, pd.DataFrame):
                runs_copy = runs_df.copy()
                if metadata:
                    runs_copy['job_id'] = metadata.get('job_id')
                combined_runs.append(runs_copy)
            if isinstance(mask_summary_df, pd.DataFrame):
                idx_name = mask_summary_df.index.name or 'mode'
                mask_summary_reset = mask_summary_df.reset_index()
                mask_summary_reset.rename(columns={idx_name: 'mode'}, inplace=True)
                combined_mask_summaries.append(mask_summary_reset)

        if not combined_runs:
            continue

        all_runs_df = pd.concat(combined_runs, ignore_index=True)
        if 'miscal_type1' not in all_runs_df.columns and 'miscal_area' in all_runs_df.columns:
            all_runs_df = all_runs_df.assign(miscal_type1=all_runs_df['miscal_area'])
        if 'gap_type1' not in all_runs_df.columns and 'gap_area' in all_runs_df.columns:
            all_runs_df = all_runs_df.assign(gap_type1=all_runs_df['gap_area'])
        if 'target_type1' not in all_runs_df.columns and 'target_area' in all_runs_df.columns:
            all_runs_df = all_runs_df.assign(target_type1=all_runs_df['target_area'])
        summary_df = all_runs_df.groupby('mode').agg(
            miscal_type1_mean=('miscal_type1', 'mean'),
            miscal_type1_std=('miscal_type1', 'std'),
            gap_type1_mean=('gap_type1', 'mean'),
            gap_type1_std=('gap_type1', 'std'),
            target_type1_mean=('target_type1', 'mean'),
            miscal_fdp_mean=('miscal_fdp', 'mean'),
            miscal_fdp_std=('miscal_fdp', 'std'),
            gap_fdp_mean=('gap_fdp', 'mean'),
            gap_fdp_std=('gap_fdp', 'std'),
            target_fdp_mean=('target_fdp', 'mean'),
        ).fillna(0.0)

        ci_factor = 1.0 / 10.0
        std_cols_summary = [col for col in summary_df.columns if col.endswith('_std')]
        if std_cols_summary:
            summary_df[std_cols_summary] = summary_df[std_cols_summary] * ci_factor

        unique_trials = all_runs_df[['job_id', 'run']].drop_duplicates().shape[0] if {'job_id', 'run'}.issubset(all_runs_df.columns) else len(all_runs_df)

        print("\n--- Aggregate across {} jobs ({} unique runs, {} rows total) ---".format(
            len(combined_runs), unique_trials, len(all_runs_df)))

        type1_cols = [col for col in ['miscal_type1_mean', 'miscal_type1_std', 'gap_type1_mean', 'gap_type1_std', 'target_type1_mean'] if col in summary_df.columns]
        fdp_cols = [col for col in ['miscal_fdp_mean', 'miscal_fdp_std', 'gap_fdp_mean', 'gap_fdp_std', 'target_fdp_mean'] if col in summary_df.columns]

        if type1_cols:
            print("Type-I error:")
            print(summary_df[type1_cols].to_string(float_format=lambda x: f"{x:.4f}"))
        if fdp_cols:
            print("\nFDP calibration (target ≈ alpha):")
            print(summary_df[fdp_cols].to_string(float_format=lambda x: f"{x:.4f}"))

        if combined_mask_summaries:
            all_mask_df = pd.concat(combined_mask_summaries, ignore_index=True)
            mask_summary_df = all_mask_df.groupby('mode').agg(
                prob_sum_mean=('prob_sum_mean', 'mean') if 'prob_sum_mean' in all_mask_df.columns else ('prob_sum', 'mean'),
                prob_sum_std=('prob_sum_std', 'mean') if 'prob_sum_std' in all_mask_df.columns else ('prob_sum', 'std'),
                prob_mean_mean=('prob_mean_mean', 'mean') if 'prob_mean_mean' in all_mask_df.columns else ('prob_mean', 'mean'),
                prob_mean_std=('prob_mean_std', 'mean') if 'prob_mean_std' in all_mask_df.columns else ('prob_mean', 'std'),
            ).fillna(0.0)

            std_cols_mask = [col for col in mask_summary_df.columns if col.endswith('_std')]
            if std_cols_mask:
                mask_summary_df[std_cols_mask] = mask_summary_df[std_cols_mask] * ci_factor

            print("\nMask probability mass (mean ± std):")
            print(mask_summary_df.to_string(float_format=lambda x: f"{x:.4f}"))


def _distribution_name(dist):
    mapping_csv = {
        'normal': 'Normal',
        'correlated': 'Correlated',
        'laplace': 'Laplace',
    }
    mapping_latex = {
        'normal': 'Normal',
        'correlated': 'Correlated',
        'laplace': 'Laplace',
    }
    csv_name = mapping_csv.get(dist, dist.replace('_', ' ').title())
    latex_name = mapping_latex.get(dist, csv_name)
    return csv_name, latex_name


def _metric_label(metric):
    return metric.upper()


def _test_label(test):
    return test.upper()


def _format_gain(mean, std):
    if mean is None:
        return '--'
    mean_val = float(mean)
    if np.isnan(mean_val):
        return '--'
    std_val = 0.0
    if std is not None:
        std_val = float(std)
        if np.isnan(std_val):
            std_val = 0.0
    return f"{mean_val:.2f}±{std_val:.3f}"


def build_feature_tables(performance, metric, test, tables_dir):
    distributions = sorted({dist for dist, _, _ in performance.keys()})
    metric_label = _metric_label(metric)
    test_label = _test_label(test)
    saved = []

    for dist in distributions:
        ns = sorted({n for (d, n, _m) in performance.keys() if d == dist})
        ms = sorted({m for (d, _n, m) in performance.keys() if d == dist})
        if not ns or not ms:
            continue

        columns = []
        for m in ms:
            columns.append((f"m={m}", 'gain_mean'))
            columns.append((f"m={m}", 'gain_std'))
        multi_cols = pd.MultiIndex.from_tuples(columns, names=['Features', 'Statistic'])

        data = []
        for n in ns:
            row = []
            for m in ms:
                entry = performance.get((dist, n, m))
                if entry:
                    row.append(entry.get('valid_power_gain_mean', np.nan))
                    row.append(entry.get('valid_power_gain_std', np.nan))
                else:
                    row.extend([np.nan, np.nan])
            data.append(row)

        df_numeric = pd.DataFrame(data, index=ns, columns=multi_cols)
        df_numeric.index.name = 'n'
        ci_factor = 1.0 / 10.0
        for m in ms:
            df_numeric[(f"m={m}", 'gain_std')] = df_numeric[(f"m={m}", 'gain_std')] * ci_factor
        csv_name, latex_name = _distribution_name(dist)

        csv_path = tables_dir / f"valid_power_gain_{dist}_{metric}_{test}.csv"
        df_numeric.to_csv(csv_path)

        formatted = pd.DataFrame(index=ns)
        for m in ms:
            mean_col = df_numeric[(f"m={m}", 'gain_mean')]
            std_col = df_numeric[(f"m={m}", 'gain_std')]
            formatted[str(m)] = [
                _format_gain(mean, std)
                for mean, std in zip(mean_col.values, std_col.values)
            ]

        formatted.index.name = 'n'
        formatted.columns = [str(m) for m in ms]

        caption = (
            f"\\textbf{{Valid Power Gain for {test_label}}}, {latex_name} $X$, Linear $Y$, "
            f"calibrated with {metric_label}. ± SE."
        )
        label = f"tab:{test}_{metric}_{dist}_gain"
        latex_path = tables_dir / f"valid_power_gain_{dist}_{metric}_{test}.tex"
        formatted.to_latex(
            latex_path,
            caption=caption,
            label=label,
            escape=False,
            index=True,
            column_format='l' + 'c' * len(ms)
        )

        saved.append((csv_path, latex_path))

    return saved


def _select_reference_pair(performance):
    distributions = {dist for dist, _, _ in performance.keys()}
    combo_sets = {dist: {(n, m) for (_dist, n, m) in performance.keys() if _dist == dist}
                  for dist in distributions}
    common = set.intersection(*(s for s in combo_sets.values())) if combo_sets else set()
    target_n, target_m = 125, 25

    def score(pair):
        n, m = pair
        return abs(n - target_n) + 5 * abs(m - target_m)

    if common:
        return min(common, key=score)

    # Choose pair that appears most frequently across distributions
    counts = {}
    for dist, combos in combo_sets.items():
        for combo in combos:
            counts[combo] = counts.get(combo, 0) + 1
    if not counts:
        return None
    best = max(counts.items(), key=lambda x: (x[1], -score(x[0])))[0]
    return best


def build_distribution_table(performance, metric, test, tables_dir):
    ref_pair = _select_reference_pair(performance)
    if ref_pair is None:
        return None

    n_ref, m_ref = ref_pair
    rows = []
    csv_rows = []
    csv_index = []
    distributions = sorted({dist for dist, _, _ in performance.keys()})

    for dist in distributions:
        entry = performance.get((dist, n_ref, m_ref))
        if not entry:
            continue
        csv_name, latex_name = _distribution_name(dist)
        raw_mean = entry.get('valid_power_raw_mean', np.nan)
        raw_std = entry.get('valid_power_raw_std', np.nan)
        cal_mean = entry.get('valid_power_cal_mean', np.nan)
        cal_std = entry.get('valid_power_cal_std', np.nan)
        gain_mean = entry.get('valid_power_gain_mean', np.nan)
        gain_std = entry.get('valid_power_gain_std', np.nan)

        ci_factor = 1.0 / 10.0
        if not np.isnan(raw_std):
            raw_std *= ci_factor
        if not np.isnan(cal_std):
            cal_std *= ci_factor
        if not np.isnan(gain_std):
            gain_std *= ci_factor

        csv_rows.append({
            ('Uncalibrated', 'mean'): raw_mean,
            ('Uncalibrated', 'std'): raw_std,
            ('Calibrated', 'mean'): cal_mean,
            ('Calibrated', 'std'): cal_std,
            ('Gain', 'mean'): gain_mean,
            ('Gain', 'std'): gain_std,
        })
        csv_index.append(csv_name)

        rows.append({
            'Distribution': latex_name,
            'Uncalibrated': _format_gain(raw_mean, raw_std),
            'Calibrated': _format_gain(cal_mean, cal_std),
            'Gain': _format_gain(gain_mean, gain_std),
        })

    if not rows:
        return None

    csv_df = pd.DataFrame(csv_rows, index=csv_index)
    csv_df.index.name = 'Distribution'
    # Ensure consistent column order
    csv_df = csv_df[[('Uncalibrated', 'mean'), ('Uncalibrated', 'std'),
                     ('Calibrated', 'mean'), ('Calibrated', 'std'),
                     ('Gain', 'mean'), ('Gain', 'std')]]
    csv_df.columns = pd.MultiIndex.from_tuples(csv_df.columns, names=['Metric', 'Statistic'])

    csv_path = tables_dir / f"valid_power_distributions_{metric}_{test}.csv"
    csv_df.to_csv(csv_path)

    latex_df = pd.DataFrame(rows).set_index('Distribution')
    metric_label = _metric_label(metric)
    test_label = _test_label(test)
    caption = (
        f"\\textbf{{Valid Power across distributions for {test_label}.}} "
        f"($n={n_ref},\\, m={m_ref}$, Linear Response $Y$, calibrated with {metric_label}). "
        "± SE."
    )
    label = f"tab:{test}_{metric}_distributions"
    latex_path = tables_dir / f"valid_power_distributions_{metric}_{test}.tex"
    latex_df.to_latex(
        latex_path,
        caption=caption,
        label=label,
        escape=False,
        index=True,
        column_format='lccc'
    )

    return csv_path, latex_path


def generate_sweep_tables(final_results, metric, test, output_dir):
    performance = final_results.get('performance', {})
    if not performance:
        return

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    saved_tables = []
    saved_tables.extend(build_feature_tables(performance, metric, test, tables_dir))
    dist_paths = build_distribution_table(performance, metric, test, tables_dir)
    if dist_paths:
        saved_tables.append(dist_paths)

    if saved_tables:
        print(f"Saved sweep summary tables to {tables_dir}")


def generate_second_order_tables(final_results, metric, test, output_dir):
    performance = final_results.get('performance', {})
    if not performance:
        return

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    order_names = {1: "Linear", 2: "Nonlinear"}

    for dist, combos in performance.items():
        tables_dir_dist = tables_dir / dist
        tables_dir_dist.mkdir(parents=True, exist_ok=True)

        csv_rows = []
        csv_index = []
        formatted_entries = []

        for ot in (1, 2):
            test_label = f"{order_names[ot]}"
            for oa in (1, 2):
                adv_label = f"{order_names[oa]}"
                row_key = (test_label, adv_label)
                csv_row = {}
                formatted_row = {'Test': test_label, 'Adversary': adv_label}

                for truth in (1, 2):
                    truth_label = f"{order_names[truth]} truth"
                    perf_payload = combos.get((ot, oa, truth))
                    mean_val = float('nan')
                    std_val = float('nan')
                    if perf_payload is not None:
                        alphas = perf_payload.get('alphas', np.array([0.2]))
                        alpha_idx = int(np.argmin(np.abs(alphas - 0.2)))
                        gains = perf_payload.get('valid_pow_gain')
                        gain_ses = perf_payload.get('valid_pow_gain_se')
                        if gains is not None:
                            mean_val = float(gains[alpha_idx])
                        if gain_ses is not None:
                            std_val = float(gain_ses[alpha_idx])

                    csv_row[(truth_label, 'mean')] = mean_val
                    csv_row[(truth_label, 'std')] = std_val

                    if np.isnan(mean_val):
                        formatted_row[truth_label] = '--'
                    else:
                        std_repr = 0.0 if np.isnan(std_val) else std_val
                        formatted_row[truth_label] = f"{mean_val:.2f}±{std_repr:.3f}"

                csv_rows.append(csv_row)
                csv_index.append(row_key)
                formatted_entries.append(formatted_row)

        if not csv_rows:
            continue

        csv_df = pd.DataFrame(csv_rows, index=pd.MultiIndex.from_tuples(csv_index, names=['GCM', 'Adversary']))
        csv_df.columns = pd.MultiIndex.from_tuples(csv_df.columns, names=['Truth', 'Statistic'])
        csv_path = tables_dir_dist / f"second_order_valid_power_gain_{dist}.csv"
        csv_df.to_csv(csv_path)

        formatted_df = pd.DataFrame(formatted_entries)
        formatted_df.set_index(['Test', 'Adversary'], inplace=True)

        caption = (
            f"\\textbf{{Valid Power Gain across truths.}} "
            f"({order_names[1]} vs. {order_names[2]} GCM and adversary, metric={metric}, distribution={dist}). "
            "± SE."
        )
        label = f"tab:second_order_{metric}_{test}_{dist}"
        latex_path = tables_dir_dist / f"second_order_valid_power_gain_{dist}.tex"
        formatted_df.to_latex(
            latex_path,
            escape=False,
            caption=caption,
            label=label,
            column_format='llcc'
        )

    print(f"Saved second-order summary tables to {tables_dir}")


def aggregate_benchmark_results(results_dir):
    """
    Aggregate and print benchmark results.

    Args:
        results_dir: Path to directory containing benchmark result files
    """
    import numpy as np

    result_dir = Path(results_dir)

    if not result_dir.exists():
        print(f"Error: Result directory {result_dir} does not exist")
        return None

    # Load all result files
    result_files = sorted(result_dir.glob("*.pkl"))
    if not result_files:
        print(f"Error: No result files found in {result_dir}")
        return None

    print(f"Found {len(result_files)} result files in {result_dir}")

    # Collect results from all jobs
    all_results = []
    config = None
    runtime_entries = []

    tests_seen = set()
    contra_seen = set()

    for result_file in result_files:
        with open(result_file, 'rb') as f:
            data = pickle.load(f)

        # Extract result - handle nested structure from save_experiment_result
        if 'result' in data:
            result_data = data['result']

            # Check if it's a nested dict with another 'result' key (benchmark format)
            if isinstance(result_data, dict) and 'result' in result_data:
                all_results.append(result_data['result'])

                # Extract config from nested structure
                job_cfg = result_data.get('config')
                if job_cfg:
                    tests_seen.update(job_cfg.get('tests', []))
                    contra_seen.update(job_cfg.get('contra_methods', []))
                    if config is None:
                        config = job_cfg
            else:
                # Direct result format
                all_results.append(result_data)

                # Try to get config from top level
                job_cfg = data.get('config')
                if job_cfg:
                    tests_seen.update(job_cfg.get('tests', []))
                    contra_seen.update(job_cfg.get('contra_methods', []))
                    if config is None:
                        config = job_cfg

        # Track runtime - check both metadata and top-level
        metadata = data.get('metadata', {})
        runtime = metadata.get('runtime_seconds') or data.get('runtime_seconds')
        if runtime is not None:
            runtime_entries.append((runtime, metadata or data))

    if not all_results:
        print("Error: No results found in result files")
        return None

    # Provide default config if none found
    if config is None:
        print("Warning: No config found in result files, using defaults")
        config = {}

    if tests_seen or contra_seen:
        config = dict(config)
        if tests_seen:
            config['tests'] = sorted(tests_seen)
        if contra_seen:
            config['contra_methods'] = sorted(contra_seen)

    # Extract configuration for display
    if config:
        print(
            "Configuration: x_dist={x}, response={resp}, tests={tests}, "
            "n={n}, m={m}, active={active}, gamma={gamma}".format(
                x=config.get('x_dist'),
                resp=config.get('response'),
                tests=",".join(sorted(config.get('tests', []))) if config.get('tests') else "none",
                n=config.get('n'),
                m=config.get('m'),
                active=config.get('active_features'),
                gamma=config.get('gamma')
            )
        )
    # Aggregate results by test type
    include_gcm = "gcm" in tests_seen
    include_hrt = "hrt" in tests_seen
    contra_methods = config.get('contra_methods', [])

    alpha_grid, summary = summarize_benchmark_runs(
        all_results,
        include_gcm,
        include_hrt,
        contra_methods,
    )

    print(f"\nResults averaged over {len(all_results)} runs:")
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
                if np.isnan(valid_mean):
                    continue
                power_mean = stats["power_mean"][idx]
                fdr_mean = stats["fdr_mean"][idx]
                counts = stats.get("count")
                count_val = int(counts[idx]) if counts is not None and idx < len(counts) else 0
                print(
                    f"  {label:<18s} | n={count_val:3d} valid={valid_mean:.3f} power={power_mean:.3f} fdr={fdr_mean:.3f}"
                )

    # Print runtime summary
    if runtime_entries:
        max_runtime, meta = max(runtime_entries, key=lambda x: x[0])
        job_id = meta.get('job_id', 'unknown')
        print(f"\nLongest job runtime: {max_runtime:.2f}s (job_id={job_id})")

    if config is not None:
        args_ns = SimpleNamespace(
            x_dist=config.get('x_dist', 'unknown'),
            response=config.get('response', 'unknown'),
            metric=config.get('metric', 'fdp'),
            contra_methods=contra_methods,
            tests=config.get('tests', []),
        )
        config_obj = BenchmarkConfig(
            n=config.get('n', 0),
            m=config.get('m', 0),
            active_features=config.get('active_features', 0),
            gamma=config.get('gamma', 0.0),
            x_distribution=config.get('x_dist', 'unknown'),
            response=config.get('response', 'unknown'),
        )
        export_benchmark_tables(
            args_ns,
            config_obj,
            alpha_grid,
            summary,
        )

    return None


def main():
    parser = argparse.ArgumentParser(description="Collect cluster experiment results")
    parser.add_argument('experiment', choices=['sweep', 'second_order', 'semi', 'masks', 'mask_freeze', 'benchmarks'],
                       help='Type of experiment to collect')
    parser.add_argument('metric', choices=['fdp', 'type1', 'area'],
                       help='Metric variant of the experiment')
    parser.add_argument('--results-dir', default='cluster_results',
                       help='Base directory containing result folders')
    parser.add_argument('--output', default=None,
                       help='Optional override for output file path (.pkl)')
    parser.add_argument('--test', choices=['gcm', 'kcit', 'rcit', 'hrt', 'linear_benchmark', 'nonlinear_benchmark', 'gdsc_benchmark'], default='gcm',
                       help='Test variant used in the experiment')

    args = parser.parse_args()

    if args.metric == "area":
        args.metric = "type1"
    subdir = Path(args.results_dir) / f"{args.experiment}_{args.metric}_{args.test}"

    # Benchmarks just print, don't save combined results
    if args.experiment == 'benchmarks':
        print(f"Collecting {args.experiment} ({args.metric}) results from {subdir}")
        aggregate_benchmark_results(subdir)
        return

    # Special handling for semi: collect from both gcm and hrt folders
    if args.experiment == 'semi':
        print(f"Collecting semi ({args.metric}) results from both GCM and HRT folders")
        gcm_dir = Path(args.results_dir) / f"semi_{args.metric}_gcm"
        hrt_dir = Path(args.results_dir) / f"semi_{args.metric}_hrt"

        aggregated_gcm = load_and_aggregate_results(gcm_dir, args.experiment) if gcm_dir.exists() else {}
        aggregated_hrt = load_and_aggregate_results(hrt_dir, args.experiment) if hrt_dir.exists() else {}

        # Combine results from both test types
        aggregated = {}
        for key, val in aggregated_gcm.items():
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].extend(val)
        for key, val in aggregated_hrt.items():
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].extend(val)

        if not aggregated:
            print(f"No results found for semi experiment")
            return
    else:
        print(f"Collecting {args.experiment} ({args.metric}) results from {subdir}")
        # Load and aggregate results
        aggregated = load_and_aggregate_results(subdir, args.experiment)

        if not aggregated:
            print(f"No results found for {args.experiment}")
            return

    # Process based on experiment type
    if args.experiment == 'sweep':
        final_results = aggregate_sweep_results(aggregated, args.metric)
        combined_dir = Path(
            f"outputs/cluster_{args.experiment}_{args.metric}_{args.test}"
        ) / f"combined_{args.experiment}_{args.metric}_{args.test}"
        generate_sweep_tables(final_results, args.metric, args.test, combined_dir)
    elif args.experiment == 'second_order':
        final_results = aggregate_second_order_results(aggregated, args.metric)
        combined_dir = Path(
            f"outputs/cluster_{args.experiment}_{args.metric}_{args.test}"
        ) / f"combined_{args.experiment}_{args.metric}_{args.test}"
        generate_second_order_tables(final_results, args.metric, args.test, combined_dir)
    elif args.experiment == 'semi':
        final_results = aggregate_semi_results(aggregated)
        # Export tables for semi experiment
        export_semi_tables(final_results['performance'], args.metric)
    elif args.experiment in {'masks', 'mask_freeze'}:
        final_results = aggregate_mask_results(aggregated)
        if args.experiment == 'mask_freeze':
            summarize_mask_freeze_results(final_results)

    # Save combined results
    if args.output is None:
        if args.experiment == 'semi':
            # For semi, save in metric-specific folder (combines both tests)
            output_path = Path(args.results_dir) / f"semi_{args.metric}" / f"combined_semi_{args.metric}.pkl"
        else:
            output_path = subdir / f"combined_{args.experiment}_{args.metric}_{args.test}.pkl"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(final_results, f)

    print(f"Combined results saved to {output_path}")
    if args.experiment in ('sweep', 'second_order'):
        cal = final_results['calibration']
        if args.experiment == 'second_order':
            total = sum(len(v) for v in cal.values())
        else:
            total = len(cal)
        print(f"Total parameter combinations: {total}")
    elif args.experiment in {'masks', 'mask_freeze'}:
        print(f"Total parameter combinations: {len(final_results['results'])}")
    elif args.experiment == 'semi':
        print(f"Total response-test combinations: {len(final_results['performance'])}")

    runtime_summary = final_results.get('runtime_summary')
    if runtime_summary and runtime_summary.get('max_runtime_seconds') is not None:
        meta = runtime_summary.get('job_metadata', {})
        runtime = runtime_summary['max_runtime_seconds']
        job_id = meta.get('job_id', 'unknown')
        print(f"Longest job runtime: {runtime:.2f}s (job_id={job_id})")

    # Print summary
    if args.experiment == 'sweep':
        print("\nSweep combinations:")
        for (dist, n, m), data in final_results['calibration'].items():
            print(f"  {dist}, n={n}, m={m}: {data['num_runs']} runs")

    elif args.experiment == 'second_order':
        total_combos = sum(len(v) for v in final_results['calibration'].values())
        print(f"\nSecond-order combinations (total {total_combos}):")
        for dist, combos in final_results['calibration'].items():
            for (oa, ot), data in combos.items():
                runs = data.get('num_runs', 'n/a')
                print(f"  {dist}: Adv={oa}, Test={ot}: {runs} runs")
    elif args.experiment == 'semi':
        print("\nSemi experiment:")
        for (response, test), payload in final_results['performance'].items():
            num_jobs = payload.get('num_jobs', 'n/a')
            print(f"  {response} + {test.upper()}: {num_jobs} jobs aggregated")

    elif args.experiment == 'masks':
        print("\nMask combinations:")
        for (n, p, dist), data in final_results['results'].items():
            print(f"  n={n}, p={p}, {dist}: {data['num_runs']} runs")


if __name__ == '__main__':
    main()
