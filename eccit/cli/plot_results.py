#!/usr/bin/env python3
"""Plot aggregated cluster results based on combined *.pkl files."""

import argparse
from pathlib import Path
import dill as pickle

from eccit.experiments.plots import (
    plot_cdf_sweep,
    plot_perf_sweep,
    plot_calibration_offset,
    plot_second_order_calibration,
    plot_second_order_performance,
    plot_mask_probabilities,
    plot_mask_prob_trajectories,
    plot_mask_ranking,
    plot_sampled_mask_ranks,
    plot_semi_performance,
    plot_sweep_valid_power_gain_bars,
    plot_sweep_valid_power_gain_by_features,
)


def _print_runtime_summary(data):
    summary = data.get('runtime_summary') if isinstance(data, dict) else None
    if not summary:
        return
    max_runtime = summary.get('max_runtime_seconds')
    if max_runtime is None:
        return
    meta = summary.get('job_metadata', {})
    job_id = meta.get('job_id', 'unknown')
    dist = meta.get('distribution')
    label_parts = [f"job_id={job_id}"]
    if dist:
        label_parts.append(f"dist={dist}")
    if 'n' in meta:
        label_parts.append(f"n={meta['n']}")
    if 'm' in meta:
        label_parts.append(f"m={meta['m']}")
    print(f"Longest job runtime: {max_runtime:.2f}s ({', '.join(label_parts)})")


def _filter_student_t(calibration: dict, performance: dict):
    if isinstance(calibration, dict):
        calibration = {k: v for k, v in calibration.items() if k[0] != "student_t"}
    if isinstance(performance, dict):
        performance = {k: v for k, v in performance.items() if k[0] != "student_t"}
    return calibration, performance
def detect_experiment(filename: Path):
    name = filename.stem.lower()
    if "sweep" in name:
        experiment = "sweep"
    elif "second" in name:
        experiment = "second_order"
    elif "semi" in name:
        experiment = "semi"
    elif "mask" in name:
        experiment = "masks"
    else:
        raise ValueError(f"Cannot infer experiment type from {filename}")

    if "fdp" in name:
        metric = "fdp"
    elif "type1" in name:
        metric = "type1"
    elif "area" in name:
        metric = "type1"
    else:
        metric = None

    # Extract test type from filename
    test = "gcm"  # default
    if "_hrt" in name:
        test = "hrt"
    elif "_gcm" in name:
        test = "gcm"
    elif "_kcit" in name:
        test = "kcit"
    elif "_rcit" in name:
        test = "rcit"

    return experiment, metric, test


def plot_file(path: Path, out_dir: Path):
    experiment, metric_hint, test = detect_experiment(path)
    with open(path, "rb") as f:
        data = pickle.load(f)

    _print_runtime_summary(data)

    if experiment == "sweep":
        metric = metric_hint or "fdp"
        if metric == "area":
            metric = "type1"
        out_dir.mkdir(parents=True, exist_ok=True)
        calibration, performance = _filter_student_t(
            data.get('calibration', {}),
            data.get('performance', {}),
        )
        dists = tuple(sorted({k[0] for k in calibration.keys()}))
        n_list = tuple(sorted({k[1] for k in calibration.keys()}))
        m_list = tuple(sorted({k[2] for k in calibration.keys()}))
        plot_cdf_sweep(calibration, n_list=n_list, m_list=m_list, dist_list=dists, out_dir=out_dir)
        plot_perf_sweep(performance, n_list=n_list, m_list=m_list, dist_list=dists, out_dir=out_dir, metric=metric)
        plot_calibration_offset(calibration, n_list=n_list, m_list=m_list, dist_list=dists, out_dir=out_dir, metric=metric)
    elif experiment == "second_order":
        metric = metric_hint or "fdp"
        if metric == "area":
            metric = "type1"
        out_dir.mkdir(parents=True, exist_ok=True)
        calibration_by_dist = data['calibration']
        performance_by_dist = data['performance']
        for dist, cal_payload in calibration_by_dist.items():
            dist_dir = out_dir / dist
            dist_dir.mkdir(parents=True, exist_ok=True)
            plot_second_order_calibration(cal_payload, dist_dir, distribution=dist, test=test)
            perf_payload = performance_by_dist.get(dist, {})
            if perf_payload:
                plot_second_order_performance(perf_payload, dist_dir, metric=metric, distribution=dist, test=test)
    elif experiment == "masks":
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_results = data['results'] if isinstance(data, dict) and 'results' in data else data
        for (n, p, dist), payload in mask_results.items():
            for idx, (result, metadata) in enumerate(payload['individual_results']):
                diag = result['diagnostics']
                metric = metadata.get('metric', 'fdp')
                if metric == "area":
                    metric = "type1"
                run_id = metadata.get('job_id', idx)
                run_dir = out_dir / f"n{n}_p{p}_{dist}" / f"run_{run_id:04d}"
                run_dir.mkdir(parents=True, exist_ok=True)

                plot_mask_probabilities(diag['final_probs'], run_dir / "mask_probs.pdf", metric=metric)
                plot_mask_prob_trajectories(diag['mask_prob_hist'], run_dir / "mask_prob_trajectories.pdf", metric=metric)
                plot_mask_ranking(
                    result['masks_table'], diag['final_mask'],
                    out_top=str(run_dir / "mask_ranking_top.pdf"),
                    out_bottom=str(run_dir / "mask_ranking_worst.pdf"),
                    metric=metric
                )
                plot_sampled_mask_ranks(result['samples_table'], out_path=str(run_dir / "sampled_mask_ranks.pdf"), metric=metric)
    elif experiment == "semi":
        out_dir.mkdir(parents=True, exist_ok=True)
        performance = data.get('performance', {}) if isinstance(data, dict) else {}
        plot_semi_performance(performance, out_dir / "semi_valid_power.pdf")
    else:
        raise ValueError(f"Unsupported experiment type: {experiment}")

    print(f"Processed {path} -> {out_dir}")


def plot_combined_sweep_bars(files, base_out):
    """Create combined bar plot if all 4 sweep files (gcm/hrt x type1/fdp) are present."""
    # Detect which files are sweep files
    sweep_files = {}
    for file in files:
        path = Path(file)
        experiment, metric, test = detect_experiment(path)
        if experiment == 'sweep' and test in ['gcm', 'hrt'] and metric in ['type1', 'fdp']:
            sweep_files[(test, metric)] = path

    # Check if we have all 4 combinations
    required = [('gcm', 'type1'), ('gcm', 'fdp'), ('hrt', 'type1'), ('hrt', 'fdp')]
    if not all((test, metric) in sweep_files for test, metric in required):
        return  # Not all files present, skip combined plot

    # Load all 4 result files
    results_dict = {}
    for (test, metric), path in sweep_files.items():
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        cal, perf = _filter_student_t(payload.get('calibration', {}), payload.get('performance', {}))
        filtered = dict(payload)
        filtered['calibration'] = cal
        filtered['performance'] = perf
        results_dict[(test, metric)] = filtered

    # Create combined bar plots
    combined_dir = base_out / 'combined_sweep'
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Plot by distribution
    out_path = combined_dir / 'valid_power_gain_bars.pdf'
    plot_sweep_valid_power_gain_bars(results_dict, out_path=out_path)
    print(f"Created combined sweep bar plot: {out_path}")

    # Plot by features
    out_path_features = combined_dir / 'valid_power_gain_by_features.pdf'
    plot_sweep_valid_power_gain_by_features(results_dict, out_path=out_path_features)
    print(f"Created combined sweep by features plots: {combined_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated cluster results")
    parser.add_argument('files', nargs='+', help='Combined result pickle files')
    parser.add_argument('--out-dir', default='outputs/cluster_plots',
                        help='Base directory for generated plots')
    args = parser.parse_args()

    base_out = Path(args.out_dir)

    # Plot individual files
    for file in args.files:
        path = Path(file)
        experiment, metric, test = detect_experiment(path)
        target = base_out / path.stem
        plot_file(path, target)

    # Try to create combined sweep bar plot if all files are present
    plot_combined_sweep_bars(args.files, base_out)


if __name__ == '__main__':
    main()
