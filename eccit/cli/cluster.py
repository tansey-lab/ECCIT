#!/usr/bin/env python3
"""
Runner for slurm jobs.

Usage:
    eccit-cluster submit sweep fdp gcm
    eccit-cluster plot sweep fdp gcm

    # Test parameter is optional (defaults to 'gcm'):
    eccit-cluster generate sweep fdp
    eccit-cluster submit masks type1
"""

import json
import argparse
import os
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch
import subprocess
import sys
import time
import shutil

from joblib import Parallel, delayed

from eccit.utils.job_utils import (
    save_experiment_result,
    generate_sweep_jobs,
    generate_second_order_jobs,
    generate_semi_jobs,
    generate_mask_jobs,
    generate_mask_freeze_jobs,
    write_job_file,
)
from eccit.utils.helpers import make_alpha_adjuster, make_alpha_adjuster_from_fdp

DEFAULT_BENCHMARK_ALPHA_LEVELS = [0.05, 0.10, 0.15, 0.20]


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
def _aggregate_sweep_runs(runs, metric):
    aggregated = {}
    if metric == "area":
        metric = "type1"
    if metric == 'fdp':
        diag0 = runs[0][7]
        alpha_grid = diag0.get('alpha_grid_fdp', np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]))
        fdp_curves = np.vstack([r[7]['fdp_curve'] for r in runs])
        fdp_mean = np.maximum.accumulate(np.clip(fdp_curves.mean(axis=0), 0.0, 1.0))
        alpha_adjustor = make_alpha_adjuster_from_fdp(alpha_grid, fdp_mean)
        aggregated.update(dict(
            alpha_grid=alpha_grid,
            fdp_mean=fdp_mean,
            all_nulls=np.concatenate([r[4] for r in runs]),
            calibrator=(lambda p: p),
            alpha_adjustor=alpha_adjustor,
            max_area=max(r[3] for r in runs),
            num_runs=len(runs)
        ))
    else:
        curves = np.vstack([r[5] for r in runs])
        joint_cdf = np.clip(np.maximum.accumulate(curves.max(axis=0)), 0, 1)
        grid = runs[0][6]
        if metric == "type1":
            def calibrator_fn(p_raw):
                p_arr = np.asarray(p_raw, dtype=float)
                return float(p_arr) if np.isscalar(p_raw) else p_arr
        else:
            def calibrator_fn(p_raw):
                p_arr = np.asarray(p_raw)
                p_cal = np.interp(p_arr, grid, joint_cdf, 0.0, 1.0)
                return float(p_cal) if np.isscalar(p_raw) else p_cal
        aggregated.update(dict(
            grid=grid,
            joint_cdf=joint_cdf,
            mean_cdf=curves.mean(axis=0),
            all_nulls=np.concatenate([r[4] for r in runs]),
            calibrator=calibrator_fn,
            alpha_adjustor=make_alpha_adjuster(grid, joint_cdf),
            max_type1=max(r[3] for r in runs),
            num_runs=len(runs)
        ))
    return aggregated


def run_single_sweep_job(params, output_dir):
    from eccit.calibration_runner import calibrate_step
    from eccit.experiments.sweep import sweep_performance

    start_time = time.time()
    runs = []
    base_seed = params['seed']
    for r in range(params['num_runs']):
        seed = base_seed + r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        _, result = calibrate_step(
            params['n'],
            params['m'],
            params['distribution'],
            True,
            params['metric'],
            params.get('test', 'gcm'),
            params.get('gcm_kwargs'),
            params.get('kcit_kwargs'),
            params.get('rcit_kwargs'),
            params.get('hrt_kwargs'),
            params.get('hrt_n_components', 10),
            params.get('hrt_n_steps', 1000),
            params.get('hrt_likelihood', 'gaussian'),
        )
        runs.append(result)

    aggregated = _aggregate_sweep_runs(runs, params['metric'])
    key = (params['distribution'], params['n'], params['m'])
    perf = sweep_performance(
        {key: aggregated},
        metric=params['metric'],
        response_order=params.get('response_order', 1),
        test=params.get('test', 'gcm'),
        gcm_kwargs=params.get('gcm_kwargs'),
        kcit_kwargs=params.get('kcit_kwargs'),
        rcit_kwargs=params.get('rcit_kwargs'),
        hrt_kwargs=params.get('hrt_kwargs'),
    )

    result_payload = {
        'calibration': {key: aggregated},
        'performance': perf
    }

    runtime = time.time() - start_time
    metadata = dict(params)
    metadata['runtime_seconds'] = runtime
    metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'
    save_experiment_result(result_payload, output_dir / f"{params['job_id']:04d}.pkl", metadata)
    print(f"Job {params['job_id']} finished in {runtime:.2f} seconds")
    return result_payload


def _aggregate_second_order_runs(runs, metric):
    aggregated = {}
    if metric == "area":
        metric = "type1"
    if metric == 'fdp':
        alpha_grid = runs[0][7].get('alpha_grid_fdp', np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]))
        fdp_curves = np.vstack([r[7]['fdp_curve'] for r in runs])
        fdp_mean = np.maximum.accumulate(np.clip(fdp_curves.mean(axis=0), 0.0, 1.0))
        alpha_adjustor = make_alpha_adjuster_from_fdp(alpha_grid, fdp_mean)
        aggregated.update(dict(
            alpha_grid=alpha_grid,
            fdp_mean=fdp_mean,
            calibrator=(lambda p: p),
            alpha_adjustor=alpha_adjustor,
            num_runs=len(runs)
        ))
    else:
        curves = np.vstack([r[5] for r in runs])
        joint_cdf = np.clip(np.maximum.accumulate(curves.max(axis=0)), 0, 1)
        grid = runs[0][6]
        if metric == "type1":
            def calibrator_fn(p_raw):
                p_arr = np.asarray(p_raw, dtype=float)
                return float(p_arr) if np.isscalar(p_raw) else p_arr
        else:
            def calibrator_fn(p_raw):
                p_arr = np.asarray(p_raw)
                p_cal = np.interp(p_arr, grid, joint_cdf, 0.0, 1.0)
                return float(p_cal) if np.isscalar(p_raw) else p_cal
        aggregated.update(dict(
            grid=grid,
            joint_cdf=joint_cdf,
            calibrator=calibrator_fn,
            alpha_adjustor=make_alpha_adjuster(grid, joint_cdf),
            num_runs=len(runs)
        ))
    return aggregated


def run_single_second_order_job(params, output_dir):
    from eccit.calibration_runner import calibrate_step_order
    from eccit.experiments.second_order import evaluate_second_order_performance

    start_time = time.time()
    base_seed = params['seed']
    alpha_trains = params.get('alpha_trains', [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    tasks = []
    for alpha_train in alpha_trains:
        for run_idx in range(params['num_runs']):
            seed = base_seed + run_idx + int(alpha_train * 100)
            tasks.append((alpha_train, seed))

    def _run_calibration(alpha_train: float, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        _, result = calibrate_step_order(
            params['n'],
            params['m'],
            params['distribution'],
            params['order_adv'],
            params['order_test'],
            alpha_train,
            params['metric'],
            params.get('test', 'gcm'),
            params.get('gcm_kwargs'),
            params.get('kcit_kwargs'),
            params.get('rcit_kwargs'),
            params.get('hrt_kwargs'),
            params.get('hrt_n_components', 10),
            params.get('hrt_n_steps', 1000),
            params.get('hrt_likelihood', 'gaussian'),
        )
        return result

    if len(tasks) == 1:
        runs = [_run_calibration(*tasks[0])]
    else:
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        try:
            max_workers = int(slurm_cpus) if slurm_cpus is not None else None
        except ValueError:
            max_workers = None
        if not max_workers:
            max_workers = os.cpu_count() or 1
        n_jobs = min(len(tasks), max_workers)
        runs = Parallel(n_jobs=n_jobs)(delayed(_run_calibration)(alpha_train, seed) for alpha_train, seed in tasks)

    aggregated = _aggregate_second_order_runs(runs, params['metric'])
    key = (params['order_adv'], params['order_test'])
    dist = params['distribution']

    cal_results_local = {key: aggregated}
    perf = evaluate_second_order_performance(
        cal_results_local,
        n=params['n'],
        m=params['m'],
        distribution=dist,
        n_responses=params.get('n_responses', 100),
        metric=params['metric'],
        test=params.get('test', 'gcm'),
        gcm_kwargs=params.get('gcm_kwargs'),
        kcit_kwargs=params.get('kcit_kwargs'),
        rcit_kwargs=params.get('rcit_kwargs'),
        hrt_kwargs=params.get('hrt_kwargs'),
    )

    cal_payload = {(dist, params['order_adv'], params['order_test']): aggregated}
    perf_payload = {(dist, *k): v for k, v in perf.items()}

    result_payload = {
        'calibration': cal_payload,
        'performance': perf_payload
    }

    runtime = time.time() - start_time
    metadata = dict(params)
    metadata['runtime_seconds'] = runtime
    metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'

    save_experiment_result(result_payload, output_dir / f"{params['job_id']:04d}.pkl", metadata)
    print(f"Job {params['job_id']} finished in {runtime:.2f} seconds")
    return result_payload


def run_single_semi_job(params, output_dir):
    from eccit.experiments.semi import run_semi_experiment, aggregate_semi_runs

    start_time = time.time()
    response = params['response']
    num_runs = params.get('num_runs', 1)
    base_seed = params.get('seed', 0)

    run_results = []
    for r in range(num_runs):
        seed = base_seed + r
        result = run_semi_experiment(
            n=params['n'],
            m=params['m'],
            test=params.get('test', 'gcm'),
            metric=params['metric'],
            response=response,
            num_responses=params.get('n_responses', 100),
            feat_size=params.get('feat_size'),
            seed=seed,
            gdsc_path=params.get('gdsc_path'),
            alpha_train=params.get('alpha_train', 0.2),
            gcm_kwargs=params.get('gcm_kwargs'),
            kcit_kwargs=params.get('kcit_kwargs'),
            rcit_kwargs=params.get('rcit_kwargs'),
            hrt_kwargs=params.get('hrt_kwargs'),
            hrt_n_components=params.get('hrt_n_components', 10),
            hrt_n_steps=params.get('hrt_n_steps', 1000),
            hrt_likelihood=params.get('hrt_likelihood', 'gaussian'),
            contra_methods=params.get('contra_methods', ['crt', 'hrt', 'fastcrt']),
        )
        run_results.append(result)

    aggregated = aggregate_semi_runs(run_results)
    result_payload = {
        'performance': {response: aggregated},
    }

    runtime = time.time() - start_time
    metadata = dict(params)
    metadata['runtime_seconds'] = runtime
    metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'

    save_experiment_result(result_payload, output_dir / f"{params['job_id']:04d}.pkl", metadata)
    print(f"Job {params['job_id']} ({response}) finished in {runtime:.2f} seconds")
    return result_payload


def run_single_mask_job(params, output_dir):
    from eccit.experiments.masks import run_mask_experiment
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    start_time = time.time()
    result = run_mask_experiment(
        n=params['n'],
        p=params['p'],
        distribution=params['distribution'],
        num_epochs=params.get('num_epochs', 5000),
        mask_draws=5,
        top_k=20,
        S_samples=1000,
        metric=params['metric'],
        output_dir=output_dir / f"mask_job_{params['job_id']}",
        test=params.get('test', 'gcm'),
        gcm_kwargs=params.get('gcm_kwargs'),
        kcit_kwargs=params.get('kcit_kwargs'),
        rcit_kwargs=params.get('rcit_kwargs'),
        hrt_kwargs=params.get('hrt_kwargs'),
    )
    runtime = time.time() - start_time
    metadata = dict(params)
    metadata['runtime_seconds'] = runtime
    metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'
    save_experiment_result(result, output_dir / f"{params['job_id']:04d}.pkl", metadata)
    print(f"Job {params['job_id']} finished in {runtime:.2f} seconds")
    return result


def run_single_mask_freeze_job(params, output_dir):
    from eccit.experiments.masks import run_mask_experiment

    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_time = time.time()
    result = run_mask_experiment(
        n=params['n'],
        p=params['p'],
        distribution=params['distribution'],
        num_epochs=params.get('num_epochs', 200),
        metric=params['metric'],
        output_dir=output_dir / f"mask_freeze_job_{params['job_id']}",
        test=params.get('test', 'gcm'),
        n_runs=params.get('n_runs', 1),
        toy_seed=seed,
        gcm_kwargs=params.get('gcm_kwargs'),
        kcit_kwargs=params.get('kcit_kwargs'),
        rcit_kwargs=params.get('rcit_kwargs'),
        hrt_kwargs=params.get('hrt_kwargs'),
    )
    runtime = time.time() - start_time
    metadata = dict(params)
    metadata['runtime_seconds'] = runtime
    metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'
    save_experiment_result(result, output_dir / f"{params['job_id']:04d}.pkl", metadata)
    print(f"Job {params['job_id']} finished in {runtime:.2f} seconds")
    return result


def run_single_benchmark_job(params, output_dir):
    """Run a single benchmark job (exactly 1 run)."""
    from eccit.experiments.benchmarks import run_single_experiment, BenchmarkConfig
    import argparse

    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_time = time.time()

    # Build config from params
    config = BenchmarkConfig(
        n=params['n'],
        m=params['m'],
        active_features=params['active_features'],
        gamma=params['gamma'],
        x_distribution=params['x_dist'],
        response=params['response'],
    )

    # Create minimal args namespace
    args = argparse.Namespace(
        tests=params['tests'],
        alpha=params['alpha'],
        metric=params['metric'],
        contra_methods=params.get('contra_methods', ['crt', 'hrt', 'fastcrt']),
        num_epochs=params.get('num_epochs', 200),
        order_adv=params.get('order_adv', 1),
        order_test=params.get('order_test', 1),
        seed=params['seed'],
        x_dist=params['x_dist'],
        response=params['response'],
        n=params['n'],
        m=params['m'],
        active_features=params['active_features'],
        gamma=params['gamma'],
        preset=params.get('preset'),
    )
    args.alpha_levels = np.array(DEFAULT_BENCHMARK_ALPHA_LEVELS, dtype=float)

    # Run single experiment
    run_idx = params['run_idx']
    _, result = run_single_experiment((run_idx, config, args))

    runtime = time.time() - start_time

    result_payload = {
        'result': result,
        'run_idx': run_idx,
        'config': params,
    }

    metadata = dict(params)
    metadata['runtime_seconds'] = runtime
    metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'

    save_experiment_result(result_payload, output_dir / f"{params['job_id']:04d}.pkl", metadata)
    print(f"Job {params['job_id']} (run {run_idx}) finished in {runtime:.2f} seconds")
    return result_payload


def generate_benchmark_jobs(
    n, m, active_features, gamma, x_dist, response,
    alpha, metric, tests, contra_methods,
    num_epochs, seed, preset,
    order_adv, order_test,
):
    """Generate benchmark jobs; each job evaluates one method."""
    jobs = []
    runs_per_method = 100
    tests = list(tests or [])
    contras = list(contra_methods or [])
    job_id = 0

    if not tests and not contras:
        tests = ["gcm", "hrt"]

    for test in tests:
        for _ in range(runs_per_method):
            job_params = {
                'job_id': job_id,
                'run_idx': job_id,
                'n': n,
                'm': m,
                'active_features': active_features,
                'gamma': gamma,
                'x_dist': x_dist,
                'response': response,
                'alpha': alpha,
                'metric': metric,
                'tests': [test],
                'contra_methods': [],
                'num_epochs': num_epochs,
                'seed': seed,
                'preset': preset,
                'order_adv': order_adv,
                'order_test': order_test,
            }
            jobs.append(job_params)
            job_id += 1

    for contra in contras:
        for _ in range(runs_per_method):
            job_params = {
                'job_id': job_id,
                'run_idx': job_id,
                'n': n,
                'm': m,
                'active_features': active_features,
                'gamma': gamma,
                'x_dist': x_dist,
                'response': response,
                'alpha': alpha,
                'metric': metric,
                'tests': [],
                'contra_methods': [contra],
                'num_epochs': num_epochs,
                'seed': seed,
                'preset': preset,
                'order_adv': order_adv,
                'order_test': order_test,
            }
            jobs.append(job_params)
            job_id += 1

    return jobs


def generate_jobs_and_script(experiment, metric, test, num_runs=10, response_order=2):
    job_dir = Path("cluster_jobs")
    job_dir.mkdir(exist_ok=True)

    if experiment == "benchmarks":
        preset = test or "linear_benchmark"
        if preset not in {"linear_benchmark", "nonlinear_benchmark"}:
            preset = "linear_benchmark"
        if preset == "linear_benchmark":
            jobs = generate_benchmark_jobs(
                n=200,
                m=10,
                active_features=4,
                gamma=0.5,
                x_dist="correlated",
                response="linear_continuous",
                alpha=0.2,
                metric=metric,
                tests=["gcm", "hrt"],
                contra_methods=["crt", "hrt", "fastcrt"],
                num_epochs=200,
                order_adv=1,
                order_test=1,
                seed=0,
                preset=preset,
            )
        elif preset == "nonlinear_benchmark":
            jobs = generate_benchmark_jobs(
                n=300,
                m=30,
                active_features=10,
                gamma=0.5,
                x_dist="cancer",
                response="nonlinear",
                alpha=0.2,
                metric=metric,
                tests=["gcm", "hrt"],
                contra_methods=["hrt", "fastcrt"],
                num_epochs=300,
                order_adv=2,
                order_test=1,
                seed=0,
                preset=preset,
            )
        else:
            raise ValueError(f"Unknown benchmark preset '{preset}'")
    elif experiment == "sweep":
        # Use larger sample sizes for HRT test
        n_list = list(range(25, 501, 25)) if test == "hrt" else list(range(25, 501, 25))
        jobs = generate_sweep_jobs(
            n_list,
            [10, 25, 50],
            ["normal", "correlated", "laplace"],
            num_runs,
            metric,
            test,
            response_order,
        )
    elif experiment == "second_order":
        jobs = generate_second_order_jobs(
            n=125,
            m=25,
            distribution_list=["normal", "correlated", "laplace"],
            num_runs=num_runs,
            metric=metric,
            test=test,
        )
    elif experiment == "semi":
        jobs = generate_semi_jobs(
            n=125,
            m=25,
            responses=("linear", "nonlinear"),
            num_runs=1,  # Each job does 1 run
            metric=metric,
            test=test,
            n_responses=100,
            num_jobs=100,  # Create num_runs independent jobs per response
        )
    elif experiment == "masks":
        jobs = generate_mask_jobs(
            [100],
            [5, 10],
            ["correlated", "normal"],
            num_runs,
            metric,
            test,
        )
    elif experiment == "mask_freeze":
        jobs = generate_mask_freeze_jobs(
            num_jobs=100,
            n=20,
            p=10,
            distribution="normal",
            metric=metric,
            test=test,
            num_epochs=200,
            n_runs=1,
            base_seed=0,
        )

    job_file = job_dir / f"{experiment}_{metric}_{test}_jobs.txt"
    write_job_file(jobs, job_file)

    cpus_per_task = 16 if experiment in {"second_order"} else 1

    out_dir_name = f"{experiment}_{metric}_{test}"
    result_dir = Path("cluster_results") / out_dir_name
    if result_dir.exists():
        archive_dir = result_dir.parent / "old"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archived = archive_dir / f"{out_dir_name}_{timestamp}"
        print(f"Archiving existing results to {archived}")
        shutil.move(str(result_dir), str(archived))

    repo_root = Path.cwd().resolve()
    script_path = job_dir / f"submit_{out_dir_name}.sbatch"
    with open(script_path, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=calib_{experiment}_{metric}
#SBATCH --partition=componc_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --array=1-{len(jobs)}
#SBATCH --time=4:00:00
#SBATCH --mem={'16GB' if test == 'hrt' else '8GB'}
#SBATCH --output=cluster_logs/slurm_%A_%a.out
#SBATCH --error=cluster_logs/slurm_%A_%a.err

mkdir -p cluster_logs
mkdir -p {result_dir}
cd {repo_root}
{sys.executable} -m eccit.cli.cluster run {experiment} {metric} $SLURM_ARRAY_TASK_ID --job-file {job_file.name} --output-dir {result_dir} --test {test}
""")

    print(f"Created {len(jobs)} jobs: {job_file}")
    print(f"SLURM script: {script_path}")
    print(f"Submit: sbatch {script_path}")
    return jobs, job_file, script_path


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    gen_parser = subparsers.add_parser('generate')
    gen_parser.add_argument('experiment', choices=['sweep', 'second_order', 'semi', 'masks', 'mask_freeze', 'benchmarks'])
    gen_parser.add_argument('metric', choices=['fdp', 'type1', 'area'])
    gen_parser.add_argument('test', choices=['gcm', 'kcit', 'rcit', 'hrt', 'linear_benchmark', 'nonlinear_benchmark'], default='gcm', nargs='?')
    gen_parser.add_argument('--num-runs', type=int, default=10, help='Number of Monte Carlo runs per job (and per response for semi).')
    gen_parser.add_argument('--response-order', type=int, choices=[1, 2], default=2, help='Order of response function: 1=linear, 2=nonlinear (default: 2)')

    submit_parser = subparsers.add_parser('submit')
    submit_parser.add_argument('experiment', choices=['sweep', 'second_order', 'semi', 'masks', 'mask_freeze', 'benchmarks'])
    submit_parser.add_argument('metric', choices=['fdp', 'type1', 'area'])
    submit_parser.add_argument('test', choices=['gcm', 'kcit', 'rcit', 'hrt', 'linear_benchmark', 'nonlinear_benchmark'], default='gcm', nargs='?')
    submit_parser.add_argument('--num-runs', type=int, default=10, help='Number of Monte Carlo runs per job (and per response for semi).')
    submit_parser.add_argument('--response-order', type=int, choices=[1, 2], default=2, help='Order of response function: 1=linear, 2=nonlinear (default: 2)')

    plot_parser = subparsers.add_parser('plot')
    plot_parser.add_argument('experiment', choices=['sweep', 'second_order', 'semi', 'masks', 'mask_freeze', 'benchmarks'])
    plot_parser.add_argument('metric', choices=['fdp', 'type1', 'area'])
    plot_parser.add_argument('test', choices=['gcm', 'kcit', 'rcit', 'hrt', 'linear_benchmark', 'nonlinear_benchmark'], default='gcm', nargs='?')
    plot_parser.add_argument('--out-dir', default=None, help='Override output directory for plots')

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('experiment', choices=['sweep', 'second_order', 'semi', 'masks', 'mask_freeze', 'benchmarks'])
    run_parser.add_argument('metric', choices=['fdp', 'type1', 'area'])
    run_parser.add_argument('task_id', type=int)
    run_parser.add_argument('--job-file', required=True)
    run_parser.add_argument('--output-dir', default='cluster_results')
    run_parser.add_argument('--test', choices=['gcm', 'kcit', 'rcit', 'hrt', 'linear_benchmark', 'nonlinear_benchmark'], default=None,
                            help='Override test (defaults to job file value)')

    args = parser.parse_args()

    if args.metric == "area":
        args.metric = "type1"

    if args.command == 'generate':
        generate_jobs_and_script(args.experiment, args.metric, args.test, num_runs=args.num_runs,
                                 response_order=getattr(args, 'response_order', 1))

    elif args.command == 'submit':
        _, _, script_path = generate_jobs_and_script(args.experiment, args.metric, args.test, num_runs=args.num_runs,
                                                      response_order=getattr(args, 'response_order', 1))
        subprocess.run(['sbatch', str(script_path)], check=True)

    elif args.command == 'plot':
        if args.experiment == 'benchmarks':
            presets = []
            base_dir = Path('cluster_results')
            for preset in ['linear_benchmark', 'nonlinear_benchmark']:
                dir_name = f"benchmarks_{args.metric}_{preset}"
                if (base_dir / dir_name).exists():
                    presets.append(preset)

            if not presets and args.test is not None:
                presets.append(args.test)
            if not presets:
                presets = ['linear_benchmark']

            for preset in presets:
                subprocess.run([
                    sys.executable, '-m', 'eccit.cli.collect_results',
                    args.experiment, args.metric, '--test', preset
                ], check=True)
        else:
            # Collect results
            subprocess.run([
                sys.executable, '-m', 'eccit.cli.collect_results',
                args.experiment, args.metric, '--test', args.test
            ], check=True)

            if args.experiment not in {'mask_freeze'}:
                # Generate plots for experiments with plotting support
                if args.experiment == 'semi':
                    # For semi, combined file is in metric-specific folder
                    combined_file = Path(
                        f"cluster_results/semi_{args.metric}/"
                        f"combined_semi_{args.metric}.pkl"
                    )
                    out_dir = args.out_dir or f"outputs/cluster_semi_{args.metric}"
                else:
                    combined_file = Path(
                        f"cluster_results/{args.experiment}_{args.metric}_{args.test}/"
                        f"combined_{args.experiment}_{args.metric}_{args.test}.pkl"
                    )
                    out_dir = args.out_dir or f"outputs/cluster_{args.experiment}_{args.metric}_{args.test}"

                subprocess.run([
                    sys.executable, '-m', 'eccit.cli.plot_results',
                    str(combined_file), '--out-dir', out_dir
                ], check=True)

    elif args.command == 'run':
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(Path("cluster_jobs") / args.job_file) as f:
            line = f.readlines()[args.task_id - 1]
        params = json.loads(line.strip())
        params['metric'] = args.metric
        if args.test is not None:
            params['test'] = args.test

        if args.experiment == 'sweep':
            run_single_sweep_job(params, output_dir)
        elif args.experiment == 'second_order':
            run_single_second_order_job(params, output_dir)
        elif args.experiment == 'semi':
            run_single_semi_job(params, output_dir)
        elif args.experiment == 'masks':
            run_single_mask_job(params, output_dir)
        elif args.experiment == 'mask_freeze':
            run_single_mask_freeze_job(params, output_dir)
        elif args.experiment == 'benchmarks':
            run_single_benchmark_job(params, output_dir)
    

if __name__ == '__main__':
    main()
