import os
import json
import dill as pickle
from pathlib import Path
from itertools import product
from typing import Optional, Sequence
import numpy as np


def save_experiment_result(result, output_path, metadata=None):
    """
    Save experiment results in standard pickle format.
    
    Args:
        result: Experiment result object
        output_path: Path to save result
        metadata: Optional metadata dict
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'result': result,
        'metadata': metadata or {}
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)


def load_experiment_result(result_path):
    """
    Load experiment result from pickle file.
    
    Returns:
        result: The saved experiment result
        metadata: Associated metadata dict
    """
    with open(result_path, 'rb') as f:
        data = pickle.load(f)
    return data['result'], data.get('metadata', {})


def generate_sweep_jobs(n_list, m_list, dist_list, num_runs=10, metric="fdp", test="gcm", response_order=1):
    """
    Generate parameter combinations for sweep experiment cluster jobs.

    Returns list of job parameter dictionaries.

    Parameters:
    -----------
    response_order : int
        Order of response function (1=linear, 2=nonlinear)
    """
    jobs = []
    job_id = 0

    for distribution in dist_list:
        for n in n_list:
            for m in m_list:
                if m > 0.5 * n: # bootstrapping X minimums
                    continue
                jobs.append({
                    'job_id': job_id,
                    'experiment': 'sweep',
                    'distribution': distribution,
                    'n': n,
                    'm': m,
                    'num_runs': num_runs,
                    'metric': metric,
                    'test': test,
                    'response_order': response_order,
                    'seed': job_id
                })
                job_id += 1

    return jobs


def generate_second_order_jobs(n=100, m=50, distribution="correlated", 
                              distribution_list=None, num_runs=5, metric="fdp", test="gcm"):
    """
    Generate parameter combinations for second-order experiment cluster jobs.
    """
    jobs = []
    job_id = 0
    orders = [1, 2]
    alpha_trains = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    distributions = distribution_list or [distribution]
    for dist in distributions:
        for order_adv in orders:
            for order_test in orders:
                gcm_kwargs = None
                hrt_kwargs = None
                if test == "gcm":
                    if order_test == 1:
                        gcm_kwargs = {
                            "y_estimator": "linear",
                            "x_estimator": "linear",
                        }
                    else:
                        gcm_kwargs = {
                            "y_estimator": "poly2",
                            "y_estimator_params": {"lambda": 1e-1},
                            "x_estimator": "linear",
                        }
                elif test == "hrt":
                    if order_test == 1:
                        hrt_kwargs = {
                            "estimator_type": "linear",
                            "conditional_type": "linear",
                        }
                    else:
                        hrt_kwargs = {
                            "estimator_type": "poly2",
                            "estimator_params": {"lambda": 1e-1},
                            "conditional_type": "poly2",
                            "conditional_kwargs": {"lambda": 1e-1},
                        }
                jobs.append({
                    'job_id': job_id,
                    'experiment': 'second_order',
                    'n': n,
                    'm': m,
                    'distribution': dist,
                    'order_adv': order_adv,
                    'order_test': order_test,
                    'alpha_trains': alpha_trains,
                    'num_runs': num_runs,
                    'metric': metric,
                    'test': test,
                    'gcm_kwargs': gcm_kwargs,
                    'hrt_kwargs': hrt_kwargs,
                    'seed': job_id
                })
                job_id += 1
    
    return jobs


def generate_semi_jobs(
    *,
    n: int = 200,
    m: int = 50,
    responses: Optional[Sequence[str]] = None,
    num_runs: int = 1,
    metric: str = "fdp",
    test: str = "gcm",
    n_responses: int = 100,
    num_jobs: int = 1,
) -> list:
    """Generate jobs for the semi-supervised GDSC experiment."""
    responses = list(responses or ("linear", "nonlinear"))
    jobs = []
    job_id = 0

    for repeat in range(max(1, num_jobs)):
        base_seed = repeat * 1000
        for response in responses:
            jobs.append({
                'job_id': job_id,
                'experiment': 'semi',
                'n': n,
                'm': m,
                'response': response,
                'responses_all': responses,
                'num_runs': num_runs,
                'metric': metric,
                'test': test,
                'n_responses': n_responses,
                'seed': base_seed + job_id,
            })
            job_id += 1

    return jobs


def generate_mask_jobs(n_list=[20, 100, 200], p_list=[5, 10], 
                      distribution_list=["normal", "correlated", "laplace"], 
                      num_runs=5, metric="fdp", test="gcm"):
    """
    Generate parameter combinations for mask experiment cluster jobs.
    """
    jobs = []
    job_id = 0
    
    for n in n_list:
        for p in p_list:
            for distribution in distribution_list:
                for run in range(num_runs):
                    jobs.append({
                        'job_id': job_id,
                        'experiment': 'masks',
                        'n': n,
                        'p': p,
                        'distribution': distribution,
                        'run': run,
                        'metric': metric,
                        'test': test,
                        'seed': job_id,
                        'num_epochs': 10000 if p <= 10 else 5000  # Adjust epochs by problem size
                    })
                    job_id += 1
    
    return jobs


def generate_mask_freeze_jobs(num_jobs=100, n=20, p=10, distribution="normal",
                              metric="fdp", test="gcm", num_epochs=200, n_runs=1,
                              base_seed=0):
    """Generate jobs for the frozen-vs-trained mask comparison experiment."""
    jobs = []

    for job_id in range(num_jobs):
        jobs.append({
            'job_id': job_id,
            'experiment': 'mask_freeze',
            'n': n,
            'p': p,
            'distribution': distribution,
            'metric': metric,
            'test': test,
            'seed': base_seed + job_id,
            'num_epochs': num_epochs,
            'n_runs': n_runs,
        })

    return jobs


def write_job_file(jobs, output_path):
    """
    Write job parameters to text file for cluster submission.
    
    Each line contains JSON-encoded job parameters.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for job in jobs:
            f.write(json.dumps(job) + '\n')


def read_job_file(job_file_path):
    """
    Read job parameters from text file.
    
    Returns list of job parameter dictionaries.
    """
    jobs = []
    with open(job_file_path, 'r') as f:
        for line in f:
            jobs.append(json.loads(line.strip()))
    return jobs


def load_and_aggregate_results(result_dir, experiment_type):
    """
    Load all result files from directory and aggregate by experiment type.
    
    Args:
        result_dir: Directory containing pickle result files
        experiment_type: "sweep", "second_order", or "masks"
        
    Returns:
        Dict mapping parameter combinations to lists of results
    """
    result_dir = Path(result_dir)
    result_files = list(result_dir.glob("*.pkl"))
    
    if not result_files:
        print(f"No result files found for {experiment_type} in {result_dir}")
        return {}
    
    aggregated = {}
    
    for result_file in result_files:
        try:
            result, metadata = load_experiment_result(result_file)
            
            # Create key based on experiment type
            if experiment_type == "sweep":
                key = (metadata.get('distribution'), metadata.get('n'), metadata.get('m'))
            elif experiment_type == "second_order":
                key = (
                    metadata.get('distribution'),
                    metadata.get('order_adv'),
                    metadata.get('order_test')
                )
            elif experiment_type in {"masks", "mask_freeze"}:
                key = (metadata.get('n'), metadata.get('p'), metadata.get('distribution'))
            elif experiment_type == "semi":
                key = (metadata.get('response'), metadata.get('n'), metadata.get('m'))
            else:
                key = str(metadata.get('job_id', 'unknown'))
            
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append((result, metadata))
            
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    print(f"Loaded {len(result_files)} result files for {experiment_type}")
    print(f"Aggregated into {len(aggregated)} parameter combinations")
    
    return aggregated


def create_job_files_for_cluster():
    """
    Create all job files needed for cluster submission.
    
    Generates job parameter files for all three experiment types.
    """
    job_dir = Path("cluster_jobs")
    job_dir.mkdir(exist_ok=True)
    
    print("Creating cluster job files...")
    
    # Sweep experiment jobs
    sweep_jobs = generate_sweep_jobs(
        n_list=[20, 50, 100, 500],
        m_list=[10, 25, 50],
        dist_list=["normal", "correlated"],
        num_runs=5,
        metric="fdp"
    )
    write_job_file(sweep_jobs, job_dir / "sweep_jobs.txt")
    print(f"Created {len(sweep_jobs)} sweep jobs")
    
    # Second-order experiment jobs
    second_order_jobs = generate_second_order_jobs(
        n=100, m=50, distribution="correlated",
        num_runs=4, metric="fdp"
    )
    write_job_file(second_order_jobs, job_dir / "second_order_jobs.txt")
    print(f"Created {len(second_order_jobs)} second-order jobs")
    
    # Mask experiment jobs
    mask_jobs = generate_mask_jobs(
        n_list=[100, 200], p_list=[8, 10, 12],
        distribution_list=["normal", "correlated"],
        num_runs=3, metric="fdp"
    )
    write_job_file(mask_jobs, job_dir / "masks_jobs.txt")
    print(f"Created {len(mask_jobs)} mask jobs")
    
    print(f"All job files saved to {job_dir}/")
    return sweep_jobs, second_order_jobs, mask_jobs


if __name__ == '__main__':
    # Generate job files when run directly
    create_job_files_for_cluster()
