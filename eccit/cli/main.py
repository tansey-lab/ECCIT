#!/usr/bin/env python3
"""
- sweep: different datasets and distributions
- second_order: higher order tests instead of just linear
- masks: verifying learned masks
- gradients: gradient flow analysis for mask training

Usage:
    eccit sweep fdp gcm
    eccit sweep fdp kcit
    eccit sweep fdp hrt
    eccit masks type1 gcm
    eccit second_order fdp gcm --output-dir custom_outputs/
    eccit gradients type1
    eccit benchmarks type1

    # Test parameter is optional (defaults to 'gcm'):
    eccit sweep fdp
    eccit masks type1
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'experiment',
        choices=['sweep', 'second_order', 'masks', 'gradients', 'benchmarks'],
        help='Type of experiment to run'
    )
    
    parser.add_argument(
        'metric',
        choices=['fdp', 'type1', 'area'],
        default='type1',
        nargs='?',
        help='Miscalibration metric to optimize (default: type1)'
    )
    
    parser.add_argument(
        'test',
        choices=['gcm', 'kcit', 'rcit', 'hrt', 'linear_benchmark', 'nonlinear_benchmark'],
        default='gcm',
        nargs='?',
        help='Conditional independence test to use (default: gcm)'
    )

    parser.add_argument(
        '--output-dir',
        default='outputs/',
        help='Directory for output files (default: outputs/)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducibility (default: 0)'
    )

    parser.add_argument(
        '--benchmark-config',
        choices=['linear_benchmark', 'nonlinear_benchmark'],
        default='linear_benchmark',
        help='Preset configuration for benchmark experiment (default: linear_benchmark)'
    )

    parser.add_argument(
        '--response-order',
        type=int,
        choices=[1, 2],
        default=2,
        help='Order of response function: 1=linear, 2=nonlinear (default: 2)'
    )

    args = parser.parse_args()
    
    # Set random seeds
    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.metric == "area":
        args.metric = "type1"
    print(f"Running {args.experiment} experiment with {args.metric} metric")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Test: {args.test}")
    print("="*50)
    
    # Run the specified experiment
    try:
        if args.experiment == 'sweep':
            from eccit.experiments.sweep import run_sweep_experiment
            results = run_sweep_experiment(
                metric=args.metric,
                output_dir=args.output_dir,
                test=args.test,
                response_order=args.response_order
            )
            print(f"\nSweep experiment completed. Results saved to {args.output_dir}")
            
        elif args.experiment == 'second_order':
            from eccit.experiments.second_order import run_second_order_experiment
            results = run_second_order_experiment(
                metric=args.metric,
                output_dir=args.output_dir,
                test=args.test
            )
            print(f"\nSecond-order experiment completed. Results saved to {args.output_dir}")
            
        elif args.experiment == 'masks':
            from eccit.experiments.masks import run_mask_experiment
            results = run_mask_experiment(
                metric=args.metric,
                output_dir=args.output_dir,
                test=args.test
            )
            print(f"\nMask experiment completed. Results saved to {args.output_dir}")

        elif args.experiment == 'gradients':
            from eccit.experiments.gradients import run_gradient_experiment
            results = run_gradient_experiment(
                metric=args.metric,
                output_dir=args.output_dir
            )
            print(f"\nGradient experiment completed. Results saved to {args.output_dir}")

        elif args.experiment == 'benchmarks':

            from eccit.experiments import benchmarks

            preset_args = {
                'linear_benchmark': [
                    '--x-dist', 'correlated',
                    '--response', 'linear_continuous',
                    '--n', '200',
                    '--m', '10',
                    '--active-features', '4',
                    '--gamma', '0.5',
                    '--tests', 'gcm', 'hrt',
                ],
                'nonlinear_benchmark': [
                    '--x-dist', 'gdsc',
                    '--response', 'nonlinear',
                    '--n', '200',
                    '--m', '10',
                    '--active-features', '4',
                    '--gamma', '0.5',
                    '--tests', 'gcm', 'hrt',
                ],
            }

            cli_args = [
                '--metric', args.metric,
            ] + preset_args[args.benchmark_config]

            bench_args = benchmarks.parse_args(cli_args)
            benchmarks.run_experiment(bench_args)
            print("\nBenchmark experiment completed.")

    except ImportError as e:
        print(f"Error importing experiment module: {e}")
        print("Make sure all experiment modules are properly implemented.")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
