"""Test if benchmarks properly use calibrator after seed fix."""
import numpy as np
import argparse
from eccit.experiments.benchmarks import BenchmarkConfig, run_single_experiment

# Simulate a single benchmark run
config = BenchmarkConfig(
    n=300,
    m=30,
    active_features=8,
    gamma=0.5,
    x_distribution="cancer",
    response="nonlinear",
)

args = argparse.Namespace(
    tests=["gcm"],
    alpha=0.2,
    metric="fdp",
    contra_methods=[],
    num_epochs=10,
    order_adv=2,
    order_test=1,
    seed=42,
    x_dist="cancer",
    response="nonlinear",
    n=300,
    m=30,
    active_features=8,
    gamma=0.5,
    preset=None,
)

print("=" * 80)
print("Testing benchmark calibration with seed fix")
print("=" * 80)

# Run 5 test instances
for run_idx in range(1):
    print(f"\n[Run {run_idx}]")
    _, result = run_single_experiment((run_idx, config, args))

    alphas = result.get("alphas", np.array([args.alpha]))

    def _print_curves(label: str, curves: dict):
        print(f"\n  {label}:")
        for idx, alpha in enumerate(alphas):
            power_val = curves["power"][idx]
            valid_val = curves["valid"][idx]
            fdr_val = curves["fdr"][idx]
            print(
                f"    alpha={alpha:.2f} | power={power_val:.3f} valid={valid_val:.3f} fdr={fdr_val:.3f}"
            )

    if "gcm" in result:
        _print_curves("GCM", result["gcm"])
        _print_curves("Calibrated GCM", result["gcm_cal"])

    if "hrt" in result:
        _print_curves("HRT", result["hrt"])
        _print_curves("Calibrated HRT", result["hrt_cal"])

print("\n" + "=" * 80)
