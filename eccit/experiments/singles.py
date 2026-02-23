"""Quick single-experiment calibration demos."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np

from eccit.calibration_single import calibrate_single_experiment
from eccit.experiments.plots import plot_single_performance
from eccit.utils.helpers import evaluate_power_true_single, evaluate_type1_true_single, make_X

Z_DIM = 10

@dataclass(frozen=True)
class SingleDGPConfig:
    """Fixed coefficients for one single-experiment configuration."""

    x_idx: np.ndarray
    x_coeff: np.ndarray
    y_lin_idx: np.ndarray
    y_lin_coeff: np.ndarray
    y_nl_idx: int
    y_nl_coeff: float
    g_coeff: float
    noise_x: float = 1.0
    noise_y: float = 1.0
    main_gain: float = 2.0
    nl_gain: float = 3.0
    nl_slope: float = 1.25
    g_gain: float = 1.0
    g_slope: float = 1.0


def _signed_non_tiny(rng: np.random.Generator, size: int, *, min_coef: float = 1.0) -> np.ndarray:
    mags = np.abs(rng.normal(size=size)) + float(min_coef)
    signs = rng.choice(np.array([-1.0, 1.0]), size=size)
    return mags * signs


def build_single_dgp_config(z_dim: int, seed: int) -> SingleDGPConfig:
    """Build a fixed-coefficient nonlinear Y setup for one scenario.

    Y_null follows the same simplified nonlinear structure used by `make_Y(..., order=2)`:
    linear effects plus a tanh nonlinearity on one active covariate.
    """
    rng = np.random.default_rng(seed)
    k_x = min(4, z_dim)
    x_idx = np.sort(rng.choice(z_dim, size=k_x, replace=False))
    x_coeff = _signed_non_tiny(rng, k_x, min_coef=1.0)

    y_block = np.sort(rng.choice(z_dim, size=min(4, z_dim), replace=False))
    y_lin_idx = y_block[: min(2, y_block.size)]
    y_lin_coeff = _signed_non_tiny(rng, y_lin_idx.size, min_coef=1.0)
    y_nl_idx = int(y_block[2]) if y_block.size >= 3 else int(y_block[-1])
    y_nl_coeff = float(np.abs(rng.normal()) + 1.0)
    g_coeff = float(_signed_non_tiny(rng, 1, min_coef=1.0)[0])

    return SingleDGPConfig(
        x_idx=x_idx,
        x_coeff=x_coeff,
        y_lin_idx=y_lin_idx,
        y_lin_coeff=y_lin_coeff,
        y_nl_idx=y_nl_idx,
        y_nl_coeff=y_nl_coeff,
        g_coeff=g_coeff,
    )


def make_generators(cfg: SingleDGPConfig) -> tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    def f_linear(z: np.ndarray) -> np.ndarray:
        noise = np.random.normal(scale=cfg.noise_x, size=z.shape[0])
        return z[:, cfg.x_idx].dot(cfg.x_coeff) + noise

    def h_nonlinear(z: np.ndarray) -> np.ndarray:
        linear = np.zeros(z.shape[0], dtype=float)
        if cfg.y_lin_idx.size > 0:
            linear = cfg.main_gain * z[:, cfg.y_lin_idx].dot(cfg.y_lin_coeff)
        nonlinear = cfg.nl_gain * cfg.y_nl_coeff * np.tanh(cfg.nl_slope * z[:, cfg.y_nl_idx])
        noise = np.random.normal(scale=cfg.noise_y, size=z.shape[0])
        return linear + nonlinear + noise

    def g_signal(x: np.ndarray) -> np.ndarray:
        x_vec = np.asarray(x).reshape(x.shape[0], -1)[:, 0]
        return cfg.g_gain * np.tanh(cfg.g_slope * cfg.g_coeff * x_vec)

    return f_linear, h_nonlinear, g_signal


def draw_z_normal(n: int) -> np.ndarray:
    return make_X(n, Z_DIM, distribution="normal", gamma=0.0)


def draw_z_correlated(n: int) -> np.ndarray:
    return make_X(n, Z_DIM, distribution="normal", gamma=0.5)


def draw_z_gdsc(n: int) -> np.ndarray:
    return make_X(n, Z_DIM, distribution="gdsc")


def draw_z_breast(n: int) -> np.ndarray:
    return make_X(n, Z_DIM, distribution="breast_cancer")


def draw_z_wine(n: int) -> np.ndarray:
    return make_X(n, Z_DIM, distribution="wine")


DRAW_Z_FNS: dict[str, Callable[[int], np.ndarray]] = {
    "normal": draw_z_normal,
    "correlated": draw_z_correlated,
    "gdsc": draw_z_gdsc,
    "breast": draw_z_breast,
    "wine": draw_z_wine,
}


def make_dataset(
    draw_z_fn: Callable[[int], np.ndarray],
    f_fn: Callable[[np.ndarray], np.ndarray],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    Z = draw_z_fn(n)
    X = f_fn(Z)
    return Z, X


def binom_sem(rate: float, batches: int) -> float:
    if batches <= 1:
        return 0.0
    rate = np.clip(rate, 0.0, 1.0)
    return float(np.sqrt(rate * (1.0 - rate) / batches))


def run_scenario(
    name: str,
    draw_z_fn: Callable[[int], np.ndarray],
    f_fn: Callable[[np.ndarray], np.ndarray],
    h_fn: Callable[[np.ndarray], np.ndarray],
    g_fn: Callable[[np.ndarray], np.ndarray],
    test_name: str,
    args: argparse.Namespace,
    alpha_grid: List[float],
    seed_offset: int = 0,
    verbose: bool = True,
) -> List[dict]:
    scenario_rows: List[dict] = []
    if verbose:
        print(f"Dataset: {name} | test={test_name}")

    for idx, alpha in enumerate(alpha_grid):
        Z, X = make_dataset(draw_z_fn, f_fn, args.n_samples)
        cal_seed = None if args.seed is None else args.seed + seed_offset + 101 * idx
        result = calibrate_single_experiment(
            Z,
            X,
            alpha=alpha,
            tau=args.tau,
            num_epochs=args.epochs,
            subsample_frac=args.subsample_frac,
            order_adv=args.order_adv,
            order_test=args.order_test,
            use_linear=not args.use_mlp,
            weight_lr=args.weight_lr,
            draws_per_epoch=args.draws_per_epoch,
            bootstrap_period=args.bootstrap_period,
            test=test_name,
            seed=cal_seed,
        )

        eval_seed_base = None if args.seed is None else args.seed + seed_offset + 907 * idx

        type1_raw = evaluate_type1_true_single(
            draw_z_fn,
            f_fn,
            h_fn,
            test=test_name,
            cutoff=alpha,
            n_samples=args.n_samples,
            num_batches=args.eval_draws,
            seed=eval_seed_base,
        )
        type1_cal = evaluate_type1_true_single(
            draw_z_fn,
            f_fn,
            h_fn,
            test=test_name,
            cutoff=result["calibrated_cutoff"],
            n_samples=args.n_samples,
            num_batches=args.eval_draws,
            seed=None if eval_seed_base is None else eval_seed_base + 1,
        )

        power_raw = evaluate_power_true_single(
            draw_z_fn,
            f_fn,
            h_fn,
            g_fn,
            test=test_name,
            cutoff=alpha,
            n_samples=args.n_samples,
            num_batches=args.eval_draws,
            seed=None if eval_seed_base is None else eval_seed_base + 2,
        )
        power_cal = evaluate_power_true_single(
            draw_z_fn,
            f_fn,
            h_fn,
            g_fn,
            test=test_name,
            cutoff=result["calibrated_cutoff"],
            n_samples=args.n_samples,
            num_batches=args.eval_draws,
            seed=None if eval_seed_base is None else eval_seed_base + 3,
        )

        row = {
            "alpha": alpha,
            "baseline_cutoff": alpha,
            "calibrated_cutoff": result["calibrated_cutoff"],
            "type1_raw": type1_raw,
            "type1_raw_se": binom_sem(type1_raw, args.eval_draws),
            "type1_cal": type1_cal,
            "type1_cal_se": binom_sem(type1_cal, args.eval_draws),
            "power_raw": power_raw,
            "power_raw_se": binom_sem(power_raw, args.eval_draws),
            "power_cal": power_cal,
            "power_cal_se": binom_sem(power_cal, args.eval_draws),
        }
        scenario_rows.append(row)

        if verbose:
            print(
                "  alpha={alpha:.3f} | base thr={base:.4f} cal thr={cal:.4f} | "
                "type-I raw={t1r:.4f} cal={t1c:.4f} | power raw={pr:.4f} cal={pc:.4f}".format(
                    alpha=alpha,
                    base=row["baseline_cutoff"],
                    cal=row["calibrated_cutoff"],
                    t1r=row["type1_raw"],
                    t1c=row["type1_cal"],
                    pr=row["power_raw"],
                    pc=row["power_cal"],
                )
            )

    if verbose:
        print("")
    return scenario_rows


def export_type1_summary(results: dict, alpha_target: float, test_name: str) -> None:
    rows_to_save = []
    print(f"\nType-I summary at alpha={alpha_target:.2f} ({test_name.upper()}):")
    for dataset, rows in results.items():
        if not rows:
            continue
        closest = min(rows, key=lambda r: abs(r["alpha"] - alpha_target))
        summary = {
            "dataset": dataset,
            "alpha": closest["alpha"],
            "type1_raw": closest["type1_raw"],
            "type1_raw_se": closest["type1_raw_se"],
            "type1_cal": closest["type1_cal"],
            "type1_cal_se": closest["type1_cal_se"],
            "power_raw": closest["power_raw"],
            "power_cal": closest["power_cal"],
        }
        rows_to_save.append(summary)
        print(
            f"  {dataset:<12s} | raw={summary['type1_raw']:.3f}±{summary['type1_raw_se']:.3f} "
            f"cal={summary['type1_cal']:.3f}±{summary['type1_cal_se']:.3f}"
        )

    if not rows_to_save:
        return

    out_dir = Path("outputs") / "singles"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"type1_alpha_{alpha_target:.2f}_{test_name}.csv"
    fieldnames = [
        "dataset",
        "alpha",
        "type1_raw",
        "type1_raw_se",
        "type1_cal",
        "type1_cal_se",
        "power_raw",
        "power_cal",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_to_save:
            writer.writerow(row)
    print(f"Saved summary table to {out_path}\n")


def run_configuration(
    scenario_name: str,
    test_name: str,
    cfg_seed: int,
    seed_offset: int,
    args: argparse.Namespace,
    alpha_grid: List[float],
    *,
    verbose: bool = False,
) -> dict:
    draw_z_fn = DRAW_Z_FNS[scenario_name]
    cfg = build_single_dgp_config(Z_DIM, cfg_seed)
    f_fn, h_fn, g_fn = make_generators(cfg)

    if verbose:
        print(
            "  DGP seed={seed} | x_idx={x_idx} | y_lin_idx={y_idx} | y_nl_idx={y_nl}".format(
                seed=cfg_seed,
                x_idx=cfg.x_idx.tolist(),
                y_idx=cfg.y_lin_idx.tolist(),
                y_nl=cfg.y_nl_idx,
            )
        )

    rows = run_scenario(
        scenario_name,
        draw_z_fn,
        f_fn,
        h_fn,
        g_fn,
        test_name,
        args,
        alpha_grid,
        seed_offset=seed_offset,
        verbose=verbose,
    )
    return {
        "scenario": scenario_name,
        "test": test_name,
        "rows": rows,
        "cfg_seed": cfg_seed,
        "x_idx": cfg.x_idx.tolist(),
        "y_lin_idx": cfg.y_lin_idx.tolist(),
        "y_nl_idx": cfg.y_nl_idx,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Single-experiment calibration demos")
    parser.add_argument(
        "--scenario",
        choices=["all", "normal", "independent", "correlated", "gdsc", "breast", "wine"],
        default="all",
    )
    # parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--test", type=str, choices=("gcm", "hrt"), default=None, help="Run a single test")
    parser.add_argument(
        "--tests",
        nargs="*",
        choices=("gcm", "hrt"),
        default=["gcm", "hrt"],
        help="Tests to run (defaults to both). Ignored if --test is provided.",
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--draws-per-epoch", type=int, default=100)
    parser.add_argument("--eval-draws", type=int, default=500)
    parser.add_argument("--bootstrap-period", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--subsample-frac", type=float, default=0.8)
    parser.add_argument("--order-adv", type=int, default=1)
    parser.add_argument("--order-test", type=int, default=1)
    parser.add_argument("--use-mlp", action="store_true", help="use nonlinear adversary")
    parser.add_argument("--weight-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=6,
        help="Parallel workers over (scenario, test) configurations; use 1 for sequential.",
    )
    args = parser.parse_args(args=argv)

    alpha_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    tests = [args.test] if args.test else list(dict.fromkeys(args.tests))
    if not tests:
        tests = ["gcm", "hrt"]

    scenarios: list[tuple[str, Callable[[int], np.ndarray]]] = []
    if args.scenario in ("all", "normal", "independent"):
        scenarios.append(("normal", draw_z_normal))
    if args.scenario in ("all", "correlated"):
        scenarios.append(("correlated", draw_z_correlated))
    if args.scenario in ("all", "gdsc"):
        scenarios.append(("gdsc", draw_z_gdsc))
    if args.scenario == "breast":
        scenarios.append(("breast", draw_z_breast))
    if args.scenario == "wine":
        scenarios.append(("wine", draw_z_wine))

    if not scenarios:
        raise ValueError("No scenarios selected")

    tasks: list[tuple[str, str, int, int]] = []
    for test_idx, test_name in enumerate(tests):
        for scenario_idx, (scenario_name, _draw_z_fn) in enumerate(scenarios):
            base_seed = 0 if args.seed is None else int(args.seed)
            cfg_seed = base_seed + 10_000 * (test_idx + 1) + 1_000 * (scenario_idx + 1)
            seed_offset = 50_000 * test_idx + 5_000 * scenario_idx
            tasks.append((scenario_name, test_name, cfg_seed, seed_offset))

    total = len(tasks)
    results_by_test: dict[str, dict[str, list[dict]]] = {test_name: {} for test_name in tests}

    if total > 1 and int(args.n_jobs) > 1:
        max_workers = min(int(args.n_jobs), total)
        print(f"Running {total} configurations in parallel with {max_workers} workers")
        done = 0
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(
                    run_configuration,
                    scenario_name,
                    test_name,
                    cfg_seed,
                    seed_offset,
                    args,
                    alpha_grid,
                    verbose=False,
                ): (scenario_name, test_name)
                for (scenario_name, test_name, cfg_seed, seed_offset) in tasks
            }
            for future in as_completed(future_map):
                result = future.result()
                done += 1
                scenario_name, test_name = future_map[future]
                print(
                    f"[{done}/{total}] finished test={test_name} scenario={scenario_name} "
                    f"(cfg_seed={result['cfg_seed']})"
                )
                print(
                    "  DGP x_idx={x_idx} y_lin_idx={y_idx} y_nl_idx={y_nl}".format(
                        x_idx=result["x_idx"],
                        y_idx=result["y_lin_idx"],
                        y_nl=result["y_nl_idx"],
                    )
                )
                rows = result["rows"]
                results_by_test[test_name][scenario_name] = rows
                if rows:
                    plot_single_performance(scenario_name, rows, test_name=test_name)
    else:
        print(f"Running {total} configurations sequentially")
        for idx, (scenario_name, test_name, cfg_seed, seed_offset) in enumerate(tasks, start=1):
            print(f"[{idx}/{total}] test={test_name} scenario={scenario_name}")
            result = run_configuration(
                scenario_name,
                test_name,
                cfg_seed,
                seed_offset,
                args,
                alpha_grid,
                verbose=True,
            )
            rows = result["rows"]
            results_by_test[test_name][scenario_name] = rows
            if rows:
                plot_single_performance(scenario_name, rows, test_name=test_name)

    for test_name in tests:
        print(f"\nTest: {test_name.upper()}")
        export_type1_summary(results_by_test[test_name], alpha_target=0.2, test_name=test_name)


if __name__ == "__main__":
    main()
