"""Gradient flow analysis for mask training experiments."""

import numpy as np
import torch
from pathlib import Path

from eccit.calibration import YGenerator, miscal_area, miscal_fdp, miscal_type1_multi
from eccit.cits import compute_conditional_pvals
from eccit.utils.helpers import make_X

DEFAULT_TYPE1_ALPHA = 0.2


def expected_area_uniform_per_run(m_null: int) -> float:
    """Compute expected area under offset curve for uniform p-values."""
    if m_null <= 0:
        return float("nan")
    c = (2.0 * np.pi) ** 0.5 / 8.0  # ≈ 0.3133
    return float(c / np.sqrt(m_null))


def analyze_gradients(
    mask,
    n=20,
    p=10,
    metric="type1",
    steps=5,
    train_mask=False,
    weight_lr=1e-3,
    mask_lr=1e-2,
    order=1,
    alpha=DEFAULT_TYPE1_ALPHA,
):
    """Run optimization steps and collect gradient statistics for analysis."""
    torch.manual_seed(0)
    np.random.seed(0)

    X = torch.from_numpy(make_X(n, p, distribution="correlated")).float()
    mask_np = np.asarray(mask, dtype=float)
    mask_t = torch.tensor(mask_np, dtype=torch.float32)

    ygen = YGenerator(p, order=order)
    # Initialise mask logits close to the provided bitvector.
    with torch.no_grad():
        logits = torch.full_like(mask_t, fill_value=-6.0)
        logits[mask_t >= 0.5] = 6.0
        ygen.mask_logits.copy_(logits)

    ygen.mask_logits.requires_grad_(train_mask)

    optim_params = [{"params": ygen.mean_parameters(), "lr": weight_lr}]
    if train_mask:
        optim_params.append({"params": [ygen.mask_logits], "lr": mask_lr})
    opt = torch.optim.Adam(optim_params)

    initial_weight = ygen.mean_first_layer().weight.detach().clone()
    initial_mask = torch.sigmoid(ygen.mask_logits.detach()).clone()

    null_indices = np.where(mask_np < 0.5)[0]
    print("Mask:", mask_np)
    print("Null indices:", null_indices)
    if train_mask:
        print(
            "Initial mask probs:",
            torch.sigmoid(ygen.mask_logits).detach().cpu().numpy(),
        )

    for step in range(steps):
        opt.zero_grad()
        mask_used = torch.sigmoid(ygen.mask_logits) if train_mask else mask_t
        Y, _ = ygen.forward_with_mask(X, mask_used)
        pvals = compute_conditional_pvals(
            X,
            Y,
            test="gcm",
            order=1,
            use_linear=True,
        )

        null_mask = mask_t < 0.5
        if null_mask.sum() == 0:
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            if metric == "area":
                loss = -miscal_area(pvals[null_mask])
            elif metric == "type1":
                loss = -miscal_type1_multi(pvals, null_mask, alpha=alpha)
            elif metric == "fdp":
                loss = -miscal_fdp(pvals, null_mask)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        loss.backward()

        weight_grad = ygen.mean_first_layer().weight.grad.detach().abs().max().item()
        mask_grad = (
            ygen.mask_logits.grad.detach().abs().max().item()
            if train_mask and ygen.mask_logits.grad is not None
            else 0.0
        )

        # Check Adam optimizer state before step
        param_groups = opt.param_groups
        adam_state_info = []
        for group_idx, group in enumerate(param_groups):
            for param_idx, param in enumerate(group['params']):
                if param in opt.state:
                    state = opt.state[param]
                    exp_avg = state.get('exp_avg', torch.zeros_like(param))
                    exp_avg_sq = state.get('exp_avg_sq', torch.zeros_like(param))
                    adam_state_info.append(f"G{group_idx}P{param_idx}:m={exp_avg.abs().max():.2e},v={exp_avg_sq.abs().max():.2e}")

        opt.step()

        weight_delta = (ygen.mean_first_layer().weight.detach() - initial_weight).abs().max().item()
        mask_delta = (
            (torch.sigmoid(ygen.mask_logits.detach()) - initial_mask).abs().max().item()
            if train_mask
            else 0.0
        )

        print(
            f"step {step}: loss={loss.item():.4f}, max|grad_w|={weight_grad:.4e}, "
            f"max|grad_mask|={mask_grad:.4e}, max|dw|={weight_delta:.4e}, max|dpi|={mask_delta:.4e}"
        )

    if train_mask:
        print(
            "Final mask probs:",
            torch.sigmoid(ygen.mask_logits).detach().cpu().numpy(),
        )


def run_noise_grid_analysis(
    n_list=(100, 200, 400),
    m_list=(10, 20, 50),
    num_runs=500,
    test="gcm",
    metric="type1",
    alpha=DEFAULT_TYPE1_ALPHA,
):
    """Run noise-only calibration offset analysis over (n,m) grid."""
    results = {}
    grid = np.linspace(0.0, 1.0, 101, dtype=float)

    for n in n_list:
        for m in m_list:
            print(f"\n[Noise-only] n={n}, m={m}, runs={num_runs}")
            per_run_vals = []

            for run in range(num_runs):
                torch.manual_seed(10_000 + 97 * run)
                np.random.seed(10_000 + 97 * run)

                X_np = make_X(n, m, distribution="correlated")
                X = torch.from_numpy(X_np).float()
                Y = torch.randn(n)

                pvals = compute_conditional_pvals(
                    X, Y,
                    test=test,
                    order=1,
                    use_linear=True,
                    to_numpy=True
                )
                # All m are null in noise-only
                if metric == "area":
                    ecdf = (pvals[:, None] <= grid[None, :]).mean(axis=0)
                    offset_r = ecdf - grid
                    area_r = float(np.trapz(np.abs(offset_r), grid))
                    per_run_vals.append(area_r)
                elif metric == "type1":
                    per_run_vals.append(float((pvals <= alpha).mean()))
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

            mean_val = float(np.mean(per_run_vals))
            std_val = float(np.std(per_run_vals, ddof=1))
            if metric == "area":
                expected_val = expected_area_uniform_per_run(m_null=m)
            else:
                expected_val = float(alpha)

            print(
                f"  mean per-run metric = {mean_val:.5f}  (sd={std_val:.5f})   "
                f"[uniform baseline = {expected_val:.5f}]"
            )

            results[(n, m)] = {
                "mean_metric": mean_val,
                "std_metric": std_val,
                "expected_metric": expected_val,
            }
    return results


def run_gradient_experiment(metric="type1", output_dir="outputs"):
    """
    Run gradient flow analysis experiment.

    Main entry point for gradient experiments.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    if metric == "area":
        metric = "type1"
    print(f"Running gradient flow analysis with metric: {metric}")

    # Define test mask (3 null features, 7 active features)
    base_mask = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=float)

    # Well-specified model (order=1)
    print("\n" + "="*60)
    print("WELL-SPECIFIED MODEL (YGenerator order=1)")
    print("="*60)

    print("=== Frozen hard mask (expected stalled gradients) ===")
    analyze_gradients(
        mask=base_mask,
        metric=metric,
        train_mask=False,
        order=1
    )

    print("\n=== Trainable soft mask (gradients flow through mask logits) ===")
    analyze_gradients(
        mask=base_mask,
        metric=metric,
        train_mask=True,
        order=1
    )

    print("\n=== All-null mask, frozen ===")
    analyze_gradients(mask=np.zeros_like(base_mask), metric=metric, train_mask=False, order=1)

    # Mis-specified model (order=2)
    print("\n" + "="*60)
    print("MIS-SPECIFIED MODEL (YGenerator order=2)")
    print("="*60)

    print("=== Frozen hard mask (expected stalled gradients) ===")
    analyze_gradients(
        mask=base_mask,
        metric=metric,
        train_mask=False,
        order=2
    )

    print("\n=== Trainable soft mask (gradients flow through mask logits) ===")
    analyze_gradients(
        mask=base_mask,
        metric=metric,
        train_mask=True,
        order=2
    )

    print("\n=== All-null mask, frozen ===")
    analyze_gradients(mask=np.zeros_like(base_mask), metric=metric, train_mask=False, order=2)

    # Calibration offset analysis
    print("\n" + "="*60)
    print("CALIBRATION OFFSET ANALYSIS")
    print("="*60)

    results = run_noise_grid_analysis(
        n_list=(50, 100),
        m_list=(10, 20, 40),
        num_runs=500,
        test="gcm",
        metric=metric,
        alpha=DEFAULT_TYPE1_ALPHA,
    )

    summary_label = "Type-I Error" if metric == "type1" else "area"
    print(f"\nSummary (mean per-run {summary_label} vs uniform baseline):")
    for (n, m) in sorted(results):
        r = results[(n, m)]
        print(f"  (n={n:>3}, m={m:>2})  mean={r['mean_metric']:.5f} "
              f"(expected={r['expected_metric']:.5f})  sd={r['std_metric']:.5f}")

    print("\nGradient experiment completed.")


if __name__ == "__main__":
    run_gradient_experiment(metric="type1")
