import numpy as np
import torch
import torch.optim as optim

from eccit.cits import compute_conditional_pvals
from eccit.calibration import YGenerator, miscal_area, miscal_fdp, miscal_type1_multi
from eccit.utils.helpers import make_alpha_adjuster, make_alpha_adjuster_from_fdp
from eccit.utils.sgd_factor_model import factorize


def calibrate_run(X, metric="fdp", use_linear=True, pmax=0.2, num_epochs=200, subsample_frac=0.8,
                  order_adv=2, order_test=1, alpha_train=0.2, mask_lr=5e-3, weight_lr=5e-4,
                  test="gcm", gcm_kwargs=None, kcit_kwargs=None, rcit_kwargs=None, hrt_kwargs=None,
                  hrt_n_components=10, hrt_n_steps=1000, hrt_likelihood='gaussian',
                  freeze_adversary=False, debug_gradients=False):
    """
    Run calibration experiment with specified miscalibration metric.
    
    Adversarially generates worst-case Y responses and learns to calibrate p-values.
    
    Args:
        X: Input features tensor
        metric: "fdp", "area", or "type1" - which miscalibration metric to optimize
        use_linear: Use linear GCM vs neural network
        pmax: Cutoff for area metric
        num_epochs: Training epochs
        subsample_frac: Bootstrap fraction
        order_adv: Polynomial order for adversarial Y generation
        order_test: Polynomial order for GCM test
        alpha_train: Alpha level for FDP metric
        mask_lr: Learning rate for mask logits
        weight_lr: Learning rate for generator weights
        test: Which conditional independence test to use ("gcm", "kcit", "rcit", or "hrt")
        gcm_kwargs: Optional kwargs configuring linear GCM estimators
        kcit_kwargs: Optional kwargs forwarded to the KCIT implementation
        rcit_kwargs: Optional kwargs forwarded to the RCIT implementation
        hrt_kwargs: Optional kwargs forwarded to the HRT implementation
        hrt_n_components: Number of components for factor model (used with HRT)
        hrt_n_steps: Number of steps for factor model fitting (used with HRT)
        hrt_likelihood: Likelihood type for factor model (used with HRT)
        freeze_adversary: If True, freeze the mean model parameters and only learn the mask
        debug_gradients: If True, prints gradient magnitudes for adversary params each epoch

    Returns:
        Tuple of (calibrator, diagnostics, training_history)
    """
    n_samples, n_features = X.shape

    # Fit factor model if using HRT test
    W_hat, V_hat, U_hat = None, None, None
    if test == "hrt":
        print(f"Fitting factor model for HRT test with {hrt_n_components} components...")
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        _, _, W_hat, V_hat, U_hat, _ = factorize(
            X_np,
            n_components=hrt_n_components,
            n_steps=hrt_n_steps,
            likelihood=hrt_likelihood
        )
        num_epochs //= 20
        num_epochs = max(1, num_epochs)

        # Add HRT components and classifier type to hrt_kwargs
        hrt_kwargs = dict(hrt_kwargs or {})
        hrt_kwargs['W_hat'] = W_hat
        hrt_kwargs['V_hat'] = V_hat
        hrt_kwargs['U_hat'] = U_hat
        if 'estimator_type' not in hrt_kwargs:
            hrt_kwargs['estimator_type'] = "linear" if order_test == 1 else "mlp"

    if test == "gcm" and gcm_kwargs is None and order_test > 1:
        gcm_kwargs = {
            "y_estimator": "mlp",
            "x_estimator": "mlp",
        }
    ygen = YGenerator(n_features, order=order_adv)
    mask_params = [ygen.mask_logits]
    other_params = []
    for name, param in ygen.named_parameters():
        if name == "mask_logits":
            continue
        if freeze_adversary:
            param.requires_grad_(False)
        else:
            other_params.append(param)

    if freeze_adversary:
        opt_y = optim.Adam([
            {"params": mask_params, "lr": mask_lr},
        ])
    else:
        opt_y = optim.Adam([
            {"params": mask_params, "lr": mask_lr},
            {"params": other_params, "lr": weight_lr},
        ])
    n_subsample = int(np.round(n_samples * subsample_frac))

    kcit_kwargs = dict(kcit_kwargs or {})
    if test == "kcit":
        kcit_kwargs.setdefault("num_bootstrap", 500)
        kcit_kwargs.setdefault("bootstrap", False)
        kcit_kwargs.setdefault("eigen_temp", 0.02)
        kcit_kwargs.setdefault("eigen_floor", 1e-4)
        if kcit_kwargs.get("subsample_size") is None:
            kcit_kwargs["subsample_size"] = min(n_subsample, 400)
        kcit_kwargs.setdefault("kernel", "rbf")
        kcit_kwargs.setdefault("poly_degree", 2)
        kcit_kwargs.setdefault("poly_bias", 1.0)

    rcit_kwargs = dict(rcit_kwargs or {})
    if test == "rcit":
        rcit_kwargs.setdefault("num_f", 100)
        rcit_kwargs.setdefault("num_f2", 40)
        rcit_kwargs.setdefault("approx", "gamma")
        rcit_kwargs.setdefault("redraw", True)
        rcit_kwargs.setdefault("kernel", "rbf")
        rcit_kwargs.setdefault("poly_degree", 2)
        rcit_kwargs.setdefault("poly_bias", 0.0)
        if not rcit_kwargs.get("redraw"):
            rcit_kwargs.setdefault("feature_cache", {})

    hrt_kwargs = dict(hrt_kwargs or {})
    if test == "hrt" and W_hat is not None:
        hrt_kwargs.update({
            "W_hat": W_hat,
            "V_hat": V_hat,
            "U_hat": U_hat,
        })

    # Training history
    m_hat_list = []
    pvals_list = []
    metric_list = []
    losses_list = []
    mask_prob_hist = []
    weight_hist = []

    weight_hist.append(ygen.weight_snapshot())

    mask_batch = 10
    bootstrap_period = 5
    sub_idx = None

    for epoch in range(num_epochs):
        if (epoch % bootstrap_period) == 0 or sub_idx is None:
            sub_idx = np.random.choice(n_samples, size=n_subsample, replace=True)
        # Bootstrap X
        X_subsample = X[sub_idx]

        with torch.no_grad():
            mask_prob_hist.append(ygen.mask_probs_sigmoid().detach().cpu().numpy())

        losses = []
        metric_monitor = []
        last_m_hat = None
        last_pvals = None

        for _ in range(mask_batch):
            Y, m_hat = ygen(X_subsample)
            null_mask = (m_hat < 0.5)

            pvals = compute_conditional_pvals(
                X_subsample,
                Y,
                test=test,
                order=order_test,
                use_linear=use_linear,
                gcm_kwargs=gcm_kwargs,
                kcit_kwargs=kcit_kwargs,
                rcit_kwargs=rcit_kwargs,
                hrt_kwargs=hrt_kwargs,
            )

            if metric == "fdp":
                fdp = miscal_fdp(pvals, null_mask, alpha=alpha_train)
                losses.append(-fdp)
            elif metric == "type1":
                type1 = miscal_type1_multi(pvals, null_mask, alpha=alpha_train)
                losses.append(-type1)
            elif metric == "area":
                area = miscal_area(pvals[null_mask], cutoff=pmax)
                losses.append(-area)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            with torch.no_grad():
                if metric == "fdp":
                    metric_val = miscal_fdp(pvals, null_mask, alpha=alpha_train).item()
                elif metric == "type1":
                    metric_val = miscal_type1_multi(pvals, null_mask, alpha=alpha_train).item()
                else:
                    if null_mask.sum() > 0:
                        metric_val = miscal_area(pvals[null_mask], cutoff=pmax).item()
                    else:
                        metric_val = 0.0
                metric_monitor.append(metric_val)

            last_m_hat = m_hat
            last_pvals = pvals

        opt_y.zero_grad()
        loss = torch.stack(losses).mean()
        loss.backward()

        if debug_gradients:
            mask_grad = (
                ygen.mask_logits.grad.detach().abs().max().item()
                if ygen.mask_logits.grad is not None
                else 0.0
            )
            max_weight_grad = 0.0
            for name, param in ygen.named_parameters():
                if name == "mask_logits":
                    continue
                if param.grad is not None:
                    max_weight_grad = max(max_weight_grad, param.grad.detach().abs().max().item())
            print(
                f"Epoch {epoch + 1}: loss={loss.item():.4f} | "
                f"max|∇mask|={mask_grad:.3e} max|∇weights|={max_weight_grad:.3e}"
            )
        opt_y.step()

        # Track statistics
        metric_val = float(np.mean(metric_monitor)) if metric_monitor else 0.0
        m_hat_list.append(last_m_hat.detach().cpu().numpy())
        pvals_list.append(last_pvals.detach().cpu().numpy())
        metric_list.append(metric_val)
        loss_val = loss.item()
        losses_list.append(loss_val)

        weight_hist.append(ygen.weight_snapshot())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} | Loss {loss_val:.4f} | Nulls: {int(null_mask.sum().item())}")

    # Post-training evaluation
    best_idx = int(np.argmax(metric_list))
    best_metric = metric_list[best_idx]

    # Average over multiple resamples to get final calibrator
    n_resamples = 100
    grid = np.linspace(0, 1, 100)
    
    running_sum_cdf = np.zeros_like(grid)
    # For FDP-based alpha mapping when optimizing FDP
    alpha_grid_fdp = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    fdp_sum = np.zeros_like(alpha_grid_fdp)
    all_nulls = []
    resample_metrics = []

    all_pvals_eval = []
    all_null_masks_eval = []

    for _ in range(n_resamples):
        # Bootstrap X
        X_subsample = X[np.random.choice(n_samples, size=n_subsample)]

        # Generate Y
        with torch.no_grad():
            Y, m_hat = ygen(X_subsample)

        # Compute p-values
        with torch.no_grad():
            p_tensor = compute_conditional_pvals(
                X_subsample,
                Y,
                test=test,
                order=order_test,
                use_linear=use_linear,
                gcm_kwargs=gcm_kwargs,
                kcit_kwargs=kcit_kwargs,
                rcit_kwargs=rcit_kwargs,
                hrt_kwargs=hrt_kwargs,
                to_numpy=False,
            )
            pvals = p_tensor.detach().cpu().numpy()

        m_hat_np = m_hat.cpu().numpy()
        null_mask_np = m_hat_np < 0.5
        null_p = pvals[null_mask_np]
        all_pvals_eval.append(pvals)
        all_null_masks_eval.append(null_mask_np)
        if null_p.size == 0:
            cdfs = np.zeros_like(grid)
            metric_val = 0.0
        else:
            cdfs = np.array([(null_p <= u).mean() for u in grid])
            if metric == "type1":
                metric_val = miscal_type1_multi(
                    torch.tensor(pvals, dtype=torch.float32),
                    torch.tensor(null_mask_np, dtype=torch.bool),
                    alpha=alpha_train,
                ).item()
            elif metric == "fdp":
                metric_val = miscal_fdp(
                    torch.tensor(pvals, dtype=torch.float32),
                    torch.tensor(null_mask_np, dtype=torch.bool),
                    alpha=alpha_train,
                ).item()
            else:
                metric_val = miscal_area(torch.tensor(null_p, dtype=torch.float32)).item()
        resample_metrics.append(metric_val)
        running_sum_cdf += cdfs
        all_nulls.append(null_p)

        # Also accumulate FDP under BH across alpha grid
        # Helper: BH FDP with numpy arrays
        p_np = pvals
        is_null = null_mask_np
        m_total = p_np.size
        if m_total > 0:
            p_sorted = np.sort(p_np)
            ranks = np.arange(1, m_total + 1)
            for i, a in enumerate(alpha_grid_fdp):
                bh_line = a * ranks / m_total
                sat = p_sorted <= bh_line
                if np.any(sat):
                    k = np.max(np.where(sat)[0]) + 1
                else:
                    k = 0
                t = a * k / m_total
                reject = p_np <= t
                R = int(np.sum(reject))
                if R == 0:
                    fdp_val = 0.0
                else:
                    V = int(np.sum(reject & is_null))
                    fdp_val = float(V) / float(R)
                fdp_sum[i] += fdp_val

    mean_cdf = running_sum_cdf / n_resamples
    mean_cdf = np.maximum(mean_cdf, grid)
    
    p_null_all = np.concatenate(all_nulls)
    # Average FDP curve and build alpha adjuster for FDP
    fdp_curve = fdp_sum / n_resamples
    # Enforce monotonicity and bounds to ensure a valid inverse mapping
    fdp_curve = np.maximum.accumulate(np.clip(fdp_curve, 0.0, 1.0))
    alpha_adjust_fdp = make_alpha_adjuster_from_fdp(alpha_grid_fdp, fdp_curve)

    if metric == "type1":
        def calibrator(p_raw):
            return np.asarray(p_raw, dtype=float)
    else:
        def calibrator(p_raw):
            p_cal = np.interp(p_raw, grid, mean_cdf, 0.0, 1.0)
            return np.maximum(p_raw, p_cal)

    alpha_adjust = make_alpha_adjuster(grid, mean_cdf)
    calibrated_cutoff = None
    if metric == "type1":
        calibrated_cutoff = float(alpha_adjust(alpha_train))

    # Final mask statistics
    with torch.no_grad():
        final_probs = ygen.mask_probs_sigmoid().cpu().numpy()
        final_probs = np.nan_to_num(final_probs, nan=0.0, posinf=1.0, neginf=0.0)
        k_hat = int(np.round(final_probs.sum()))
        if 0 < k_hat < len(final_probs):
            final_mask = ygen.deterministic_mask(topk=k_hat).cpu().numpy()
        else:
            final_mask = (final_probs >= 0.5).astype(np.float32)

    # Return results
    eval_pvals = np.concatenate(all_pvals_eval) if all_pvals_eval else np.array([], dtype=float)
    eval_null_mask = np.concatenate(all_null_masks_eval) if all_null_masks_eval else np.array([], dtype=bool)

    diagnostics = {
        'losses': np.array(losses_list),
        'mask_prob_hist': np.stack(mask_prob_hist, axis=0),
        'final_mask': final_mask,
        'final_probs': final_probs,
        'ygen': ygen,
        'weight_history': np.stack(weight_hist, axis=0) if weight_hist else None,
        'order_adv': order_adv,
        'order_test': order_test,
        'pmax': pmax,
        'alpha_adjust': alpha_adjust,
        'alpha_grid_fdp': alpha_grid_fdp,
        'fdp_curve': fdp_curve,
        'alpha_adjust_fdp': alpha_adjust_fdp,
        'alpha_train': alpha_train,
        'calibrated_cutoff': calibrated_cutoff,
        'use_linear': use_linear,
        'test': test,
        'kcit_kwargs': kcit_kwargs,
        'rcit_kwargs': rcit_kwargs,
        'hrt_kwargs': hrt_kwargs,
        'gcm_kwargs': gcm_kwargs,
        'metric': metric,
        'epochs_ran': len(losses_list),
        'best_loss': float(np.min(losses_list)) if losses_list else None,
        'eval_pvals': eval_pvals,
        'eval_null_mask': eval_null_mask,
        # HRT components (if fitted)
        'W_hat': W_hat,
        'V_hat': V_hat,
        'U_hat': U_hat,
    }
    
    return calibrator, m_hat_list, pvals_list, best_metric, p_null_all, mean_cdf, grid, diagnostics


def calibrate_step(n, m, distribution, use_linear, metric="fdp", test="gcm",
                   gcm_kwargs=None, kcit_kwargs=None, rcit_kwargs=None, hrt_kwargs=None,
                   hrt_n_components=10, hrt_n_steps=1000, hrt_likelihood='gaussian'):
    """Wrapper for single calibration run with data generation."""
    from eccit.utils.helpers import make_X
    
    X_np = make_X(n, m, distribution=distribution)
    X = torch.from_numpy(X_np).float()
    return (
        (distribution, n, m),
        calibrate_run(
            X,
            metric=metric,
            use_linear=use_linear,
            pmax=0.2,
            test=test,
            gcm_kwargs=gcm_kwargs,
            kcit_kwargs=kcit_kwargs,
            rcit_kwargs=rcit_kwargs,
            hrt_kwargs=hrt_kwargs,
            hrt_n_components=hrt_n_components,
            hrt_n_steps=hrt_n_steps,
            hrt_likelihood=hrt_likelihood,
        ),
    )


def calibrate_step_order(n, m, distribution, order_adv, order_test, alpha_train=0.2,
                         metric="fdp", test="gcm", gcm_kwargs=None,
                         kcit_kwargs=None, rcit_kwargs=None, hrt_kwargs=None,
                         hrt_n_components=10, hrt_n_steps=1000, hrt_likelihood='gaussian'):
    """Wrapper for calibration run with specific orders."""
    from eccit.utils.helpers import make_X
    
    X_np = make_X(n, m, distribution=distribution)
    X = torch.from_numpy(X_np).float()
    return (
        (order_adv, order_test, alpha_train),
        calibrate_run(
            X,
            metric=metric,
            order_adv=order_adv,
            order_test=order_test,
            alpha_train=alpha_train,
            use_linear=(order_test == 1),
            test=test,
            gcm_kwargs=gcm_kwargs,
            kcit_kwargs=kcit_kwargs,
            rcit_kwargs=rcit_kwargs,
            hrt_kwargs=hrt_kwargs,
            hrt_n_components=hrt_n_components,
            hrt_n_steps=hrt_n_steps,
            hrt_likelihood=hrt_likelihood,
        ),
    )
