import numpy as np
import torch
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_wine
from statsmodels.stats.multitest import multipletests

from eccit.cits.gcm import gcm_single
from eccit.cits.hrt import hrt_single


def make_single_experiment(n, draw_z, f, h, *, seed=None):
    """Generate a single-experiment sample triple."""
    if seed is not None:
        np.random.seed(seed)

    Z = np.asarray(draw_z(n))
    X = np.asarray(f(Z)).reshape(n, -1)
    Y = np.asarray(h(Z)).reshape(n)

    return Z, X, Y


def evaluate_power_true_single(
    draw_z,
    f,
    h,
    g,
    *,
    test: str = "gcm",
    cutoff: float = 0.2,
    n_samples: int = 256,
    num_batches: int = 20,
    gcm_kwargs: dict | None = None,
    hrt_kwargs: dict | None = None,
    seed: int | None = None,
) -> float:
    """Estimate statistical power using the true data generator for single tests."""

    rng = np.random.RandomState(seed)
    test_lower = (test or "gcm").lower()
    gcm_kwargs = dict(gcm_kwargs or {})
    hrt_kwargs = dict(hrt_kwargs or {})

    cutoff = float(cutoff)
    total = 0.0

    for _ in range(num_batches):
        batch_seed = None if seed is None else int(rng.randint(0, 2**32))
        Z_np, X_np, Y_np = make_single_experiment(
            n_samples,
            draw_z,
            f,
            h,
            seed=batch_seed,
        )

        # Inject true signal through X in addition to Z
        Y_np = np.asarray(Y_np).reshape(-1) + np.asarray(g(X_np)).reshape(-1)

        Z = torch.as_tensor(Z_np, dtype=torch.float64)
        X = torch.as_tensor(X_np, dtype=torch.float64)
        Y = torch.as_tensor(Y_np, dtype=torch.float64)

        if test_lower == "gcm":
            p_val = gcm_single(
                torch.as_tensor(Z, dtype=torch.float64),
                torch.as_tensor(X, dtype=torch.float64),
                torch.as_tensor(Y, dtype=torch.float64),
                to_numpy=False,
                **gcm_kwargs,
            )
        elif test_lower == "hrt":
            p_val = hrt_single(
                Z,
                X,
                Y,
                **hrt_kwargs,
            )
        else:
            raise ValueError(f"Unsupported test '{test}' for single-experiment evaluation")

        p_tensor = torch.as_tensor(p_val, dtype=torch.float64)
        total += (p_tensor <= cutoff).double().mean().item()

    return total / float(num_batches)


def evaluate_type1_true_single(
    draw_z,
    f,
    h,
    *,
    test: str = "gcm",
    cutoff: float = 0.2,
    n_samples: int = 256,
    num_batches: int = 20,
    gcm_kwargs: dict | None = None,
    hrt_kwargs: dict | None = None,
    seed: int | None = None,
) -> float:
    """Estimate type-I error using the true data generator for single-feature tests."""

    rng = np.random.RandomState(seed)
    test_lower = (test or "gcm").lower()
    gcm_kwargs = dict(gcm_kwargs or {})
    hrt_kwargs = dict(hrt_kwargs or {})

    cutoff = float(cutoff)
    total = 0.0

    for batch_idx in range(num_batches):
        batch_seed = None if seed is None else rng.randint(0, 2**32)
        Z_np, X_np, Y_np = make_single_experiment(
            n_samples,
            draw_z,
            f,
            h,
            seed=batch_seed,
        )

        Z = torch.as_tensor(Z_np, dtype=torch.float64)
        X = torch.as_tensor(X_np, dtype=torch.float64)
        Y = torch.as_tensor(Y_np, dtype=torch.float64)

        if test_lower == "gcm":
            p_val = gcm_single(
                torch.as_tensor(Z, dtype=torch.float64),
                torch.as_tensor(X, dtype=torch.float64),
                torch.as_tensor(Y, dtype=torch.float64),
                to_numpy=False,
                **gcm_kwargs,
            )
        elif test_lower == "hrt":
            p_val = hrt_single(
                Z,
                X,
                Y,
                **hrt_kwargs,
            )
        else:
            raise ValueError(f"Unsupported test '{test}' for single-experiment evaluation")

        p_tensor = torch.as_tensor(p_val, dtype=torch.float64)
        batch_rate = (p_tensor <= cutoff).double().mean().item()
        total += batch_rate

    return total / float(num_batches)


# Data generation functions
_DATASET_CACHE = {}


def make_X(n, m, distribution="normal", gamma=0, noise="normal"):
    """
    Generate synthetic feature matrix X with various distributions and noise types.

    Args:
        n: Number of samples
        m: Number of features
        distribution: Type of distribution ('normal', 'correlated', 'laplace', 'gdsc')
        gamma: Correlation parameter for normal distributions
        noise: Noise type ('normal', 'laplace')

    Returns:
        X matrix of shape (n, m)
    """
    def draw_noise(shape):
        """Draw noise samples normalized to zero mean and unit variance."""
        noise_type = (noise or "normal").lower()

        if noise_type == "normal":
            samples = np.random.normal(size=shape)
        elif noise_type == "laplace":
            samples = np.random.laplace(loc=0.0, scale=1.0 / np.sqrt(2), size=shape)
        else:
            raise ValueError(f"Unsupported noise type '{noise_type}'.")
        return samples

    distribution = (distribution or "normal").lower()

    if distribution == "normal":
        shared = draw_noise((n, 1))
        eps = draw_noise((n, m))
        X = gamma * shared + np.sqrt(max(1.0 - gamma**2, 0.0)) * eps

    elif distribution == "correlated":
        return make_X(n, m, distribution="normal", gamma=0.5, noise=noise)

    elif distribution == "laplace":
        return make_X(n, m, distribution="normal", gamma=gamma, noise="laplace")

    elif distribution == "gdsc":
        from eccit.experiments.semi import read_gdsc

        if "gdsc" not in _DATASET_CACHE:
            _DATASET_CACHE["gdsc"] = read_gdsc()
        X_full = _DATASET_CACHE["gdsc"]
        total_n, total_m = X_full.shape

        row_idx = np.random.choice(total_n, size=n, replace=n > total_n)
        col_count = min(m, total_m)
        col_idx = np.random.choice(total_m, size=col_count, replace=False)
        X = X_full[row_idx][:, col_idx]

        if m > total_m:
            repeats = int(np.ceil(m / total_m))
            tiled = np.tile(X_full[row_idx], (1, repeats))
            X = tiled[:, :m]

    elif distribution in {"california", "california_housing"}:
        # 20640 x 8
        data = fetch_california_housing(as_frame=False)
        X_full = np.asarray(data.data, dtype=float)
        total_n, total_m = X_full.shape
        row_idx = np.random.choice(total_n, size=n, replace=n > total_n)
        col_count = min(m, total_m)
        col_idx = np.random.choice(total_m, size=col_count, replace=False)
        X = X_full[row_idx][:, col_idx]

        if m > total_m:
            repeats = int(np.ceil(m / total_m))
            tiled = np.tile(X_full[row_idx], (1, repeats))
            X = tiled[:, :m]

    elif distribution in {"cancer", "breast_cancer"}:
        # 569 x 30
        data = load_breast_cancer()
        X_full = np.asarray(data.data, dtype=float)
        total_n, total_m = X_full.shape
        row_idx = np.random.choice(total_n, size=n, replace=n > total_n)
        col_count = min(m, total_m)
        col_idx = np.random.choice(total_m, size=col_count, replace=False)
        X = X_full[row_idx][:, col_idx]

        if m > total_m:
            repeats = int(np.ceil(m / total_m))
            tiled = np.tile(X_full[row_idx], (1, repeats))
            X = tiled[:, :m]

    elif distribution == "wine":
        # 178 x 13
        data = load_wine()
        X_full = np.asarray(data.data, dtype=float)
        total_n, total_m = X_full.shape
        row_idx = np.random.choice(total_n, size=n, replace=n > total_n)
        col_count = min(m, total_m)
        col_idx = np.random.choice(total_m, size=col_count, replace=False)
        X = X_full[row_idx][:, col_idx]

        if m > total_m:
            repeats = int(np.ceil(m / total_m))
            tiled = np.tile(X_full[row_idx], (1, repeats))
            X = tiled[:, :m]

    else:
        shared = draw_noise((n, 1))
        eps = draw_noise((n, m))
        X = gamma * shared + np.sqrt(max(1.0 - gamma**2, 0.0)) * eps

    return X


def make_Y(
    X,
    feat_size,
    order=1,
    *,
    noise_scale=1.0,   # noise std
    main_gain=2.0,     # boost for linear parts
    nl_gain=3.0,       # boost for nonlinear part
    nl_slope=1.25,     # tanh slope (bigger => stronger nonlinearity)
    min_coef=1.0       # floor so weights are not tiny: |N(0,1)| + min_coef
):
    """
    Simpler Y generator:
      - order=1: linear only, stronger weights
      - order>=2: 'Liang-style' blocks with an easy tanh on a single feature,
                  stronger weights, and NO sqrt normalization.

    Returns:
      Y_sim: (n,)
      sel  : sorted indices of active features
    """
    n, m = X.shape
    eps = np.random.randn(n)
    k = int(min(max(feat_size, 0), m))
    if k == 0:
        eps = np.random.randn(n) * noise_scale
        return eps.copy(), np.array([], dtype=int)

    # --------------------
    # Order 1: pure linear
    # --------------------
    if order == 1:
        sel = np.random.choice(m, feat_size, replace=False)
        coeff = np.random.randn(feat_size)
        Y_sim = X[:, sel].dot(coeff) + eps
        return Y_sim, sel

    # ---------------------------------------------------
    # Order >= 2: easy tanh + stronger linear weights
    #   * Blocks of 4: [i0, i1, i2, i3]; use i0,i1 linearly; i2 drives tanh
    #   * i3 is unused on purpose to keep it simple
    #   * No sqrt scaling anywhere
    # ---------------------------------------------------
    sel_block = np.random.choice(m, k, replace=False)
    g = k // 4
    core = sel_block[: 4 * g].reshape(g, 4) if g > 0 else np.empty((0, 4), dtype=int)
    leftover_idx = sel_block[4 * g :]

    linear_part = 0.0
    nonlinear_part = 0.0

    if g > 0:
        # Stronger linear weights (with random signs)
        w0 = (np.abs(np.random.randn(g)) + min_coef) * np.random.choice([-1.0, 1.0], size=g)
        w1 = (np.abs(np.random.randn(g)) + min_coef) * np.random.choice([-1.0, 1.0], size=g)
        # Positive amplitude for tanh outputs
        w2 = np.abs(np.random.randn(g)) + min_coef

        i0 = core[:, 0]
        i1 = core[:, 1]
        i2 = core[:, 2]

        # Linear: boosted, no normalization
        if i0.size:
            linear_part += main_gain * X[:, i0].dot(w0)
        if i1.size:
            linear_part += main_gain * X[:, i1].dot(w1)

        # Easy nonlinear: tanh on a single feature per block
        if i2.size:
            nl = np.tanh(nl_slope * X[:, i2])  # shape (n, g)
            nonlinear_part += nl_gain * nl.dot(w2)

    # Leftovers (if k not multiple of 4): small linear contribution so they count as active
    leftover_part = 0.0
    if leftover_idx.size > 0:
        w_left = (np.abs(np.random.randn(leftover_idx.size)) + 0.5) * np.random.choice([-1.0, 1.0], size=leftover_idx.size)
        leftover_part = 0.5 * main_gain * X[:, leftover_idx].dot(w_left)

    eps = np.random.randn(n) * noise_scale
    Y_sim = linear_part + nonlinear_part + leftover_part + eps
    return Y_sim, np.sort(sel_block)





# Performance evaluation functions
def eval_performance(p_raw, selected, alpha=0.2, alpha_adjust=None):
    """
    Evaluate FDR, power, and valid power for given p-values and ground truth.

    Args:
        p_raw: Raw p-values (array)
        selected: Indices or boolean mask of true non-null hypotheses
        alpha: Nominal FDR level
        alpha_adjust: Optional alpha adjustment function

    Returns:
        valid_power: Power when FDR <= alpha, otherwise 0
        power: Statistical power
        fdr: False discovery rate
    """
    p = np.asarray(p_raw, dtype=float)

    # Handle selected as either boolean mask or indices
    if np.issubdtype(np.asarray(selected).dtype, np.bool_):
        indices = np.flatnonzero(selected)
    else:
        indices = np.asarray(selected, dtype=int)

    alpha_eff = alpha_adjust(alpha) if alpha_adjust is not None else alpha
    rejected, _, _, _ = multipletests(p, alpha=alpha_eff, method='fdr_bh')

    true = np.zeros_like(p, dtype=bool)
    true[indices] = True
    disc = np.where(rejected)[0]

    fdr = (np.sum(~true[disc]) / len(disc)) if len(disc) else 0.0
    power = np.sum(true & rejected) / np.sum(true) if np.sum(true) else 0.0
    valid_power = power if fdr <= alpha else 0.0

    return float(valid_power), float(power), float(fdr)


def compute_power_stats(p_values, true_nonnull, alpha=0.2, alpha_adjust=None):
    """
    Compute power statistics (convenience wrapper for eval_performance).

    Returns:
        valid_power: Power when FDR <= alpha, otherwise 0
        power: Statistical power
        fdr: False discovery rate
    """
    return eval_performance(p_values, true_nonnull, alpha=alpha, alpha_adjust=alpha_adjust)


def summarize(pvals_lists, sel_feat_list, alphas, n_responses, alpha_adjust=None):
    """
    Summarize FDR, power, and valid power across multiple runs.

    Returns means and standard errors for FDR, power, and valid power at different alpha levels.
    """
    means, ses = [], []
    powers, power_ses = [], []
    valid_powers, valid_power_ses = [], []

    for alpha in alphas:
        fdrs, pows, valid_pows = [], [], []
        for pvals, sel in zip(pvals_lists, sel_feat_list):
            valid_power, power, fdr = eval_performance(pvals, sel, alpha=alpha, alpha_adjust=alpha_adjust)
            fdrs.append(fdr)
            pows.append(power)
            valid_pows.append(valid_power)

        means.append(np.mean(fdrs))
        ses.append(np.std(fdrs)/np.sqrt(n_responses))
        powers.append(np.mean(pows))
        power_ses.append(np.std(pows)/np.sqrt(n_responses))
        valid_powers.append(np.mean(valid_pows))
        valid_power_ses.append(np.std(valid_pows)/np.sqrt(n_responses))

    return (np.array(means), np.array(ses),
            np.array(powers), np.array(power_ses),
            np.array(valid_powers), np.array(valid_power_ses))


def make_mean_sem(p_lists, num_runs, grid):
    """Compute mean and standard error for empirical CDFs."""
    ecdf = np.array([[np.mean(p <= u) for u in grid] for p in p_lists])
    return ecdf.mean(axis=0), ecdf.std(axis=0)/np.sqrt(num_runs)


def make_alpha_adjuster(grid, cdf):
    """
    Create function to adjust alpha levels based on calibration.
    
    Interpolates the calibration CDF to provide corrected alpha levels.
    """
    g = np.asarray(grid, dtype=float)
    F = np.maximum.accumulate(np.clip(np.asarray(cdf, dtype=float), 0.0, 1.0))
    return (lambda a: float(np.interp(a, F, g, left=g[0], right=g[-1])))


def make_alpha_adjuster_from_fdp(alpha_grid, fdp_curve):
    """
    Create an alpha adjustment function from an FDP mapping.

    Given samples of f(alpha) ≈ E[FDP | BH at alpha] over `alpha_grid`,
    return a function that maps desired FDP q to an input alpha via inverse
    interpolation over the (monotone) FDP curve.
    """
    g = np.asarray(alpha_grid, dtype=float)
    f = np.asarray(fdp_curve, dtype=float)
    # Enforce bounds and monotonicity for a stable inverse
    f = np.maximum.accumulate(np.clip(f, 0.0, 1.0))
    def adjust(q):
        q_arr = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
        interp = np.interp(q_arr, f, g, left=g[0], right=g[-1])
        calibrated = np.minimum(interp, q_arr)
        if np.isscalar(q):
            return float(calibrated)
        return calibrated
    return adjust


def summarize_metrics(label, valid_vals, raw_vals, fdr_vals):
    """
    Print summary statistics for valid power, power, and FDR.

    Args:
        label: Label for the method
        valid_vals: Iterable of valid power values
        raw_vals: Iterable of raw power values
        fdr_vals: Iterable of FDR values
    """
    valid_arr = np.asarray(list(valid_vals), dtype=float)
    raw_arr = np.asarray(list(raw_vals), dtype=float)
    fdr_arr = np.asarray(list(fdr_vals), dtype=float)

    def summary(arr):
        if arr.size == 0:
            return 0.0, 0.0
        mean = float(arr.mean())
        if arr.size > 1:
            sample_std = float(arr.std(ddof=1))
        else:
            sample_std = 0.0
        ci = (sample_std / 10.0)
        return mean, ci

    v_mean, v_sem = summary(valid_arr)
    r_mean, r_sem = summary(raw_arr)
    f_mean, f_sem = summary(fdr_arr)
    print(
        f"{label:<22s} power = {r_mean:.3f} (+/- {r_sem:.3f}) | "
        f"valid power = {v_mean:.3f} (+/- {v_sem:.3f}) | "
        f"fdr = {f_mean:.3f} (+/- {f_sem:.3f})"
    )
