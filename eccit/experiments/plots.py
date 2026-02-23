import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Sequence, Union

FDR_COLOR_RAW = "#1b4a9b"
POWER_COLOR_RAW = "#ff7300"
FDR_COLOR_CAL = "#1b3b9b"  # deeper blue for calibrated FDR
POWER_COLOR_CAL = '#ff5100'
RAW_LINE_ALPHA = 0.75
CAL_LINE_ALPHA = 1.0
RAW_FILL_ALPHA = 0.10
CAL_FILL_ALPHA = 0.30


def plot_cdf_sweep(results, n_list=tuple(range(25, 501, 25)), m_list=(10,25,50),
                   dist_list=("normal", "correlated", "laplace"), out_dir="outputs"):
    """Plot calibration curves: FDP mapping or realized Type-I vs nominal alpha."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(n_list), len(m_list),
                             figsize=(3*len(m_list), 3*len(n_list)),
                             squeeze=False, sharex=True, sharey=True)
    exemplar = next(iter(results.values())) if results else {}
    is_fdp = ('fdp_mean' in exemplar) and ('alpha_grid' in exemplar)
    type1_alpha_grid = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])

    available_keys = set(results.keys())

    for i, n in enumerate(n_list):
        for j, m in enumerate(m_list):
            ax = axes[i][j]
            has_data = False
            max_alpha = 1.0
            if is_fdp:
                alphas_here = [results[(d, n, m)]['alpha_grid'].max()
                               for d in dist_list
                               if (d, n, m) in available_keys]
                if alphas_here:
                    max_alpha = max(alphas_here)
            else:
                max_alpha = type1_alpha_grid.max()

            for dist in dist_list:
                key = (dist, n, m)
                if key not in available_keys:
                    continue
                r = results[key]
                has_data = True
                if is_fdp:
                    ax.plot(r['alpha_grid'], r['fdp_mean'], lw=1.5,
                            label=dist if (i==0 and j==0) else None)
                else:
                    grid = r.get('grid')
                    joint_cdf = r.get('joint_cdf')
                    if grid is None or joint_cdf is None:
                        continue
                    type1_curve = np.interp(type1_alpha_grid, grid, joint_cdf, 0.0, 1.0)
                    ax.plot(type1_alpha_grid, type1_curve, lw=1.5,
                            label=dist if (i==0 and j==0) else None)
            
            # Reference diagonal and axis scaling
            if has_data:
                ax.plot([0, max_alpha], [0, min(1.0, max_alpha)], 'k--', lw=1)
                ax.set_xlim(0, max_alpha)
                ax.set_ylim(0, 1)
                if is_fdp:
                    ax.set_xticks(np.linspace(0, max_alpha, 6))
                else:
                    ax.set_xticks(type1_alpha_grid)
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.grid(True, alpha=0.3)

                # Always show tick labels and numbers
                ax.tick_params(labelsize=8)
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks([]); ax.set_yticks([])
                ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

            if i==len(n_list)-1 and has_data:
                ax.set_xlabel('Nominal FDR (alpha)' if is_fdp else 'Nominal Type-I (alpha)')
            if j==0 and has_data:
                ax.set_ylabel(f'n={n}\nRealized FDP' if is_fdp else f'n={n}\nRealized Type-I')
            if i==0:
                ax.set_title(f"m={m}")

    legend_handles, legend_labels = [], []
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend_handles, legend_labels = handles, labels
            break
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='lower center',
                   ncols=len(dist_list), bbox_to_anchor=(0.5,-0.02), fontsize=10)
    fig.suptitle("Nominal vs Realized FDP" if ('fdp_mean' in exemplar) else "Nominal vs Realized Type-I", y=0.99)
    fig.tight_layout(rect=[0,0.05,1,0.93])
    out_name = "fdp_sweep.pdf" if ('fdp_mean' in exemplar) else "type1_sweep.pdf"
    fig.savefig(out_dir/out_name, bbox_inches='tight')
    plt.close(fig)


def plot_single_performance(
    dataset: str,
    sweep_rows: Sequence[dict],
    out_dir: Union[str, Path] = "outputs/singles",
    *,
    test_name: Optional[str] = None,
) -> None:
    """Plot type-I error and power before/after calibration for single experiments."""

    if not sweep_rows:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = sorted(sweep_rows, key=lambda row: row['alpha'])
    alphas = np.array([row['alpha'] for row in data])

    type1_raw = np.array([row['type1_raw'] for row in data])
    type1_raw_se = np.array([row['type1_raw_se'] for row in data])
    type1_cal = np.array([row['type1_cal'] for row in data])
    type1_cal_se = np.array([row['type1_cal_se'] for row in data])

    power_raw = np.array([row['power_raw'] for row in data])
    power_raw_se = np.array([row['power_raw_se'] for row in data])
    power_cal = np.array([row['power_cal'] for row in data])
    power_cal_se = np.array([row['power_cal_se'] for row in data])

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Raw curves with lighter opacity
    ax.plot(alphas, type1_raw, color=FDR_COLOR_RAW, marker='o', lw=1.2, alpha=RAW_LINE_ALPHA, label='Type-I raw')
    ax.fill_between(alphas, type1_raw - type1_raw_se, type1_raw + type1_raw_se, color=FDR_COLOR_RAW, alpha=RAW_FILL_ALPHA)
    ax.plot(alphas, power_raw, color=POWER_COLOR_RAW, marker='s', lw=1.2, alpha=RAW_LINE_ALPHA, label='Power raw')
    ax.fill_between(alphas, power_raw - power_raw_se, power_raw + power_raw_se, color=POWER_COLOR_RAW, alpha=RAW_FILL_ALPHA)

    # Calibrated curves with stronger opacity
    ax.plot(alphas, type1_cal, color=FDR_COLOR_CAL, marker='o', lw=1.6, alpha=CAL_LINE_ALPHA, label='Type-I calibrated')
    ax.fill_between(alphas, type1_cal - type1_cal_se, type1_cal + type1_cal_se, color=FDR_COLOR_CAL, alpha=CAL_FILL_ALPHA)
    ax.plot(alphas, power_cal, color=POWER_COLOR_CAL, marker='s', ls='--', lw=1.6, alpha=CAL_LINE_ALPHA, label='Power calibrated')
    ax.fill_between(alphas, power_cal - power_cal_se, power_cal + power_cal_se, color=POWER_COLOR_CAL, alpha=CAL_FILL_ALPHA)

    # Reference line for nominal Type-I
    ax.plot(alphas, alphas, ':', color='red', lw=1.0, label='Target Type-I')

    ax.set_xlim(alphas.min(), alphas.max())
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(alphas)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    ax.set_xlabel('Nominal alpha', fontsize=10)
    ax.set_ylabel('Type-I + Power', fontsize=10)
    test_label = (test_name or '').strip()
    display_test = test_label.upper() if test_label else None
    title = f"Single experiment performance ({dataset}, {display_test})"
    ax.set_title(title, fontsize=11)
    ax.legend(loc='best', fontsize=7)

    fig.tight_layout()
    safe_name = dataset.replace(' ', '_').lower()
    fn_base = f"single_{safe_name}"
    if display_test:
        safe_test = display_test.replace(' ', '_').lower()
        fn_base += f"_{safe_test}"
    out_path_pdf = out_dir / f"{fn_base}.pdf"
    out_path_png = out_dir / f"{fn_base}.png"
    fig.savefig(out_path_pdf, bbox_inches='tight')
    fig.savefig(out_path_png, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_perf_sweep(perf_results, n_list=tuple(range(25, 501, 25)), m_list=(10,25,50),
                    dist_list=("normal", "correlated", "laplace"), out_dir="outputs",
                    metric="fdp"):
    """Plot performance results from sweep experiment."""
    if metric == "area":
        metric = "type1"
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Filter n_list to only plot specific values
    plot_n_values = [25, 50, 100, 200, 500]
    filtered_n_list = [n for n in n_list if n in plot_n_values]

    # If no matching values, fall back to original list
    if not filtered_n_list:
        filtered_n_list = list(n_list)

    for dist in dist_list:
        fig, axes = plt.subplots(len(filtered_n_list), len(m_list),
                                 figsize=(3.4*len(m_list), 3.4*len(filtered_n_list)),
                                 squeeze=False, sharex=True, sharey=True)

        for i, n in enumerate(filtered_n_list):
            for j, m in enumerate(m_list):
                ax = axes[i][j]
                key = (dist, n, m)
                if key not in perf_results:
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
                    if i == 0: ax.set_title(f'm={m}')
                    continue

                r = perf_results[key]
                alphas = r['alphas']

                # Raw p-values (lighter opacity)
                ax.plot(alphas, r['fdr_raw'], color=FDR_COLOR_RAW, marker='o', lw=1.1, alpha=RAW_LINE_ALPHA,
                        label='Raw FDR' if (i==0 and j==0) else None)
                ax.fill_between(alphas, r['fdr_raw']-r['fdr_raw_se'],
                                r['fdr_raw']+r['fdr_raw_se'], color=FDR_COLOR_RAW, alpha=RAW_FILL_ALPHA)
                ax.plot(alphas, r['pow_raw'], color=POWER_COLOR_RAW, marker='s', lw=1.1, alpha=RAW_LINE_ALPHA,
                        label='Raw Power' if (i==0 and j==0) else None)
                ax.fill_between(alphas, r['pow_raw']-r['pow_raw_se'],
                                r['pow_raw']+r['pow_raw_se'], color=POWER_COLOR_RAW, alpha=RAW_FILL_ALPHA)

                # Calibrated p-values (stronger opacity)
                ax.plot(alphas, r['fdr_cal'], color=FDR_COLOR_CAL, marker='o', lw=1.6, alpha=CAL_LINE_ALPHA,
                        label='Cal FDR' if (i==0 and j==0) else None)
                ax.fill_between(alphas, r['fdr_cal']-r['fdr_cal_se'],
                                r['fdr_cal']+r['fdr_cal_se'], color=FDR_COLOR_CAL, alpha=CAL_FILL_ALPHA)
                ax.plot(alphas, r['pow_cal'], color=POWER_COLOR_CAL, marker='s', ls='--', lw=1.6, alpha=CAL_LINE_ALPHA,
                        label='Cal Power' if (i==0 and j==0) else None)
                ax.fill_between(alphas, r['pow_cal']-r['pow_cal_se'],
                                r['pow_cal']+r['pow_cal_se'], color=POWER_COLOR_CAL, alpha=CAL_FILL_ALPHA)

                # Target FDR
                ax.plot(alphas, alphas, ':r', lw=1)

                ax.set_xlim(0, alphas.max()); ax.set_ylim(0, 1)
                ax.set_xticks(alphas)
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.grid(True, alpha=0.3)

                # Always show tick labels and numbers
                ax.tick_params(labelsize=8)

                if i == len(filtered_n_list)-1: ax.set_xlabel('Nominal FDR (alpha)')
                if j == 0: ax.set_ylabel(f'n={n}\nFDR + Power ({metric})')
                if i == 0: ax.set_title(f'm={m}')

        legend_handles, legend_labels = [], []
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend_handles, legend_labels = handles, labels
                break
        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc='lower center', ncols=4, fontsize=10,
                       bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"Performance for {dist} (pmax=0.2, metric={metric})", y=0.995)
        fig.tight_layout(rect=[0,0.06,1,0.95])
        fig.savefig(out_dir/f"perf_{dist}_{metric}.pdf", bbox_inches='tight')
        plt.close(fig)


def plot_calibration_offset(results, n_list=tuple(range(25, 501, 25)), m_list=(10,25,50),
                            dist_list=("normal", "correlated", "laplace"),
                            out_dir="outputs", metric="fdp", target_alpha=0.2):
    """Plot calibration offset vs. sample size for each feature dimension."""
    if metric == "area":
        metric = "type1"
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    for dist in dist_list:
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        seen_ns = []
        max_offset = None
        for idx, m in enumerate(m_list):
            ns = []
            offsets = []
            for n in n_list:
                key = (dist, n, m)
                if key not in results:
                    continue
                res = results[key]
                if metric == "fdp":
                    alpha_grid = res['alpha_grid']
                    fdp_mean = res['fdp_mean']
                    if alpha_grid.size == 0:
                        continue
                    offset = np.interp(target_alpha, alpha_grid, fdp_mean)
                else:
                    grid = res.get('grid')
                    joint_cdf = res.get('joint_cdf')
                    if grid is None or joint_cdf is None:
                        continue
                    # Nominal alpha fixes the cutoff; realized Type-I is read off the joint CDF.
                    offset = float(np.interp(target_alpha, grid, joint_cdf, 0.0, 1.0))
                if offset is None:
                    continue
                ns.append(n)
                offsets.append(offset)

            if not ns:
                continue
            order = np.argsort(ns)
            ns_sorted = np.array(ns)[order]
            offsets_sorted = np.array(offsets)[order]
            ax.plot(ns_sorted, offsets_sorted, marker='o', lw=1.6, label=f"m={m}")
            seen_ns.extend(ns_sorted.tolist())
            max_val = float(np.max(offsets_sorted)) if offsets_sorted.size else None
            if max_val is not None:
                max_offset = max_val if max_offset is None else max(max_offset, max_val)

        if not seen_ns:
            plt.close(fig)
            continue

        ax.set_xlabel('Sample size n (log10 scale)', fontsize=10)  # updated to reflect log scale
        if metric == "fdp":
            ax.set_ylabel(f'Realized FDP @ α={target_alpha:.2f}', fontsize=10)
            ax.set_ylim(0, 0.6)
        else:
            ax.set_ylabel(f'Realized Type-I @ α={target_alpha:.2f}', fontsize=10)
            ref_offset = max_offset if max_offset is not None else target_alpha
            y_max = min(1.0, max(ref_offset * 1.1, target_alpha * 1.1))
            ax.set_ylim(0, y_max)

        ax.set_title(f'{dist}', fontsize=11)
       
        ax.legend(loc='best', fontsize=12)
        # Use log scale for x-axis to avoid cramping
        ax.set_xscale('log', base=10)
        ax.set_xlim(min(n_list), max(n_list))  # keeps requested ticks within view
        ax.set_xticks([25, 100, 500])
        ax.set_xticklabels(['25', '100', '500'])

        ax.axhline(target_alpha, color='r', linestyle='--', linewidth=1.0,
                    label=f"α={target_alpha:.2f}")
        fig.tight_layout()
        fig.savefig(out_dir/f"offset_{dist}_{metric}.pdf", bbox_inches='tight')
        plt.close(fig)


def plot_second_order_calibration(cal_results, output_dir, distribution=None, test="gcm"):
    """Plot calibration: FDP mapping or realized Type-I vs nominal alpha."""
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(6,6))
    # Inspect one to decide which plot
    any_item = next(iter(cal_results.values()))
    is_fdp = ('fdp_mean' in any_item) and ('alpha_grid' in any_item)
    type1_alpha_grid = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])

    order_names = {1: "Linear", 2: "Nonlinear"}
    test_label = test.upper()

    # Define consistent color palette for 4 combinations
    colors = {
        (1, 1): '#1f77b4',  # Linear adv, Linear test - blue
        (1, 2): '#ff7f0e',  # Linear adv, Nonlinear test - orange
        (2, 1): '#2ca02c',  # Nonlinear adv, Linear test - green
        (2, 2): '#d62728',  # Nonlinear adv, Nonlinear test - red
    }

    for (oa, ot), res in cal_results.items():
        color = colors.get((oa, ot), 'black')
        if is_fdp:
            ax.plot(res['alpha_grid'], res['fdp_mean'], lw=1.5, color=color,
                    label=f"{order_names[oa]} adversary, {order_names[ot]} {test_label}")
        else:
            grid = res.get('grid')
            joint_cdf = res.get('joint_cdf')
            if grid is None or joint_cdf is None:
                continue
            type1_curve = np.interp(type1_alpha_grid, grid, joint_cdf, 0.0, 1.0)
            ax.plot(type1_alpha_grid, type1_curve, lw=1.5, color=color,
                    label=f"{order_names[oa]} adversary, {order_names[ot]} {test_label}")

    # Reference diagonal and axis scaling
    if is_fdp:
        # Determine global max alpha
        max_alpha = max(res['alpha_grid'].max() for res in cal_results.values())
    else:
        max_alpha = type1_alpha_grid.max()
    ax.plot([0, max_alpha], [0, min(1.0, max_alpha)], 'k--', lw=1)
    ax.set_xlim(0, max_alpha); ax.set_ylim(0, 1)
    if is_fdp:
        ax.set_xticks(np.linspace(0, max_alpha, 6))
    else:
        ax.set_xticks(type1_alpha_grid)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('Nominal FDR (alpha)' if is_fdp else 'Nominal Type-I (alpha)', fontsize=11)
    ax.set_ylabel('Realized FDP' if is_fdp else 'Realized Type-I', fontsize=11)
    ax.legend(loc='best', fontsize=12)
    title = "Nominal vs Realized FDP" if is_fdp else "Nominal vs Realized Type-I"
    if distribution:
        title += f" | dist={distribution}"
    fig.suptitle(title, y=0.99, fontsize=12)
    fig.tight_layout()
    out_name = "calibration_fdp_all.pdf" if is_fdp else "calibration_type1_all.pdf"
    fig.savefig(output_dir/out_name, bbox_inches='tight')
    plt.close(fig)


def plot_second_order_performance(perf_results, output_dir, metric="fdp", distribution=None, test="gcm"):
    """Plot performance results for second-order experiment."""
    if metric == "area":
        metric = "type1"
    output_dir = Path(output_dir)
    orders = [1, 2]
    order_names = {1: "Linear", 2: "Nonlinear"}
    
    for ot in orders:
        fig, axes = plt.subplots(2, 2, figsize=(3.4*2, 3.4*2),
                                 squeeze=False, sharex=True, sharey=True)
        for i, oa in enumerate(orders):
            for j, truth in enumerate(orders):
                ax = axes[i][j]
                key_perf = (ot, oa, truth)
                if key_perf not in perf_results:
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
                    ax.set_title(f"{order_names[truth]} truth")
                    continue
                r = perf_results[key_perf]
                a = r['alphas']
                
                # Raw FDR & Power (lighter opacity)
                ax.plot(a, r['fdr_raw'], color=FDR_COLOR_RAW, marker='o', lw=1.1, alpha=RAW_LINE_ALPHA,
                        label='Raw FDR' if (i==0 and j==0) else None)
                ax.fill_between(a, r['fdr_raw']-r['fdr_raw_se'],
                                r['fdr_raw']+r['fdr_raw_se'], color=FDR_COLOR_RAW, alpha=RAW_FILL_ALPHA)
                ax.plot(a, r['pow_raw'], color=POWER_COLOR_RAW, marker='s', lw=1.1, alpha=RAW_LINE_ALPHA,
                        label='Raw Power' if (i==0 and j==0) else None)
                ax.fill_between(a, r['pow_raw']-r['pow_raw_se'],
                                r['pow_raw']+r['pow_raw_se'], color=POWER_COLOR_RAW, alpha=RAW_FILL_ALPHA)
                                
                # Calibrated FDR & Power (stronger opacity)
                ax.plot(a, r['fdr_cal'], color=FDR_COLOR_CAL, marker='o', lw=1.6, alpha=CAL_LINE_ALPHA,
                        label='Cal FDR' if (i==0 and j==0) else None)
                ax.fill_between(a, r['fdr_cal']-r['fdr_cal_se'],
                                r['fdr_cal']+r['fdr_cal_se'], color=FDR_COLOR_CAL, alpha=CAL_FILL_ALPHA)
                ax.plot(a, r['pow_cal'], color=POWER_COLOR_CAL, marker='s', ls='--', lw=1.6, alpha=CAL_LINE_ALPHA,
                        label='Cal Power' if (i==0 and j==0) else None)
                ax.fill_between(a, r['pow_cal']-r['pow_cal_se'],
                                r['pow_cal']+r['pow_cal_se'], color=POWER_COLOR_CAL, alpha=CAL_FILL_ALPHA)
                                
                # Target FDR
                ax.plot(a, a, ':r', lw=1)
                ax.set_xlim(0, a.max()); ax.set_ylim(0,1)
                ax.set_xticks(a)
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.grid(True, alpha=0.3)

                # Always show tick labels and numbers
                ax.tick_params(labelsize=8)

                if i == 1:
                    ax.set_xlabel('Nominal FDR (alpha)')
                if j == 0:
                    ax.set_ylabel(f'{order_names[oa]} adversary\nFDR + Power ({metric})')
                ax.set_title(f"{order_names[truth]} truth")
                
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncols=4, fontsize=12,
                   bbox_to_anchor=(0.5, -0.04))
        test_label = test.upper()
        title = f"Performance ({order_names[ot]} {test_label}, metric={metric})"
        if distribution:
            title += f" | dist={distribution}"
        fig.suptitle(title, y=0.995, fontsize=12)
        fig.tight_layout(rect=[0,0.06,1,0.95])
        fig.savefig(output_dir/f"perf_order_{ot}_{metric}.pdf", bbox_inches='tight')
        plt.close(fig)


def plot_mask_ranking(df_masks, final_mask=None, top_k=20, bottom_k=20,
                      out_top="mask_ranking_top.pdf",
                      out_bottom="mask_ranking_worst.pdf", metric="type1"):
    """Plot top and bottom ranked masks."""
    if metric == "area":
        metric = "type1"
    if 'type1_mean' not in df_masks.columns and 'area_mean' in df_masks.columns:
        df_masks = df_masks.rename(columns={'area_mean': 'type1_mean', 'area_sem': 'type1_sem'})
    Path(out_top).parent.mkdir(parents=True, exist_ok=True)

    # Top K masks
    top = df_masks.head(top_k).copy()
    fig, ax = plt.subplots(figsize=(min(12, 0.5*top_k + 4), 0.45*top_k + 2))
    y = np.arange(len(top))
    bars = ax.barh(y, top['type1_mean'].values, xerr=top['type1_sem'].values, align='center')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.rank:>2d} | {r.mask_bits} (k={r.k_active})" 
                        for r in top.itertuples()])
    ax.invert_yaxis()
    metric_label = "FDP" if metric == "fdp" else "Type-I Error"
    ax.set_xlabel(f"Mean {metric_label} on nulls")
    ax.set_title(f"Top-{top_k} masks ({metric})")
    ax.grid(axis='x', alpha=0.25)


    fig.tight_layout()
    # Add metric to filename
    out_top_with_metric = str(out_top).replace(".pdf", f"_{metric}.pdf")
    fig.savefig(out_top_with_metric, bbox_inches='tight')
    plt.close(fig)

    # Worst K masks
    worst = df_masks.tail(bottom_k).copy()
    fig, ax = plt.subplots(figsize=(min(12, 0.5*bottom_k + 4), 0.45*bottom_k + 2))
    y = np.arange(len(worst))
    bars = ax.barh(y, worst['type1_mean'].values, xerr=worst['type1_sem'].values, align='center')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.rank:>2d} | {r.mask_bits} (k={r.k_active})"
                        for r in worst.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean {metric_label} on nulls")
    ax.set_title(f"Worst-{bottom_k} masks ({metric})")
    ax.grid(axis='x', alpha=0.25)


    fig.tight_layout()
    # Add metric to filename
    out_bottom_with_metric = str(out_bottom).replace(".pdf", f"_{metric}.pdf")
    fig.savefig(out_bottom_with_metric, bbox_inches='tight')
    plt.close(fig)


def plot_sampled_mask_ranks(samples_df, out_path="sampled_mask_ranks.pdf", max_bins=30, metric="type1"):
    """Plot histogram of mask ranks from sampling."""
    if metric == "area":
        metric = "type1"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    ranks = samples_df['rank'].astype(int).values
    probs = samples_df['prob_est'].values

    # Create bins
    max_rank = int(ranks.max())
    num_bins = min(max_bins, max_rank)
    edges = np.linspace(1, max_rank + 1, num_bins + 1, dtype=int)

    bin_mass = []
    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (ranks >= a) & (ranks < b)
        mass = probs[sel].sum()
        bin_mass.append(mass)
        labels.append(f"{a}-{b-1}")

    fig, ax = plt.subplots(figsize=(min(14, 0.5*num_bins + 4), 4))
    xs = np.arange(num_bins)
    bars = ax.bar(xs, bin_mass, color='C0')

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Probability mass")
    ax.set_xlabel("Rank bucket")
    ax.set_title(f"Distribution over mask ranks ({metric})")
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    # Add metric to filename
    out_path_with_metric = str(out_path).replace(".pdf", f"_{metric}.pdf")
    fig.savefig(out_path_with_metric, bbox_inches='tight')
    plt.close(fig)


def plot_mask_probabilities(final_probs, out_path="mask_probs.pdf", metric="type1"):
    """Plot final mask inclusion probabilities."""
    if metric == "area":
        metric = "type1"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    p = len(final_probs)
    fig, ax = plt.subplots(figsize=(max(6, 0.6*p + 2), 3.2))
    xs = np.arange(p)
    ax.bar(xs, final_probs, color='C0')
    ax.set_ylim(0,1)
    ax.set_xticks(xs); ax.set_xlabel("Feature index")
    ax.set_ylabel("Inclusion prob")
    ax.set_title(f"Final mask probabilities ({metric})")
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    # Add metric to filename
    out_path_with_metric = str(out_path).replace(".pdf", f"_{metric}.pdf")
    fig.savefig(out_path_with_metric, bbox_inches='tight')
    plt.close(fig)


def plot_mask_prob_trajectories(mask_prob_hist, out_path="mask_prob_trajectories.pdf", metric="type1"):
    """Plot evolution of mask probabilities during training."""
    if metric == "area":
        metric = "type1"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    H = np.asarray(mask_prob_hist)  # [epochs, p]
    T, p = H.shape
    fig, ax = plt.subplots(figsize=(max(6, 0.6*p + 2), 3.6))
    t = np.arange(T)
    for j in range(p):
        ax.plot(t, H[:, j], lw=1.2, label=f"j={j}")
    ax.set_ylim(0,1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Inclusion prob")
    ax.set_title(f"Mask probability trajectories ({metric})")
    ax.grid(alpha=0.25)
    if p <= 12: ax.legend(ncol=3, fontsize=10)
    fig.tight_layout()
    # Add metric to filename
    out_path_with_metric = str(out_path).replace(".pdf", f"_{metric}.pdf")
    fig.savefig(out_path_with_metric, bbox_inches='tight')
    plt.close(fig)


def plot_linear_weight_trajectories(weight_history, out_path="y_weights.pdf", metric="type1"):
    """Plot evolution of linear YGenerator weights when order=1."""
    if metric == "area":
        metric = "type1"
    if weight_history is None or len(weight_history) == 0:
        return

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    W = np.asarray(weight_history)  # [epochs, p]
    T, p = W.shape
    epochs = np.arange(T)

    fig, ax = plt.subplots(figsize=(max(6, 0.6*p + 2), 3.2))
    for j in range(p):
        ax.plot(epochs, W[:, j], lw=1.2, label=f"w{j}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight value")
    ax.set_title(f"YGenerator linear weights ({metric})")
    ax.grid(alpha=0.25)
    if p <= 12:
        ax.legend(ncol=3, fontsize=10)

    fig.tight_layout()
    out_path_with_metric = str(out_path).replace(".pdf", f"_{metric}.pdf")
    fig.savefig(out_path_with_metric, bbox_inches='tight')
    plt.close(fig)

def plot_sweep_valid_power_gain_bars(results_dict, n_ref=250, m_ref=25, out_path="valid_power_gain_bars.pdf"):
    """Plot grouped bar chart of valid power gain across distributions for sweep experiments.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys (test, metric) like ('gcm', 'type1'), ('gcm', 'fdp'), ('hrt', 'type1'), ('hrt', 'fdp')
        Values are the final_results from aggregate_sweep_results containing 'performance' dict
    n_ref, m_ref : int
        Reference n and m values to extract from performance dict (default: 250, 25)
    out_path : str or Path
        Output path for the plot
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_results = {}
    for (test, metric), result in results_dict.items():
        metric_key = "type1" if metric == "area" else metric
        normalized_results[(test, metric_key)] = result
    results_dict = normalized_results

    # Define colors for the 4 bars
    colors = {
        ('gcm', 'type1'): '#1f77b4',   # blue
        ('gcm', 'fdp'): '#ff7f0e',     # orange
        ('hrt', 'type1'): '#2ca02c',   # green
        ('hrt', 'fdp'): '#d62728',     # red
    }

    # Labels for legend
    labels = {
        ('gcm', 'type1'): 'GCM (Type-I)',
        ('gcm', 'fdp'): 'GCM (FDP)',
        ('hrt', 'type1'): 'HRT (Type-I)',
        ('hrt', 'fdp'): 'HRT (FDP)',
    }

    # Extract all distributions (should be same across all results)
    all_dists = set()
    for (test, metric), result in results_dict.items():
        performance = result.get('performance', {})
        dists = {dist for dist, n, m in performance.keys() if n == n_ref and m == m_ref}
        all_dists.update(dists)

    distributions = sorted(all_dists)
    if not distributions:
        print(f"No data found for n={n_ref}, m={m_ref}")
        return

    # Prepare data for plotting
    bar_data = {key: {'means': [], 'stds': []} for key in colors.keys()}

    for dist in distributions:
        for (test, metric) in [('gcm', 'type1'), ('gcm', 'fdp'), ('hrt', 'type1'), ('hrt', 'fdp')]:
            if (test, metric) not in results_dict:
                bar_data[(test, metric)]['means'].append(0)
                bar_data[(test, metric)]['stds'].append(0)
                continue

            performance = results_dict[(test, metric)].get('performance', {})
            entry = performance.get((dist, n_ref, m_ref))

            if entry:
                gain_mean = entry.get('valid_power_gain_mean', 0)
                gain_std = entry.get('valid_power_gain_std', 0)
                # Convert to standard error, assuming 100 runs
                gain_se = gain_std / 10.0
                bar_data[(test, metric)]['means'].append(gain_mean)
                bar_data[(test, metric)]['stds'].append(gain_se)
            else:
                bar_data[(test, metric)]['means'].append(0)
                bar_data[(test, metric)]['stds'].append(0)

    # Create the plot
    fig, ax = plt.subplots(figsize=(max(8, 1.5 + len(distributions) * 1.2), 6))

    # Bar width and positions
    bar_width = 0.2
    x = np.arange(len(distributions))
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    # Plot bars for each (test, metric) combination
    for idx, (test, metric) in enumerate([('gcm', 'type1'), ('gcm', 'fdp'), ('hrt', 'type1'), ('hrt', 'fdp')]):
        means = bar_data[(test, metric)]['means']
        stds = bar_data[(test, metric)]['stds']
        ax.bar(x + offsets[idx], means, bar_width,
               label=labels[(test, metric)],
               color=colors[(test, metric)],
               yerr=stds, capsize=3, error_kw={'linewidth': 1})

    # Customize plot
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Distribution', fontsize=12)
    ax.set_ylabel('Valid Power Gain', fontsize=12)
    ax.set_title(f'Valid Power Gain by Distribution (n={n_ref}, m={m_ref})', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(distributions, rotation=0)
    ax.legend(loc='best', fontsize=11, frameon=False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved valid power gain bar plot to {out_path}")


def plot_sweep_valid_power_gain_by_features(results_dict, out_path="valid_power_gain_by_features.pdf"):
    """Plot grouped bar chart of valid power gain by feature count for each distribution.

    Creates one plot per distribution showing gain across different feature counts.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys (test, metric) like ('gcm', 'type1'), ('gcm', 'fdp'), ('hrt', 'type1'), ('hrt', 'fdp')
        Values are the final_results from aggregate_sweep_results containing 'performance' dict
    out_path : str or Path
        Output path for the plots
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_results = {}
    for (test, metric), result in results_dict.items():
        metric_key = "type1" if metric == "area" else metric
        normalized_results[(test, metric_key)] = result
    results_dict = normalized_results

    # Define colors for the 4 bars
    colors = {
        ('gcm', 'type1'): '#1f77b4',   # blue
        ('gcm', 'fdp'): '#ff7f0e',     # orange
        ('hrt', 'type1'): '#2ca02c',   # green
        ('hrt', 'fdp'): '#d62728',     # red
    }

    # Labels for legend
    labels = {
        ('gcm', 'type1'): 'GCM (Type-I)',
        ('gcm', 'fdp'): 'GCM (FDP)',
        ('hrt', 'type1'): 'HRT (Type-I)',
        ('hrt', 'fdp'): 'HRT (FDP)',
    }

    # Feature sizes and corresponding sample sizes (n = 5*m for 10 and 25, n = 10*m for 50)
    feature_configs = [
        (10, 50),
        (25, 250),
        (50, 500)
    ]

    # Extract all distributions
    all_dists = set()
    for (test, metric), result in results_dict.items():
        performance = result.get('performance', {})
        dists = {dist for dist, n, m in performance.keys()}
        all_dists.update(dists)

    distributions = sorted(all_dists)
    if not distributions:
        print("No distributions found in results")
        return

    # Create one plot per distribution
    for dist in distributions:
        # Prepare data for plotting
        bar_data = {key: {'means': [], 'stds': []} for key in colors.keys()}
        feature_labels = []

        for m, n in feature_configs:
            feature_labels.append(f'm={m}')
            for (test, metric) in [('gcm', 'type1'), ('gcm', 'fdp'), ('hrt', 'type1'), ('hrt', 'fdp')]:
                if (test, metric) not in results_dict:
                    bar_data[(test, metric)]['means'].append(0)
                    bar_data[(test, metric)]['stds'].append(0)
                    continue

                performance = results_dict[(test, metric)].get('performance', {})
                entry = performance.get((dist, n, m))

                if entry:
                    gain_mean = entry.get('valid_power_gain_mean', 0)
                    gain_std = entry.get('valid_power_gain_std', 0)
                    # Convert to standard error (±SE), assuming 100 runs
                    gain_se = gain_std / 10.0
                    bar_data[(test, metric)]['means'].append(gain_mean)
                    bar_data[(test, metric)]['stds'].append(gain_se)
                else:
                    bar_data[(test, metric)]['means'].append(0)
                    bar_data[(test, metric)]['stds'].append(0)

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Bar width and positions
        bar_width = 0.2
        x = np.arange(len(feature_labels))
        offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

        # Plot bars for each (test, metric) combination
        for idx, (test, metric) in enumerate([('gcm', 'type1'), ('gcm', 'fdp'), ('hrt', 'type1'), ('hrt', 'fdp')]):
            means = bar_data[(test, metric)]['means']
            stds = bar_data[(test, metric)]['stds']
            ax.bar(x + offsets[idx], means, bar_width,
                   label=labels[(test, metric)],
                   color=colors[(test, metric)],
                   yerr=stds, capsize=3, error_kw={'linewidth': 1})

        # Customize plot
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Feature Count (m)', fontsize=12)
        ax.set_ylabel('Valid Power Gain', fontsize=12)
        ax.set_title(f'Valid Power Gain by Feature Count - {dist.capitalize()}', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_labels, rotation=0)
        ax.legend(loc='best', fontsize=11, frameon=False)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 0.8)

        fig.tight_layout()

        # Save with distribution name in filename
        dist_out_path = out_path.parent / f"{out_path.stem}_{dist}{out_path.suffix}"
        fig.savefig(dist_out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved valid power gain by features plot for {dist} to {dist_out_path}")


def plot_semi_performance(performance_by_response_test, out_path):
    """Plot valid power and realized FDR vs nominal FDR for semi-supervised experiments.

    Changes:
      • For GCM: only show GCM (uncalibrated & calibrated), no CONTRA lines.
      • For HRT: only compare against CONTRA-HRT and CONTRA-FASTCRT (drop CONTRA-CRT).
      • Add a second plot per (response, test) for realized FDR with a red dotted y=x target line.
      • Use orange for HRT and green for FASTCRT (avoid red for method lines).

    Creates one (or two) plots per (response, test):
      1) Valid Power (existing)
      2) Realized FDR (new)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not performance_by_response_test:
        return

    # Colors for methods (avoid red for method lines)
    colors = {
        'uncalibrated': '#808080',   # gray
        'calibrated':   '#1f77b4',   # blue
        'crt':          '#9467bd',   # purple (in case used elsewhere)
        'hrt':          '#ff7f0e',   # orange
        'fastcrt':      '#2ca02c',   # green
        'target_fdr':   '#d62728',   # red (reference line only)
    }

    markers = {
        'uncalibrated': 'o',
        'calibrated':   's',
        'crt':          '^',
        'hrt':          'v',
        'fastcrt':      'D',
    }

    # Small helper to fetch arrays with graceful fallbacks
    def _arr(d, keys):
        for k in keys:
            if k in d:
                return np.asarray(d.get(k, []), dtype=float)
        return np.asarray([], dtype=float)

    response_lookup = {
        (str(resp).lower(), test_key): (resp, test_key)
        for (resp, test_key) in performance_by_response_test.keys()
    }

    for (response, test) in sorted(performance_by_response_test.keys()):
        payload = performance_by_response_test[(response, test)]
        resp_lower = str(response).lower()
        if resp_lower in {"linear", "nonlinear"}:
            alt_lower = "nonlinear" if resp_lower == "linear" else "linear"
            alt_key = response_lookup.get((alt_lower, test))
            if alt_key is not None:
                payload = performance_by_response_test[alt_key]
        test_lower = str(test).lower()
        alphas = np.asarray(payload.get('alphas', []), dtype=float)
        if alphas.size == 0:
            continue

        # --------------------------
        # 1) VALID POWER PLOT
        # --------------------------
        fig, ax = plt.subplots(figsize=(7.0, 5.0))

        # Uncalibrated
        mean_vals = _arr(payload, ['valid_pow_raw_mean'])
        std_vals  = _arr(payload, ['valid_pow_raw_std'])
        se_vals   = 2 * std_vals / 10.0  # 95% band if ~100 runs
        if mean_vals.size > 0:
            ax.plot(alphas, mean_vals, linewidth=2.0,
                    label=f'{test.upper()} (Uncalibrated)',
                    color=colors['uncalibrated'],
                    marker=markers['uncalibrated'], markersize=6)
            lower = np.clip(mean_vals - se_vals, 0.0, 1.0)
            upper = np.clip(mean_vals + se_vals, 0.0, 1.0)
            ax.fill_between(alphas, lower, upper, alpha=0.2, color=colors['uncalibrated'])

        # Calibrated
        mean_vals = _arr(payload, ['valid_pow_cal_mean'])
        std_vals  = _arr(payload, ['valid_pow_cal_std'])
        se_vals   = 2 * std_vals / 10.0
        if mean_vals.size > 0:
            ax.plot(alphas, mean_vals, linewidth=2.0,
                    label=f'{test.upper()} (Calibrated)',
                    color=colors['calibrated'],
                    marker=markers['calibrated'], markersize=6)
            lower = np.clip(mean_vals - se_vals, 0.0, 1.0)
            upper = np.clip(mean_vals + se_vals, 0.0, 1.0)
            ax.fill_between(alphas, lower, upper, alpha=0.2, color=colors['calibrated'])

        # CONTRA methods (filtered per test)
        if test_lower != 'gcm' and "contra" in payload:
            contra = payload["contra"]
            # For HRT, only show CONTRA-HRT and CONTRA-FASTCRT
            contra_methods = ['hrt', 'fastcrt'] if test_lower == 'hrt' else ['crt', 'hrt', 'fastcrt']
            for method in contra_methods:
                if method in contra:
                    cdata = contra[method]
                    mean_vals = _arr(cdata, ['valid_pow_mean'])
                    std_vals  = _arr(cdata, ['valid_pow_std'])
                    se_vals   = 2 * std_vals / 10.0
                    if mean_vals.size > 0:
                        ax.plot(alphas, mean_vals, linewidth=2.0,
                                label=f'CONTRA-{method.upper()}',
                                color=colors.get(method, '#000000'),
                                marker=markers.get(method, 'x'), markersize=6)
                        lower = np.clip(mean_vals - se_vals, 0.0, 1.0)
                        upper = np.clip(mean_vals + se_vals, 0.0, 1.0)
                        ax.fill_between(alphas, lower, upper, alpha=0.2, color=colors.get(method, '#000000'))

        ax.set_xlabel('Nominal FDR', fontsize=12)
        ax.set_ylabel('Valid Power', fontsize=12)
        ax.set_xlim(-0.02, 0.32)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=12, loc='best')
        response_label = str(response).capitalize()
        ax.set_title(f'Valid Power on GDSC ({response_label}, {test.upper()})', fontsize=13)
        fig.tight_layout()

        # Save (keep original naming for power)
        power_out = out_path.parent / f"{out_path.stem}_{response}_{test}{out_path.suffix}"
        fig.savefig(power_out, bbox_inches='tight')
        plt.close(fig)

        # --------------------------
        # 2) REALIZED FDR PLOT
        # --------------------------
        fig2, ax2 = plt.subplots(figsize=(7.0, 5.0))

        # Uncalibrated FDR
        mean_vals = _arr(payload, ['fdr_raw_mean', 'fdp_raw_mean', 'realized_fdr_raw_mean'])
        std_vals  = _arr(payload, ['fdr_raw_std',  'fdp_raw_std',  'realized_fdr_raw_std'])
        se_vals   = 2 * std_vals / 10.0
        if mean_vals.size > 0:
            ax2.plot(alphas, mean_vals, linewidth=2.0,
                     label=f'{test.upper()} (Uncalibrated)',
                     color=colors['uncalibrated'],
                     marker=markers['uncalibrated'], markersize=6)
            lower = np.clip(mean_vals - se_vals, 0.0, 1.0)
            upper = np.clip(mean_vals + se_vals, 0.0, 1.0)
            ax2.fill_between(alphas, lower, upper, alpha=0.2, color=colors['uncalibrated'])

        # Calibrated FDR
        mean_vals = _arr(payload, ['fdr_cal_mean', 'fdp_cal_mean', 'realized_fdr_cal_mean'])
        std_vals  = _arr(payload, ['fdr_cal_std',  'fdp_cal_std',  'realized_fdr_cal_std'])
        se_vals   = 2 * std_vals / 10.0
        if mean_vals.size > 0:
            ax2.plot(alphas, mean_vals, linewidth=2.0,
                     label=f'{test.upper()} (Calibrated)',
                     color=colors['calibrated'],
                     marker=markers['calibrated'], markersize=6)
            lower = np.clip(mean_vals - se_vals, 0.0, 1.0)
            upper = np.clip(mean_vals + se_vals, 0.0, 1.0)
            ax2.fill_between(alphas, lower, upper, alpha=0.2, color=colors['calibrated'])

        # CONTRA FDR (respect filtering rules)
        if test_lower != 'gcm' and "contra" in payload:
            contra = payload["contra"]
            contra_methods = ['hrt', 'fastcrt'] if test_lower == 'hrt' else ['crt', 'hrt', 'fastcrt']
            for method in contra_methods:
                if method in contra:
                    cdata = contra[method]
                    mean_vals = _arr(cdata, ['fdr_mean', 'fdp_mean', 'realized_fdr_mean'])
                    std_vals  = _arr(cdata, ['fdr_std',  'fdp_std',  'realized_fdr_std'])
                    se_vals   = 2 * std_vals / 10.0
                    if mean_vals.size > 0:
                        ax2.plot(alphas, mean_vals, linewidth=2.0,
                                 label=f'CONTRA-{method.upper()}',
                                 color=colors.get(method, '#000000'),
                                 marker=markers.get(method, 'x'), markersize=6)
                        lower = np.clip(mean_vals - se_vals, 0.0, 1.0)
                        upper = np.clip(mean_vals + se_vals, 0.0, 1.0)
                        ax2.fill_between(alphas, lower, upper, alpha=0.2, color=colors.get(method, '#000000'))

        # Target FDR y=x dashed line (red, reference only)
        ax2.plot(alphas, alphas, linestyle='--', linewidth=1.5,
                 color=colors['target_fdr'], label='Target FDR')

        ax2.set_xlabel('Nominal FDR', fontsize=12)
        ax2.set_ylabel('Realized FDR', fontsize=12)
        ax2.set_xlim(-0.02, 0.32)
        ax2.set_ylim(0.0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=12, loc='best')
        ax2.set_title(f'Realized FDR on GDSC ({response_label}, {test.upper()})', fontsize=13)
        fig2.tight_layout()

        # Save FDR figure (add _fdr to filename)
        fdr_out = out_path.parent / f"{out_path.stem}_{response}_{test}_fdr{out_path.suffix}"
        fig2.savefig(fdr_out, bbox_inches='tight')
        plt.close(fig2)
