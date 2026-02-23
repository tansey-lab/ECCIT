# ECCIT

ECCIT (Empirically Calibrated Conditional Independence Tests) is a framework for calibrating conditional independence tests (CITs) in settings where nominal guarantees fail in practice. Conditional independence tests are widely used for causal discovery and feature selection, but even when paired with false discovery rate (FDR) control procedures, they can fail to provide reliable frequentist guarantees.

ECCIT measures and corrects this miscalibration. For a chosen base CIT, such as the Generalized Covariance Measure (GCM) or the Holdout Randomization Test (HRT), ECCIT optimizes an adversary that selects features and response functions to maximize a miscalibration metric, then fits a monotone calibration map to adjust the resulting p-values or testing thresholds.

This repository contains the code used to run ECCIT experiments, reproduce benchmarks, and generate plots for local and cluster workflows.

---


## Setup

```bash
pip install -e .
```

## Usage

### Local experiments
```bash
eccit sweep fdp gcm
eccit sweep type1 gcm
eccit masks type1 gcm
eccit second_order fdp gcm

# test defaults to gcm
eccit sweep fdp
```

### Single-experiment demo
```bash
python -m eccit.experiments.singles --scenario all
```

### Cluster workflow
```bash
eccit-cluster submit sweep fdp gcm
eccit-cluster plot sweep fdp gcm

# generate jobs without submitting
eccit-cluster generate sweep type1 gcm
```

### Manual aggregation/plots
```bash
eccit-collect-results sweep fdp --test gcm
eccit-plot-results cluster_results/sweep_fdp_gcm/combined_sweep_fdp_gcm.pkl --out-dir outputs/cluster_plots
```

## GDSC note
Some experiments (for example `gdsc` settings in semi/benchmark workflows) require `gdsc_all_features.csv` in the repo root. You can obtain GDSC data from the cancerrxgene website: `https://www.cancerrxgene.org/`.

## Project layout
- `eccit/cli/` - CLI entrypoints
- `eccit/experiments/` - experiment runners
- `eccit/cits/` - CIT implementations (GCM, HRT)
- `eccit/utils/` - shared helpers/utilities
- `inputs/` - local input arrays used by some workflows
