# ECCIT

ECCIT (Empirically Calibrated Conditional Independence Tests) is a framework for empirically calibrating conditional independence tests (CITs) when nominal guarantees break down in practice. CITs can be miscalibrated due to finite-sample effects or model misspecification, which can distort null p-values and cause inflated false discoveries from the test procedures.

ECCIT addresses this by adversarially constructing responses that expose miscalibration for a chosen base test (e.g., GCM or HRT), then fitting a monotone calibration map to correct the resulting p-values or testing thresholds. The framework is test-agnostic and supports calibration under different metrics based around type-I error or false discovery rates.

This repository contains the code used to run ECCIT experiments, reproduce benchmark sweeps, and generate plots for local and cluster workflows.

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
