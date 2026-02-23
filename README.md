# ECCIT

## Setup

```bash
conda env create -f environment.yml
conda activate eccit
# or:
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
