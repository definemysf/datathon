# DualGuard Race-Focused Hiring Audit

This folder contains Miles' local PoC for the datathon.

## Pipeline

`run_race_dualguard_trained.py` runs the full local workflow:

1. Build positive and negative resume-job weak-label pairs from the local datasets.
2. Train a local resume-job matching model.
3. Generate racialized-name counterfactuals.
4. Measure behavioral score gaps.
5. Run layer-wise activation probes with held-out names.
6. Train an XGBoost surrogate and export SHAP explanations.

The audited matcher is not a hand-written score formula. It is a trained matching model using DistilBERT semantic evidence plus structured job-match features.

## Current Results

Latest run:

- Base resume-job pairs: `160`
- Matching training rows: `640`
- Racial counterfactual audit rows: `1280`
- Matching head test AUC: `0.6152`
- Matching head test accuracy: `0.6000`
- Mean absolute racialized-name behavioral gap: `0.0349`
- Mean signed gap, Black-associated minus White-associated names: `-0.0186`
- Max absolute racialized-name behavioral gap: `0.2581`
- Layer 0 held-out-name probe AUC: `0.5000`
- Final layer held-out-name probe AUC: `0.9287`
- Max held-out-name probe AUC: `0.9896`
- Final layer shuffled-label control AUC: `0.5104`
- XGBoost surrogate fidelity R2: `0.9232`
- XGBoost surrogate MAE: `0.0354`

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r miles/requirements.txt
python miles/scripts/run_race_dualguard_trained.py \
  --root /path/to/datathrone \
  --n-base 160 \
  --batch-size 16
```

The script expects the local data directory to contain:

- `postings.csv`
- `archive/Resume/Resume.csv`

These raw data files are intentionally not committed.

## Result Files

Key files are in `miles/results/race_trained/`:

- `audit_metrics.json`
- `ppt_ready_summary_race.md`
- `racial_behavioral_gap.png`
- `heldout_name_probe_auc.png`
- `xgboost_global_shap.png`
- `xgboost_local_shap.png`
