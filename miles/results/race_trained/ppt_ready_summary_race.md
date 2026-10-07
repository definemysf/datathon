# Race-Focused DualGuard Results

## What changed from the earlier PoC

The audited score is no longer a hand-written weighted formula. The local hiring matcher is a trained model:

- DistilBERT is used locally for semantic representations and hidden states.
- A trained gradient-boosted matching head predicts resume-job suitability from DistilBERT semantic score and job-relevant structured features.
- XGBoost is used only as a surrogate explanation layer, not as the audited model.
- The activation probe uses held-out racialized names: probe training names and audit/test names are disjoint.
- A shuffled-label probe control is included.

## Exact numbers

- Local model: `distilbert-base-uncased` + trained matching head
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
- Race cue mean absolute SHAP: `0.0018`

## Interpretation

The trained local matcher shows a measurable racialized-name behavioral gap, while the activation audit shows that race-associated name information is strongly decodable in later DistilBERT layers even when probe names are held out. The shuffled-label control stays near chance, so the probe is not simply reporting arbitrary separability.

The XGBoost surrogate reaches high fidelity (`R2 = 0.9232`) and attributes most score variation to job-relevant matching features such as category-job keyword hits, semantic score, job length, and overlap features. This gives the demo a complete audit stack:

`Trained local matcher -> Behavioral racial counterfactual audit -> Held-out activation probe -> XGBoost/SHAP explanation`

## Files for deck

- `outputs_race_trained/racial_behavioral_gap.png`
- `outputs_race_trained/heldout_name_probe_auc.png`
- `outputs_race_trained/xgboost_global_shap.png`
- `outputs_race_trained/xgboost_local_shap.png`
- `outputs_race_trained/audit_metrics.json`
