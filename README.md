# PP Virtual Quality Lab

A Streamlit decision-support tool that predicts three polypropylene (PP)
product properties - **Izod Impact**, **Tensile Modulus**, and **Flexural
Modulus** - from process inputs (MFR, XS, C2) and a selected product grade.

> **This tool is decision-support only.** Predictions do not replace
> laboratory confirmation. See [Limitations](#limitations--intended-use)
> below before relying on any output.

## Quick start (local)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

The app loads `pp_virtual_lab_models .joblib` (note the space before the
extension - this is the actual committed filename) from the same directory
as `app.py`, resolved relative to the script location so it works
regardless of the working directory Streamlit is launched from.

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests in `tests/test_model_utils.py` load the **real** committed model
artifact (not a mock) and verify: the artifact's structural shape, that
each grade in the UI's dropdown has a matching trained schema entry, that
an untrained grade is rejected rather than silently mispredicted, and that
the local CSV prediction logger behaves correctly.

## Project structure

```
app.py                          Streamlit UI (thin - delegates to model_utils)
model_utils.py                  Model loading, schema validation, prediction,
                                 and CSV logging - framework-agnostic, testable
tests/test_model_utils.py       pytest suite
requirements.txt                Runtime dependencies (pinned)
requirements-dev.txt            Runtime + pytest, for local development
pp_virtual_lab_models .joblib   Trained model artifact (see below)
```

## Model artifact

`pp_virtual_lab_models .joblib` is a dict with three keys:

```python
{
    "models": {
        "Izod Impact": <sklearn RandomForestRegressor>,
        "Tensile Modulus": <sklearn RandomForestRegressor>,
        "Flexural Modulus": <sklearn RandomForestRegressor>,
    },
    "feature_columns": {
        # per target: ["MFR", "XS", "C2", "Grade_<code>", ...] (34 columns:
        # 3 numeric process inputs + one-hot dummies for 31 trained grades)
        "Izod Impact": [...],
        "Tensile Modulus": [...],
        "Flexural Modulus": [...],
    },
    "metrics": {
        # validation metrics computed at training time
        "Izod Impact": {"rows": 220, "r2": 0.974, "mae": 1.83},
        "Tensile Modulus": {"rows": 220, "r2": 0.574, "mae": 87.05},
        "Flexural Modulus": {"rows": 220, "r2": 0.599, "mae": 57.83},
    },
}
```

The R² / MAE values shown in the app's "Model Validation" section are read
directly from `metrics` at runtime, not hardcoded, so they always reflect
whatever artifact is actually loaded.

**This repository does not contain the training data, training script, or
model versioning/provenance information.** The artifact is treated as a
black-box input; if you retrain or replace it, keep the same three-key
structure (`models` / `feature_columns` / `metrics`) and per-target
`feature_columns` schema, or update `model_utils.py` accordingly.

### Trained vs. selectable grades

The artifact was trained on 31 grade codes (one-hot encoded as
`Grade_<code>` feature columns), but the UI only exposes 11 of them in the
`Grade` dropdown in `app.py`. `GRADE_OPTIONS` should only ever contain
grades that exist as a `Grade_<code>` column in every target's
`feature_columns` - `model_utils.build_feature_row()` enforces this at
prediction time regardless, raising a clear error if a mismatch ever
occurs, but the dropdown should not offer known-invalid options in the
first place.

## Prediction logging

Every successful prediction is appended, best-effort, to
`predictions_log.csv` (timestamp, inputs, predicted values) in the app's
own directory, via `model_utils.log_prediction()`. This file is
git-ignored - it's a local audit trail, not application data.

**Important caveat:** if this app is deployed on **Streamlit Community
Cloud**, the filesystem is ephemeral. `predictions_log.csv` will be wiped
on every app restart or redeploy and will not accumulate real history
there. This logging is useful for local runs or any deployment with a
persistent disk; it is not a substitute for a real database if you need
durable prediction history in that environment.

## Limitations & intended use

- **Impact prediction (R² ≈ 0.97) is well validated. Tensile and Flexural
  Modulus (R² ≈ 0.57-0.60) are early estimates** and must not be used
  alone for product release decisions - see the in-app warning.
- Input bounds on MFR/XS/C2 (`min_value`/`max_value` in `app.py`) are
  generic, physically-reasonable PP process ranges, **not** statistically
  derived from the actual training data - the artifact carries no
  per-feature training range metadata to derive tighter bounds from.
- Only the 11 grades listed in `GRADE_OPTIONS` are selectable; predictions
  are refused for any grade without a matching trained schema entry.
- No authentication - if deployed publicly, anyone with the link can use
  the tool.
