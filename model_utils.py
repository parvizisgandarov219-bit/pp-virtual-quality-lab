"""Model loading, schema validation, and prediction helpers for the
PP Virtual Quality Lab.

Kept free of any Streamlit dependency so it can be imported and tested
directly (e.g. with pytest) without a running Streamlit session.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

TARGET_PROPERTIES = ["Izod Impact", "Tensile Modulus", "Flexural Modulus"]

# Grades with a verified trained schema entry (Grade_<code> column present
# for every target - see tests/test_model_utils.py). This is the single
# source of truth for both the single-prediction dropdown and batch
# validation, so they cannot drift apart.
SUPPORTED_GRADES = [
    "CA0900BM",
    "CB0900MO",
    "CB1248MO",
    "CB1640MO",
    "CB1849MO",
    "CB3000GT",
    "CB3648MO",
    "CB4048MO",
    "CB4848MO",
    "CB6448MO",
    "CB8248MO",
]

# Input bounds shared by the single-prediction form and batch validation.
# These are generic, physically-reasonable PP process ranges, not ranges
# statistically derived from the training data (the artifact carries no
# per-feature training range metadata) - see README.
MFR_BOUNDS = (0.0, 150.0)
XS_BOUNDS = (0.0, 40.0)
C2_BOUNDS = (0.0, 25.0)

# Grade family logic for batch prediction (see batch_utils.py). Not derived
# from or cross-checked against the grade code itself - this repo has no
# verified Grade -> Family mapping, so family is taken as given by the
# uploader via an optional "Grade Family" column.
FAMILY_HOMO = "HOMO"
FAMILY_RACO = "RACO"
FAMILY_HECO = "HECO"
FAMILY_ICP = "ICP"
VALID_FAMILIES = {FAMILY_HOMO, FAMILY_RACO, FAMILY_HECO, FAMILY_ICP}
FAMILIES_REQUIRING_C2 = {FAMILY_RACO, FAMILY_HECO, FAMILY_ICP}

LOG_COLUMNS = [
    "timestamp_utc",
    "grade",
    "mfr",
    "xs",
    "c2",
    "izod_impact",
    "tensile_modulus",
    "flexural_modulus",
]


def load_model_artifact(path: Path) -> dict:
    """Load the trained model artifact and verify it has the expected shape.

    Expected structure:
        {
            "models": {target_name: fitted sklearn estimator, ...},
            "feature_columns": {target_name: [ordered feature names], ...},
            "metrics": {target_name: {"rows": int, "r2": float, "mae": float}, ...},
        }

    Raises FileNotFoundError or ValueError with a human-readable message
    if the artifact is missing or malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError(
            f"Model artifact has unexpected type {type(artifact).__name__}; "
            "expected a dict with 'models', 'feature_columns', and 'metrics'."
        )

    required_keys = {"models", "feature_columns", "metrics"}
    missing_keys = required_keys - artifact.keys()
    if missing_keys:
        raise ValueError(
            f"Model artifact is missing expected key(s): {sorted(missing_keys)}"
        )

    missing_targets = [
        target for target in TARGET_PROPERTIES
        if target not in artifact["models"]
        or target not in artifact["feature_columns"]
        or target not in artifact["metrics"]
    ]
    if missing_targets:
        raise ValueError(
            f"Model artifact is missing data for target(s): {missing_targets}"
        )

    return artifact


def build_feature_row(
    target: str,
    feature_columns: dict,
    grade: str,
    mfr: float,
    xs: float,
    c2: float,
) -> pd.DataFrame:
    """Build a single-row DataFrame matching the exact schema the target's
    model was trained on (raw numeric features + one-hot grade dummies).

    Raises ValueError if the selected grade was not part of the training
    schema for this target, instead of silently predicting on an
    all-zero / unknown grade encoding.
    """
    columns = feature_columns[target]
    row = {column: 0.0 for column in columns}

    for base_column, value in (("MFR", mfr), ("XS", xs), ("C2", c2)):
        if base_column not in row:
            raise ValueError(
                f"Expected feature '{base_column}' is missing from the "
                f"{target} model's trained schema."
            )
        row[base_column] = value

    grade_column = f"Grade_{grade}"
    if grade_column not in row:
        raise ValueError(
            f"Grade '{grade}' was not part of the training data for the "
            f"{target} model, so a reliable prediction cannot be produced "
            "for this grade."
        )
    row[grade_column] = 1.0

    return pd.DataFrame([row], columns=columns)


def predict_all(
    models: dict,
    feature_columns: dict,
    grade: str,
    mfr: float,
    xs: float,
    c2: float,
) -> dict:
    """Run all three target models for the given inputs and return
    {target_name: predicted_value}.

    Raises ValueError if the grade isn't part of any target's trained
    schema (via build_feature_row).
    """
    predictions = {}
    for target in TARGET_PROPERTIES:
        feature_row = build_feature_row(target, feature_columns, grade, mfr, xs, c2)
        predictions[target] = models[target].predict(feature_row)[0]
    return predictions


def log_prediction(log_path: Path, grade: str, mfr: float, xs: float, c2: float, predictions: dict) -> None:
    """Append one row to a local CSV audit log of predictions made.

    Best-effort only: intended for local runs and deployments with a
    persistent filesystem. On Streamlit Community Cloud specifically,
    the filesystem is ephemeral and this log will not survive an app
    restart or redeploy - see README for details. Callers should treat
    logging failures as non-fatal.
    """
    log_path = Path(log_path)
    file_exists = log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            grade,
            mfr,
            xs,
            c2,
            predictions.get("Izod Impact"),
            predictions.get("Tensile Modulus"),
            predictions.get("Flexural Modulus"),
        ])
