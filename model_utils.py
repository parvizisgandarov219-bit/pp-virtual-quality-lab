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
#
# HB3500GP (HOMO) and RB4545MO (RACO) are appended after the original 11
# HECO grades, rather than inserted alphabetically, specifically so the
# original grades keep their original list positions - app.py's default
# selectbox index (8 = CB4848MO) depends on that order being stable.
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
    "HB3500GP",
    "RB4545MO",
]

# Input bounds shared by the single-prediction form and batch validation.
# These are generic, physically-reasonable PP process ranges, not ranges
# statistically derived from the training data (the artifact carries no
# per-feature training range metadata) - see README.
MFR_BOUNDS = (0.0, 150.0)
XS_BOUNDS = (0.0, 40.0)
C2_BOUNDS = (0.0, 25.0)

# Grade family classification. HOMO/RACO/HECO are the three families the
# system recognizes; ICP is kept only as an accepted synonym for HECO
# ("High Impact Copolymer" is the same thing as "HECO" in this project's
# usage) so text like "ICP" in an uploaded file's Grade Family column is
# still accepted rather than rejected outright.
FAMILY_HOMO = "HOMO"
FAMILY_RACO = "RACO"
FAMILY_HECO = "HECO"
FAMILY_ICP = "ICP"
VALID_FAMILIES = {FAMILY_HOMO, FAMILY_RACO, FAMILY_HECO, FAMILY_ICP}
FAMILIES_REQUIRING_C2 = {FAMILY_RACO, FAMILY_HECO, FAMILY_ICP}

# Input tokens that should be treated as an alias of another family before
# any comparison/validation happens. Currently just ICP -> HECO.
FAMILY_SYNONYMS = {FAMILY_ICP: FAMILY_HECO}

# Exact user-facing labels for each family, as specified by the project
# owner - used verbatim in the UI, not paraphrased.
FAMILY_LABELS = {
    FAMILY_HOMO: "HOMO — Homopolymer",
    FAMILY_RACO: "RACO — Random Copolymer",
    FAMILY_HECO: "HECO — High Impact Copolymer (ICP)",
}

# Grade Family is derived from the grade code's first letter:
#   C -> HECO (High Impact Copolymer / ICP)
#   R -> RACO (Random Copolymer)
#   H -> HOMO (Homopolymer)
# SUPPORTED_GRADES spans all three prefixes (11 "C" / HECO grades,
# HB3500GP / HOMO, RB4545MO / RACO), so all three branches are reachable
# through the real UI, not just via direct unit tests.
GRADE_FAMILY_PREFIX_MAP = {
    "C": FAMILY_HECO,
    "R": FAMILY_RACO,
    "H": FAMILY_HOMO,
}


def derive_grade_family(grade: str) -> str:
    """Derive the Grade Family from a grade code's first letter.

    Raises ValueError if grade is empty, or its first letter isn't one of
    C/R/H. In practice this is unreachable for any grade that has already
    passed a SUPPORTED_GRADES membership check, since every currently
    supported grade starts with "C" - the check exists for correctness on
    any future grade code, not because it's expected to trigger today.
    """
    if not grade:
        raise ValueError("Grade is required to derive its Grade Family.")

    prefix = grade[0].upper()
    family = GRADE_FAMILY_PREFIX_MAP.get(prefix)
    if family is None:
        raise ValueError(
            f"Cannot derive Grade Family for grade '{grade}': grade codes "
            "are expected to start with 'C' (HECO), 'R' (RACO), or 'H' "
            f"(HOMO), but '{grade}' starts with '{prefix}'."
        )
    return family


def normalize_family_input(raw_family: str) -> str | None:
    """Normalize a raw Grade Family string (e.g. from an uploaded file) to
    one of FAMILY_HOMO/FAMILY_RACO/FAMILY_HECO, applying FAMILY_SYNONYMS.

    Returns None if raw_family isn't a recognized family or synonym.
    Case-insensitive; leading/trailing whitespace should already be
    stripped by the caller.
    """
    normalized = raw_family.upper()
    normalized = FAMILY_SYNONYMS.get(normalized, normalized)
    if normalized in (FAMILY_HOMO, FAMILY_RACO, FAMILY_HECO):
        return normalized
    return None

# Display unit for Izod Impact predictions, confirmed by the project owner.
# Tensile/Flexural Modulus are already labeled "MPa" at their call sites.
# This is a display label only - it does not convert, scale, or otherwise
# alter the model's numeric output in any way.
IZOD_IMPACT_UNIT = "kJ/m²"

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


def resolve_c2_for_family(family: str | None, c2_input: float) -> float:
    """Resolve the C2 (ethylene comonomer) value to use for an interactive
    single prediction, given an optional Grade Family selection.

    Mirrors the batch-prediction rule (see batch_utils.py): a HOMO grade
    is a homopolymer, so ethylene comonomer isn't applicable and the
    resolved C2 is always 0.0 regardless of the raw input - the calling
    UI is expected to disable/zero the C2 control when HOMO is selected,
    and this function is the single source of truth enforcing that rule
    regardless. Any other family, or no family specified (None), passes
    c2_input through unchanged - identical to the original single-
    prediction behavior that always required a manually entered C2.
    """
    if family == FAMILY_HOMO:
        return 0.0
    return c2_input


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
