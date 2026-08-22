"""Future model validation support for the PP Virtual Quality Lab.

Compares the EXISTING trained model's predictions against real lab
"Actual" measurements uploaded later (after new lab results come in), to
check whether model accuracy has drifted over time.

This module never retrains, refits, or otherwise modifies the deployed
model in any way - it only scores it, using exactly the same prediction
path (model_utils.predict_all) as Single Prediction and Batch Prediction.

Kept free of any Streamlit dependency, like batch_utils.py, so it can be
imported and tested directly with pytest.

Design notes:
  - Expected lab file columns: Production Date, Batch Number, Grade, MFR,
    XS, C2, Actual Izod Impact, Actual Tensile Modulus, Actual Flexural
    Modulus. Production Date and Batch Number are metadata only, passed
    through unchanged like in batch_utils.py - never read by the model.
  - Grade/MFR/XS/C2/Grade Family validation and the HOMO/RACO/HECO family
    rule are identical to batch_utils.py's, including using
    model_utils.trained_grades() (the model's real 31-grade schema) as
    the grade allowlist, not the narrower SUPPORTED_GRADES UI list - a
    future lab file may legitimately reference any grade the model was
    actually trained on.
  - Every row is validated independently: one bad row (bad grade, bad
    input, missing/non-numeric Actual value) never blocks the rest of
    the file from being scored.
  - R² / MAE / RMSE are computed only from rows that validated and
    predicted successfully (Validation Status == OK).
"""

import re

import numpy as np
import pandas as pd

from batch_utils import (
    REQUIRED_FILE_COLUMNS,
    _clean_str,
    _is_blank,
    _parse_numeric,
    detect_column_mapping,
)
from model_utils import (
    C2_BOUNDS,
    FAMILY_HOMO,
    MFR_BOUNDS,
    TARGET_PROPERTIES,
    XS_BOUNDS,
    derive_grade_family,
    normalize_family_input,
    predict_all,
    trained_grades,
)

PREDICTION_COLUMN_NAMES = {target: f"Predicted {target}" for target in TARGET_PROPERTIES}
ACTUAL_COLUMN_NAMES = {target: f"Actual {target}" for target in TARGET_PROPERTIES}
STATUS_COLUMN = "Validation Status"
ERROR_COLUMN = "Validation Error"
STATUS_OK = "OK"
STATUS_ERROR = "ERROR"

# Recognized (canonicalized) header names for the three lab "Actual"
# measurement columns -> canonical column name. Matching is
# case/whitespace/punctuation-insensitive (see _normalize_key), and
# accepts the bare property name as well as the "Actual "-prefixed form.
_ACTUAL_KEY_MAP = {
    "actualizodimpact": "Actual Izod Impact",
    "izodimpact": "Actual Izod Impact",
    "actualtensilemodulus": "Actual Tensile Modulus",
    "tensilemodulus": "Actual Tensile Modulus",
    "actualflexuralmodulus": "Actual Flexural Modulus",
    "flexuralmodulus": "Actual Flexural Modulus",
}

REQUIRED_ACTUAL_COLUMNS = list(ACTUAL_COLUMN_NAMES.values())


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def detect_actual_column_mapping(columns) -> dict:
    """Map canonical "Actual <Target>" names to the actual column names
    present in the uploaded file, matched case/whitespace/punctuation-
    insensitively. First match wins for a given canonical name."""
    mapping = {}
    for column in columns:
        canonical = _ACTUAL_KEY_MAP.get(_normalize_key(column))
        if canonical and canonical not in mapping:
            mapping[canonical] = column
    return mapping


def _parse_actual(value, field_name: str):
    """Returns (float_value, error_message). Exactly one is None.

    Unlike MFR/XS/C2 (model inputs), Actual lab measurements have no
    fixed bounds - they're real measured outputs, not process inputs -
    so this only checks presence and numeric-ness, never a range.
    """
    if _is_blank(value):
        return None, f"{field_name} is missing"
    try:
        return float(value), None
    except (TypeError, ValueError):
        return None, f"{field_name} is not numeric (got {value!r})"


def validate_and_score_batch(df: pd.DataFrame, models: dict, feature_columns: dict) -> pd.DataFrame:
    """Validate every row of df, predict all three target properties using
    the EXISTING model, and carry the uploaded "Actual" lab values through
    unchanged for later scoring via compute_validation_metrics().

    Returns a new DataFrame: original columns first (unchanged, including
    whatever Actual columns the file had), then prediction columns, then
    Validation Status/Error - identical column-ordering convention to
    batch_utils.validate_and_predict_batch().

    Raises ValueError only for file-level problems (a required column is
    entirely absent, including any of the three Actual columns). Row-level
    problems never raise - they're recorded per row instead, and other
    rows are still processed.
    """
    mapping = detect_column_mapping(df.columns)
    actual_mapping = detect_actual_column_mapping(df.columns)
    missing = [name for name in REQUIRED_FILE_COLUMNS if name not in mapping]
    missing += [name for name in REQUIRED_ACTUAL_COLUMNS if name not in actual_mapping]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Columns found in file: {list(df.columns)}"
        )

    valid_grades = trained_grades(feature_columns)

    grade_col = mapping["Grade"]
    mfr_col = mapping["MFR"]
    xs_col = mapping["XS"]
    c2_col = mapping.get("C2")
    family_col = mapping.get("Grade Family")

    statuses = []
    error_messages = []
    prediction_values = {target: [] for target in TARGET_PROPERTIES}

    for _, row in df.iterrows():
        row_errors = []

        grade = _clean_str(row[grade_col])
        grade_valid = False
        if not grade:
            row_errors.append("Grade is missing")
        elif grade not in valid_grades:
            row_errors.append(
                f"Unsupported grade '{grade}': not part of the model's "
                f"{len(valid_grades)} trained grades."
            )
        else:
            grade_valid = True

        mfr, mfr_error = _parse_numeric(row[mfr_col], "MFR", MFR_BOUNDS)
        if mfr_error:
            row_errors.append(mfr_error)

        xs, xs_error = _parse_numeric(row[xs_col], "XS", XS_BOUNDS)
        if xs_error:
            row_errors.append(xs_error)

        derived_family = None
        if grade_valid:
            try:
                derived_family = derive_grade_family(grade)
            except ValueError as exc:
                row_errors.append(str(exc))

        family = derived_family
        if family_col is not None:
            raw_family = _clean_str(row[family_col])
            if raw_family:
                supplied_family = normalize_family_input(raw_family)
                if supplied_family is None:
                    row_errors.append(
                        f"Invalid Grade Family '{raw_family}'. Expected HOMO, "
                        "RACO, or HECO (ICP is accepted as a synonym for HECO)."
                    )
                elif derived_family is not None and supplied_family != derived_family:
                    row_errors.append(
                        f"Grade Family '{raw_family}' does not match grade "
                        f"'{grade}' (grade codes starting with '{grade[0]}' "
                        f"are {derived_family})."
                    )
                else:
                    family = supplied_family

        c2_cell = row[c2_col] if c2_col is not None else None
        c2_blank = c2_col is None or _is_blank(c2_cell)

        if family == FAMILY_HOMO and c2_blank:
            c2 = 0.0
        elif c2_blank:
            reason = f" for family {family}" if family else ""
            row_errors.append(f"C2 is required{reason}")
            c2 = None
        else:
            c2, c2_error = _parse_numeric(c2_cell, "C2", C2_BOUNDS)
            if c2_error:
                row_errors.append(c2_error)

        # Actual lab measurements - always validated, needed for scoring.
        for target in TARGET_PROPERTIES:
            actual_column = ACTUAL_COLUMN_NAMES[target]
            raw_actual = row[actual_mapping[actual_column]]
            _, actual_error = _parse_actual(raw_actual, actual_column)
            if actual_error:
                row_errors.append(actual_error)

        if row_errors:
            statuses.append(STATUS_ERROR)
            error_messages.append("; ".join(row_errors))
            for target in TARGET_PROPERTIES:
                prediction_values[target].append(None)
            continue

        try:
            predictions = predict_all(models, feature_columns, grade, mfr, xs, c2)
        except Exception as exc:
            statuses.append(STATUS_ERROR)
            error_messages.append(str(exc))
            for target in TARGET_PROPERTIES:
                prediction_values[target].append(None)
            continue

        statuses.append(STATUS_OK)
        error_messages.append("")
        for target in TARGET_PROPERTIES:
            prediction_values[target].append(predictions[target])

    result = df.copy()
    for target in TARGET_PROPERTIES:
        result[PREDICTION_COLUMN_NAMES[target]] = prediction_values[target]
    result[STATUS_COLUMN] = statuses
    result[ERROR_COLUMN] = error_messages
    return result


def summarize_validation_results(result_df: pd.DataFrame) -> dict:
    total = len(result_df)
    ok = int((result_df[STATUS_COLUMN] == STATUS_OK).sum()) if total else 0
    return {"total": total, "ok": ok, "errors": total - ok}


def compute_validation_metrics(result_df: pd.DataFrame) -> dict:
    """Compute R², MAE, RMSE per target property from OK rows only.

    Comparison/scoring only - never touches, retrains, or otherwise
    modifies the deployed model.

    Returns {target: {"n": int, "r2": float|None, "mae": float|None,
    "rmse": float|None}}. r2 is None whenever it isn't statistically
    meaningful: fewer than 2 scored rows, or every Actual value in the
    scored rows is identical (zero variance, undefined R²). mae/rmse are
    still computed from as few as 1 row. A target with 0 scored rows gets
    every metric set to None.
    """
    actual_mapping = detect_actual_column_mapping(result_df.columns)
    ok_rows = result_df[result_df[STATUS_COLUMN] == STATUS_OK]

    metrics = {}
    for target in TARGET_PROPERTIES:
        actual_column = actual_mapping[ACTUAL_COLUMN_NAMES[target]]
        actual = pd.to_numeric(ok_rows[actual_column], errors="coerce")
        predicted = pd.to_numeric(ok_rows[PREDICTION_COLUMN_NAMES[target]], errors="coerce")

        n = int(actual.notna().sum())
        if n == 0:
            metrics[target] = {"n": 0, "r2": None, "mae": None, "rmse": None}
            continue

        errors = actual - predicted
        mae = float(errors.abs().mean())
        rmse = float(np.sqrt((errors ** 2).mean()))

        r2 = None
        if n >= 2:
            ss_res = float((errors ** 2).sum())
            ss_tot = float(((actual - actual.mean()) ** 2).sum())
            if ss_tot > 0:
                r2 = float(1 - ss_res / ss_tot)

        metrics[target] = {"n": n, "r2": r2, "mae": mae, "rmse": rmse}

    return metrics
