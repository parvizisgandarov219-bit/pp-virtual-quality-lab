"""Batch (CSV/Excel) prediction support for the PP Virtual Quality Lab.

Kept free of any Streamlit dependency, like model_utils.py, so it can be
imported and tested directly with pytest.

Design notes (see README for the full write-up):
  - Grade Family is derived from the grade code's first letter (see
    model_utils.derive_grade_family): C -> HECO, R -> RACO, H -> HOMO.
  - If the file has no "Grade Family" column, or a row's cell is blank,
    the family is simply derived from Grade - that's not an error.
  - If a row *does* supply a Grade Family value, it must agree with the
    grade-derived family (ICP is accepted as a synonym for HECO) or the
    row is flagged as an error rather than silently corrected/overridden.
  - HOMO rows may omit C2 (it defaults to 0.0); RACO/HECO rows require it.
  - Every row is validated independently: one bad row never blocks the
    rest of the batch from producing predictions.
"""

import io
import re

import pandas as pd

from model_utils import (
    C2_BOUNDS,
    FAMILY_HOMO,
    MFR_BOUNDS,
    SUPPORTED_GRADES,
    TARGET_PROPERTIES,
    XS_BOUNDS,
    derive_grade_family,
    normalize_family_input,
    predict_all,
)

PREDICTION_COLUMN_NAMES = {
    "Izod Impact": "Predicted Izod Impact",
    "Tensile Modulus": "Predicted Tensile Modulus",
    "Flexural Modulus": "Predicted Flexural Modulus",
}
STATUS_COLUMN = "Validation Status"
ERROR_COLUMN = "Validation Error"
STATUS_OK = "OK"
STATUS_ERROR = "ERROR"

# Recognized (canonicalized) header names -> canonical column name.
# Matching is case/whitespace/punctuation-insensitive (see _normalize_key).
_CANONICAL_KEY_MAP = {
    "grade": "Grade",
    "mfr": "MFR",
    "mfrg10min": "MFR",
    "xs": "XS",
    "xswt": "XS",
    "xswtpct": "XS",
    "c2": "C2",
    "c2wt": "C2",
    "c2wtpct": "C2",
    "gradefamily": "Grade Family",
    "family": "Grade Family",
}

REQUIRED_FILE_COLUMNS = ["Grade", "MFR", "XS"]


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def detect_column_mapping(columns) -> dict:
    """Map canonical column names (Grade/MFR/XS/C2/Grade Family) to the
    actual column names present in the uploaded file, matched
    case/whitespace/punctuation-insensitively. First match wins for a
    given canonical name if a file has duplicate-looking headers.
    """
    mapping = {}
    for column in columns:
        canonical = _CANONICAL_KEY_MAP.get(_normalize_key(column))
        if canonical and canonical not in mapping:
            mapping[canonical] = column
    return mapping


def parse_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded CSV/XLSX bytes into a DataFrame.

    Raises ValueError for unsupported file types or unparseable content.
    """
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    try:
        if suffix == "csv":
            return pd.read_csv(io.BytesIO(file_bytes))
        elif suffix in ("xlsx", "xlsm"):
            return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"Could not parse '{filename}': {exc}") from exc

    raise ValueError(
        f"Unsupported file type '.{suffix}'. Please upload a .csv or .xlsx file."
    )


def _is_blank(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def _clean_str(value) -> str:
    return "" if _is_blank(value) else str(value).strip()


def _parse_numeric(value, field_name: str, bounds: tuple):
    """Returns (float_value, error_message). Exactly one is None."""
    if _is_blank(value):
        return None, f"{field_name} is missing"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, f"{field_name} is not numeric (got {value!r})"

    low, high = bounds
    if not (low <= parsed <= high):
        return None, f"{field_name} must be between {low} and {high} (got {parsed})"
    return parsed, None


def validate_and_predict_batch(df: pd.DataFrame, models: dict, feature_columns: dict) -> pd.DataFrame:
    """Validate every row of df and, for valid rows, predict all three
    target properties. Returns a new DataFrame: original columns first
    (unchanged), then prediction columns, then Validation Status/Error.

    Raises ValueError only for file-level problems (a required column is
    entirely absent). Row-level problems never raise - they're recorded
    per row in the Validation Status/Error columns instead, and other
    rows in the same batch are still processed.
    """
    mapping = detect_column_mapping(df.columns)
    missing = [name for name in REQUIRED_FILE_COLUMNS if name not in mapping]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Columns found in file: {list(df.columns)}"
        )

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
        elif grade not in SUPPORTED_GRADES:
            row_errors.append(
                f"Unsupported grade '{grade}'. Supported grades: {', '.join(SUPPORTED_GRADES)}"
            )
        else:
            grade_valid = True

        mfr, mfr_error = _parse_numeric(row[mfr_col], "MFR", MFR_BOUNDS)
        if mfr_error:
            row_errors.append(mfr_error)

        xs, xs_error = _parse_numeric(row[xs_col], "XS", XS_BOUNDS)
        if xs_error:
            row_errors.append(xs_error)

        # Grade Family is derived from the grade code's first letter. A
        # supplied value is never used to silently override that - it's
        # only ever compared against it, and a mismatch is an error.
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
            # a blank cell falls through and keeps the derived family

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


def summarize_batch_results(result_df: pd.DataFrame) -> dict:
    total = len(result_df)
    ok = int((result_df[STATUS_COLUMN] == STATUS_OK).sum()) if total else 0
    return {"total": total, "ok": ok, "errors": total - ok}


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return buffer.getvalue()
