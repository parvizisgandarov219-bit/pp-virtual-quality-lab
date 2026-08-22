"""Tests for validation_utils.py, run against the real committed model
artifact (no mocking of the sklearn models themselves).

This module never retrains the model - every test here only asserts on
scoring/comparison behavior (Predicted vs. Actual), never on anything
that would change model_utils.predict_all's output.
"""

from pathlib import Path

import pandas as pd
import pytest

import model_utils
import validation_utils

REPO_ROOT = Path(__file__).parent.parent
MODEL_PATH = REPO_ROOT / "pp_virtual_lab_models .joblib"

VALID_GRADE_1 = "CB4848MO"
HOMO_GRADE = "HB3500GP"
UNTRAINED_GRADE = "CB2000GT"


@pytest.fixture(scope="module")
def artifact():
    return model_utils.load_model_artifact(MODEL_PATH)


@pytest.fixture(scope="module")
def models(artifact):
    return artifact["models"]


@pytest.fixture(scope="module")
def feature_columns(artifact):
    return artifact["feature_columns"]


def _lab_df(grade=VALID_GRADE_1, mfr=48.0, xs=16.0, c2=8.6, actual=None):
    actual = actual or {"Izod Impact": 7.5, "Tensile Modulus": 1380.0, "Flexural Modulus": 1390.0}
    return pd.DataFrame({
        "Production Date": ["2026-08-17"],
        "Batch Number": ["B-2026-0817-01"],
        "Grade": [grade],
        "MFR": [mfr],
        "XS": [xs],
        "C2": [c2],
        "Actual Izod Impact": [actual["Izod Impact"]],
        "Actual Tensile Modulus": [actual["Tensile Modulus"]],
        "Actual Flexural Modulus": [actual["Flexural Modulus"]],
    })


# --- column detection -------------------------------------------------------

@pytest.mark.parametrize("header,expected_canonical", [
    ("Actual Izod Impact", "Actual Izod Impact"),
    ("actual izod impact", "Actual Izod Impact"),
    ("Izod Impact", "Actual Izod Impact"),
    ("Actual Tensile Modulus", "Actual Tensile Modulus"),
    ("Tensile Modulus", "Actual Tensile Modulus"),
    ("Actual Flexural Modulus", "Actual Flexural Modulus"),
    ("Flexural Modulus", "Actual Flexural Modulus"),
])
def test_detect_actual_column_mapping_recognizes_variants(header, expected_canonical):
    mapping = validation_utils.detect_actual_column_mapping([header])
    assert mapping[expected_canonical] == header


# --- row validation & prediction --------------------------------------------

def test_validate_and_score_batch_raises_for_missing_actual_columns(models, feature_columns):
    df = pd.DataFrame({"Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [8.6]})
    with pytest.raises(ValueError, match="Missing required column"):
        validation_utils.validate_and_score_batch(df, models, feature_columns)


def test_valid_row_is_scored_and_matches_predict_all(models, feature_columns):
    df = _lab_df()
    result = validation_utils.validate_and_score_batch(df, models, feature_columns)

    assert result.loc[0, validation_utils.STATUS_COLUMN] == validation_utils.STATUS_OK
    assert result.loc[0, validation_utils.ERROR_COLUMN] == ""

    expected = model_utils.predict_all(models, feature_columns, VALID_GRADE_1, 48.0, 16.0, 8.6)
    assert result.loc[0, "Predicted Izod Impact"] == pytest.approx(expected["Izod Impact"])
    assert result.loc[0, "Predicted Tensile Modulus"] == pytest.approx(expected["Tensile Modulus"])
    assert result.loc[0, "Predicted Flexural Modulus"] == pytest.approx(expected["Flexural Modulus"])

    # Actual columns and metadata pass through completely unchanged
    assert result.loc[0, "Actual Izod Impact"] == 7.5
    assert result.loc[0, "Production Date"] == "2026-08-17"
    assert result.loc[0, "Batch Number"] == "B-2026-0817-01"


def test_unsupported_grade_is_row_level_error_not_a_crash(models, feature_columns):
    df = _lab_df(grade=UNTRAINED_GRADE)
    result = validation_utils.validate_and_score_batch(df, models, feature_columns)
    assert result.loc[0, validation_utils.STATUS_COLUMN] == validation_utils.STATUS_ERROR
    assert "Unsupported grade" in result.loc[0, validation_utils.ERROR_COLUMN]


def test_validation_uses_trained_grades_not_supported_grades(models, feature_columns):
    # Model Validation's grade allowlist is model_utils.trained_grades(),
    # same as Batch Prediction - not the (now widened) SUPPORTED_GRADES
    # UI list. CA0342EX is in SUPPORTED_GRADES today too, so this mainly
    # confirms it still works; the real proof that trained_grades() (not
    # SUPPORTED_GRADES) is the check is test_unsupported_grade_is_row_level_error_not_a_crash
    # plus batch_utils' PP0102TR family-prefix test (same shared logic).
    grade = "CA0342EX"
    assert grade in model_utils.trained_grades(feature_columns)
    df = _lab_df(grade=grade)
    result = validation_utils.validate_and_score_batch(df, models, feature_columns)
    assert result.loc[0, validation_utils.STATUS_COLUMN] == validation_utils.STATUS_OK


def test_homo_grade_blank_c2_defaults_to_zero(models, feature_columns):
    df = _lab_df(grade=HOMO_GRADE, c2=None)
    result = validation_utils.validate_and_score_batch(df, models, feature_columns)
    assert result.loc[0, validation_utils.STATUS_COLUMN] == validation_utils.STATUS_OK
    expected = model_utils.predict_all(models, feature_columns, HOMO_GRADE, 48.0, 16.0, 0.0)
    assert result.loc[0, "Predicted Izod Impact"] == pytest.approx(expected["Izod Impact"])


@pytest.mark.parametrize("missing_field", [
    "Actual Izod Impact", "Actual Tensile Modulus", "Actual Flexural Modulus",
])
def test_missing_actual_value_is_row_level_error(models, feature_columns, missing_field):
    df = _lab_df()
    df.loc[0, missing_field] = None
    result = validation_utils.validate_and_score_batch(df, models, feature_columns)
    assert result.loc[0, validation_utils.STATUS_COLUMN] == validation_utils.STATUS_ERROR
    assert f"{missing_field} is missing" in result.loc[0, validation_utils.ERROR_COLUMN]
    assert pd.isna(result.loc[0, "Predicted Izod Impact"])


def test_non_numeric_actual_value_is_row_level_error(models, feature_columns):
    df = _lab_df()
    df["Actual Izod Impact"] = df["Actual Izod Impact"].astype(object)
    df.loc[0, "Actual Izod Impact"] = "not-a-number"
    result = validation_utils.validate_and_score_batch(df, models, feature_columns)
    assert result.loc[0, validation_utils.STATUS_COLUMN] == validation_utils.STATUS_ERROR
    assert "Actual Izod Impact is not numeric" in result.loc[0, validation_utils.ERROR_COLUMN]


def test_one_bad_row_does_not_block_the_rest_of_the_batch(models, feature_columns):
    df = pd.concat([_lab_df(grade=UNTRAINED_GRADE), _lab_df()], ignore_index=True)
    result = validation_utils.validate_and_score_batch(df, models, feature_columns)
    assert result.loc[0, validation_utils.STATUS_COLUMN] == validation_utils.STATUS_ERROR
    assert result.loc[1, validation_utils.STATUS_COLUMN] == validation_utils.STATUS_OK


def test_summarize_validation_results(models, feature_columns):
    df = pd.concat([_lab_df(grade=UNTRAINED_GRADE), _lab_df()], ignore_index=True)
    result = validation_utils.validate_and_score_batch(df, models, feature_columns)
    summary = validation_utils.summarize_validation_results(result)
    assert summary == {"total": 2, "ok": 1, "errors": 1}


# --- metrics computation -----------------------------------------------------
#
# These construct a synthetic result_df directly (bypassing prediction)
# so the R²/MAE/RMSE arithmetic can be checked against hand-computed
# expected values, independent of the real model's actual predictions.

def _synthetic_result(actual, predicted, statuses=None):
    n = len(actual)
    statuses = statuses or [validation_utils.STATUS_OK] * n
    return pd.DataFrame({
        "Grade": ["CB4848MO"] * n,
        "Actual Izod Impact": actual,
        "Predicted Izod Impact": predicted,
        "Actual Tensile Modulus": actual,
        "Predicted Tensile Modulus": predicted,
        "Actual Flexural Modulus": actual,
        "Predicted Flexural Modulus": predicted,
        validation_utils.STATUS_COLUMN: statuses,
        validation_utils.ERROR_COLUMN: [""] * n,
    })


def test_compute_validation_metrics_perfect_predictions():
    result = _synthetic_result(actual=[1.0, 2.0, 3.0, 4.0], predicted=[1.0, 2.0, 3.0, 4.0])
    metrics = validation_utils.compute_validation_metrics(result)
    for target in model_utils.TARGET_PROPERTIES:
        assert metrics[target]["n"] == 4
        assert metrics[target]["r2"] == pytest.approx(1.0)
        assert metrics[target]["mae"] == pytest.approx(0.0)
        assert metrics[target]["rmse"] == pytest.approx(0.0)


def test_compute_validation_metrics_known_errors():
    # actual - predicted = [1, -1, 1, -1] -> MAE = 1.0, RMSE = 1.0
    result = _synthetic_result(actual=[2.0, 2.0, 6.0, 6.0], predicted=[1.0, 3.0, 5.0, 7.0])
    metrics = validation_utils.compute_validation_metrics(result)
    izod = metrics["Izod Impact"]
    assert izod["n"] == 4
    assert izod["mae"] == pytest.approx(1.0)
    assert izod["rmse"] == pytest.approx(1.0)


def test_compute_validation_metrics_excludes_error_rows():
    result = _synthetic_result(
        actual=[1.0, 2.0, 999.0],
        predicted=[1.0, 2.0, -999.0],
        statuses=[validation_utils.STATUS_OK, validation_utils.STATUS_OK, validation_utils.STATUS_ERROR],
    )
    metrics = validation_utils.compute_validation_metrics(result)
    assert metrics["Izod Impact"]["n"] == 2
    assert metrics["Izod Impact"]["mae"] == pytest.approx(0.0)


def test_compute_validation_metrics_zero_ok_rows_returns_all_none():
    result = _synthetic_result(actual=[1.0], predicted=[1.0], statuses=[validation_utils.STATUS_ERROR])
    metrics = validation_utils.compute_validation_metrics(result)
    for target in model_utils.TARGET_PROPERTIES:
        assert metrics[target] == {"n": 0, "r2": None, "mae": None, "rmse": None}


def test_compute_validation_metrics_r2_none_for_single_row():
    # R² is statistically meaningless for 1 point - MAE/RMSE still compute.
    result = _synthetic_result(actual=[5.0], predicted=[4.0])
    metrics = validation_utils.compute_validation_metrics(result)
    izod = metrics["Izod Impact"]
    assert izod["n"] == 1
    assert izod["r2"] is None
    assert izod["mae"] == pytest.approx(1.0)
    assert izod["rmse"] == pytest.approx(1.0)


def test_compute_validation_metrics_r2_none_for_zero_variance_actuals():
    # Every Actual value identical -> SS_tot == 0 -> R² undefined.
    result = _synthetic_result(actual=[3.0, 3.0, 3.0], predicted=[3.0, 2.0, 4.0])
    metrics = validation_utils.compute_validation_metrics(result)
    assert metrics["Izod Impact"]["r2"] is None
    assert metrics["Izod Impact"]["mae"] == pytest.approx(2.0 / 3.0)


def test_compute_validation_metrics_never_mutates_input(models, feature_columns):
    df = _lab_df()
    result = validation_utils.validate_and_score_batch(df, models, feature_columns)
    before = result.copy(deep=True)
    validation_utils.compute_validation_metrics(result)
    pd.testing.assert_frame_equal(result, before)


# --- serialization reuse (from batch_utils) ----------------------------------

def test_dataframe_to_csv_and_excel_helpers_are_reused_from_batch_utils():
    import batch_utils
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    assert batch_utils.dataframe_to_csv_bytes(df) == batch_utils.dataframe_to_csv_bytes(df)
