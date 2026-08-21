"""Tests for model_utils.py, run against the real committed model artifact
(no mocking of the sklearn models themselves - this is a decision-support
tool and we want to know if the real artifact and code disagree).
"""

import csv
from pathlib import Path

import pytest

import model_utils

REPO_ROOT = Path(__file__).parent.parent
MODEL_PATH = REPO_ROOT / "pp_virtual_lab_models .joblib"

# A grade known to be present in the trained schema (see GRADE_OPTIONS in
# app.py) and one known to be absent (the reason it was removed from the
# selectable list - see git history).
VALID_GRADE = "CB4848MO"
UNTRAINED_GRADE = "CB2000GT"


@pytest.fixture(scope="module")
def artifact():
    return model_utils.load_model_artifact(MODEL_PATH)


def test_supported_grades_unchanged_by_refactor():
    # Regression guard: SUPPORTED_GRADES was moved from an inline
    # GRADE_OPTIONS list in app.py into model_utils.py (P3.1 batch
    # prediction work). Content and order must be unchanged, since
    # app.py's default selectbox index depends on this exact order.
    assert model_utils.SUPPORTED_GRADES == [
        "CA0900BM", "CB0900MO", "CB1248MO", "CB1640MO", "CB1849MO",
        "CB3000GT", "CB3648MO", "CB4048MO", "CB4848MO", "CB6448MO", "CB8248MO",
    ]
    assert model_utils.SUPPORTED_GRADES[8] == "CB4848MO"


def test_load_model_artifact_has_expected_shape(artifact):
    assert set(model_utils.TARGET_PROPERTIES) <= artifact["models"].keys()
    assert set(model_utils.TARGET_PROPERTIES) <= artifact["feature_columns"].keys()
    assert set(model_utils.TARGET_PROPERTIES) <= artifact["metrics"].keys()


def test_load_model_artifact_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError):
        model_utils.load_model_artifact(missing_path)


def test_load_model_artifact_rejects_malformed_artifact(tmp_path):
    import joblib

    bad_path = tmp_path / "bad.joblib"
    joblib.dump({"models": {}}, bad_path)  # missing feature_columns / metrics

    with pytest.raises(ValueError):
        model_utils.load_model_artifact(bad_path)


@pytest.mark.parametrize("target", model_utils.TARGET_PROPERTIES)
def test_build_feature_row_matches_model_schema(artifact, target):
    feature_columns = artifact["feature_columns"]
    model = artifact["models"][target]

    row = model_utils.build_feature_row(target, feature_columns, VALID_GRADE, 48.0, 16.0, 8.6)

    assert list(row.columns) == list(feature_columns[target])
    assert list(row.columns) == list(model.feature_names_in_)
    assert row.loc[0, "MFR"] == 48.0
    assert row.loc[0, "XS"] == 16.0
    assert row.loc[0, "C2"] == 8.6
    assert row.loc[0, f"Grade_{VALID_GRADE}"] == 1.0


@pytest.mark.parametrize("target", model_utils.TARGET_PROPERTIES)
def test_build_feature_row_rejects_untrained_grade(artifact, target):
    feature_columns = artifact["feature_columns"]
    with pytest.raises(ValueError, match=UNTRAINED_GRADE):
        model_utils.build_feature_row(target, feature_columns, UNTRAINED_GRADE, 48.0, 16.0, 8.6)


def test_predict_all_returns_numeric_value_per_target(artifact):
    predictions = model_utils.predict_all(
        artifact["models"], artifact["feature_columns"], VALID_GRADE, 48.0, 16.0, 8.6
    )
    assert set(predictions.keys()) == set(model_utils.TARGET_PROPERTIES)
    for target, value in predictions.items():
        assert isinstance(value, float), f"{target} prediction is not a float: {type(value)}"


def test_predict_all_raises_for_untrained_grade(artifact):
    with pytest.raises(ValueError):
        model_utils.predict_all(
            artifact["models"], artifact["feature_columns"], UNTRAINED_GRADE, 48.0, 16.0, 8.6
        )


@pytest.mark.parametrize("grade", [
    "CA0900BM", "CB0900MO", "CB1248MO", "CB1640MO", "CB1849MO",
    "CB3000GT", "CB3648MO", "CB4048MO", "CB4848MO", "CB6448MO", "CB8248MO",
])
def test_predict_all_works_for_every_ui_grade(artifact, grade):
    predictions = model_utils.predict_all(
        artifact["models"], artifact["feature_columns"], grade, 48.0, 16.0, 8.6
    )
    assert all(value == value for value in predictions.values()), "prediction is NaN"  # NaN != NaN


def test_metrics_are_read_from_artifact_not_hardcoded(artifact):
    metrics = artifact["metrics"]
    for target in model_utils.TARGET_PROPERTIES:
        assert 0.0 <= metrics[target]["r2"] <= 1.0
        assert metrics[target]["mae"] >= 0.0
        assert metrics[target]["rows"] > 0


def test_log_prediction_creates_file_with_header_and_row(tmp_path):
    log_path = tmp_path / "predictions_log.csv"
    predictions = {"Izod Impact": 7.71, "Tensile Modulus": 1371.0, "Flexural Modulus": 1388.0}

    model_utils.log_prediction(log_path, VALID_GRADE, 48.0, 16.0, 8.6, predictions)

    assert log_path.exists()
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == model_utils.LOG_COLUMNS
    assert rows[1][1] == VALID_GRADE
    assert rows[1][5] == "7.71"


def test_log_prediction_appends_without_duplicating_header(tmp_path):
    log_path = tmp_path / "predictions_log.csv"
    predictions = {"Izod Impact": 7.71, "Tensile Modulus": 1371.0, "Flexural Modulus": 1388.0}

    model_utils.log_prediction(log_path, VALID_GRADE, 48.0, 16.0, 8.6, predictions)
    model_utils.log_prediction(log_path, "CA0900BM", 12.5, 4.2, 1.1, predictions)

    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == model_utils.LOG_COLUMNS
    assert len(rows) == 3  # header + 2 data rows
