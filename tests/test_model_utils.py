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


def test_supported_grades_content_and_order():
    # Regression guard: the original 11 HECO grades must keep their
    # original list positions - the Single Prediction page's default
    # selectbox index (8 = CB4848MO) depends on this exact order.
    # HB3500GP (HOMO) and RB4545MO (RACO) were appended afterward, once
    # verified present in the real model schema (see
    # test_derive_grade_family_by_prefix and test_batch_utils.py).
    assert model_utils.SUPPORTED_GRADES == [
        "CA0900BM", "CB0900MO", "CB1248MO", "CB1640MO", "CB1849MO",
        "CB3000GT", "CB3648MO", "CB4048MO", "CB4848MO", "CB6448MO", "CB8248MO",
        "HB3500GP", "RB4545MO",
    ]
    assert model_utils.SUPPORTED_GRADES[8] == "CB4848MO"


def test_supported_grades_have_a_matching_trained_column_for_every_target(artifact):
    # Every SUPPORTED_GRADES entry must have a real Grade_<code> column in
    # every target's trained schema - this is what makes it safe to expose
    # in the UI at all.
    feature_columns = artifact["feature_columns"]
    for grade in model_utils.SUPPORTED_GRADES:
        for target in model_utils.TARGET_PROPERTIES:
            assert f"Grade_{grade}" in feature_columns[target], (
                f"{grade} is missing a trained column for {target}"
            )


def test_trained_grades_returns_all_31_real_trained_grades(artifact):
    # Source of truth for Batch Prediction / Model Validation: every
    # grade with a real Grade_<code> column in every target's schema,
    # independent of the curated SUPPORTED_GRADES UI allowlist.
    grades = model_utils.trained_grades(artifact["feature_columns"])
    assert len(grades) == 31
    assert grades == sorted(grades)
    for grade in ["CA0342EX", "HB3500GP", "RB4545MO", "PP0102TR"]:
        assert grade in grades
    # SUPPORTED_GRADES is a strict subset - never wider than the real schema.
    assert set(model_utils.SUPPORTED_GRADES) <= set(grades)
    # the classic untrained grade used elsewhere in the suite must stay absent
    assert UNTRAINED_GRADE not in grades


def test_trained_grades_only_includes_grades_common_to_every_target(artifact):
    feature_columns = artifact["feature_columns"]
    grades = model_utils.trained_grades(feature_columns)
    for grade in grades:
        for target in model_utils.TARGET_PROPERTIES:
            assert f"Grade_{grade}" in feature_columns[target]


def test_trained_grades_excludes_a_grade_missing_from_one_target():
    feature_columns = {
        "Izod Impact": ["MFR", "XS", "C2", "Grade_AAA", "Grade_BBB"],
        "Tensile Modulus": ["MFR", "XS", "C2", "Grade_AAA"],
        "Flexural Modulus": ["MFR", "XS", "C2", "Grade_AAA", "Grade_BBB"],
    }
    # Grade_BBB is missing from Tensile Modulus's schema, so it must not
    # be considered "trained" even though 2 of 3 targets have it -
    # predict_all would fail for it via Tensile Modulus.
    assert model_utils.trained_grades(feature_columns) == ["AAA"]


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
    "HB3500GP", "RB4545MO",
])
def test_predict_all_works_for_every_ui_grade(artifact, grade):
    predictions = model_utils.predict_all(
        artifact["models"], artifact["feature_columns"], grade, 48.0, 16.0, 8.6
    )
    assert all(value == value for value in predictions.values()), "prediction is NaN"  # NaN != NaN


def test_izod_impact_unit_is_kj_per_m2():
    assert model_utils.IZOD_IMPACT_UNIT == "kJ/m²"


def test_family_labels_are_exact():
    assert model_utils.FAMILY_LABELS[model_utils.FAMILY_HOMO] == "HOMO — Homopolymer"
    assert model_utils.FAMILY_LABELS[model_utils.FAMILY_RACO] == "RACO — Random Copolymer"
    assert model_utils.FAMILY_LABELS[model_utils.FAMILY_HECO] == "HECO — High Impact Copolymer (ICP)"


# Real grade codes drawn from the model's full 31-grade trained schema
# (see tests/test_batch_utils.py's UNTRAINED_GRADE note) - used here to
# exercise all three prefix rules even though only the "C" ones are in
# SUPPORTED_GRADES today.
@pytest.mark.parametrize("grade,expected_family", [
    ("CB4848MO", model_utils.FAMILY_HECO),
    ("CB3000GT", model_utils.FAMILY_HECO),
    ("CA0900BM", model_utils.FAMILY_HECO),
    ("RB4545MO", model_utils.FAMILY_RACO),
    ("RA0342EX", model_utils.FAMILY_RACO),
    ("HB3500GP", model_utils.FAMILY_HOMO),
    ("HB6540MO", model_utils.FAMILY_HOMO),
])
def test_derive_grade_family_by_prefix(grade, expected_family):
    assert model_utils.derive_grade_family(grade) == expected_family


def test_derive_grade_family_is_case_insensitive_on_prefix():
    assert model_utils.derive_grade_family("cb4848mo") == model_utils.FAMILY_HECO


def test_derive_grade_family_rejects_empty_grade():
    with pytest.raises(ValueError, match="required"):
        model_utils.derive_grade_family("")


def test_derive_grade_family_rejects_unrecognized_prefix():
    with pytest.raises(ValueError, match="starts with"):
        model_utils.derive_grade_family("PP0102TR")


@pytest.mark.parametrize("raw,expected", [
    ("HOMO", model_utils.FAMILY_HOMO),
    ("homo", model_utils.FAMILY_HOMO),
    ("RACO", model_utils.FAMILY_RACO),
    ("HECO", model_utils.FAMILY_HECO),
    ("ICP", model_utils.FAMILY_HECO),
    ("icp", model_utils.FAMILY_HECO),
])
def test_normalize_family_input_recognizes_valid_tokens_and_icp_synonym(raw, expected):
    assert model_utils.normalize_family_input(raw) == expected


def test_normalize_family_input_rejects_unrecognized_token():
    assert model_utils.normalize_family_input("SOMETHING_ELSE") is None


def test_resolve_c2_for_family_homo_always_zero():
    assert model_utils.resolve_c2_for_family(model_utils.FAMILY_HOMO, 8.6) == 0.0
    assert model_utils.resolve_c2_for_family(model_utils.FAMILY_HOMO, 0.0) == 0.0


@pytest.mark.parametrize("family", [
    None, model_utils.FAMILY_RACO, model_utils.FAMILY_HECO, model_utils.FAMILY_ICP,
])
def test_resolve_c2_for_family_passthrough_for_non_homo(family):
    assert model_utils.resolve_c2_for_family(family, 8.6) == 8.6
    assert model_utils.resolve_c2_for_family(family, 0.0) == 0.0


def test_resolve_c2_matches_predict_all_for_homo(artifact):
    # The single-prediction UI resolves C2 via resolve_c2_for_family before
    # calling predict_all. Confirm that path produces the exact same
    # prediction as calling predict_all with c2=0.0 directly.
    resolved = model_utils.resolve_c2_for_family(model_utils.FAMILY_HOMO, 8.6)
    via_helper = model_utils.predict_all(
        artifact["models"], artifact["feature_columns"], VALID_GRADE, 48.0, 16.0, resolved
    )
    direct = model_utils.predict_all(
        artifact["models"], artifact["feature_columns"], VALID_GRADE, 48.0, 16.0, 0.0
    )
    assert via_helper.keys() == direct.keys()
    for target in via_helper:
        assert via_helper[target] == pytest.approx(direct[target])


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
