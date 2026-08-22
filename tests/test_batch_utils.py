"""Tests for batch_utils.py, run against the real committed model artifact."""

import io

import pandas as pd
import pytest

import batch_utils
import model_utils

REPO_ROOT = __import__("pathlib").Path(__file__).parent.parent
MODEL_PATH = REPO_ROOT / "pp_virtual_lab_models .joblib"

VALID_GRADE_1 = "CB4848MO"
VALID_GRADE_2 = "CA0900BM"
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


# --- file parsing -----------------------------------------------------

def test_parse_uploaded_csv():
    csv_bytes = b"Grade,MFR,XS,C2\nCB4848MO,48.0,16.0,8.6\n"
    df = batch_utils.parse_uploaded_file(csv_bytes, "batch.csv")
    assert list(df.columns) == ["Grade", "MFR", "XS", "C2"]
    assert len(df) == 1


def test_parse_uploaded_xlsx():
    original = pd.DataFrame({"Grade": ["CB4848MO"], "MFR": [48.0], "XS": [16.0], "C2": [8.6]})
    buffer = io.BytesIO()
    original.to_excel(buffer, index=False, engine="openpyxl")

    df = batch_utils.parse_uploaded_file(buffer.getvalue(), "batch.xlsx")
    assert list(df.columns) == ["Grade", "MFR", "XS", "C2"]
    assert df.iloc[0]["Grade"] == "CB4848MO"


def test_parse_uploaded_file_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        batch_utils.parse_uploaded_file(b"whatever", "batch.txt")


def test_parse_uploaded_file_rejects_empty_csv():
    with pytest.raises(ValueError, match="Could not parse"):
        batch_utils.parse_uploaded_file(b"", "batch.csv")


# --- column mapping -----------------------------------------------------

@pytest.mark.parametrize("header,expected_canonical", [
    ("Grade", "Grade"),
    ("grade", "Grade"),
    (" Grade ", "Grade"),
    ("MFR", "MFR"),
    ("MFR, g/10 min", "MFR"),
    ("XS", "XS"),
    ("XS, wt%", "XS"),
    ("C2", "C2"),
    ("C2, wt%", "C2"),
    ("Grade Family", "Grade Family"),
    ("family", "Grade Family"),
])
def test_detect_column_mapping_recognizes_variants(header, expected_canonical):
    mapping = batch_utils.detect_column_mapping([header])
    assert mapping[expected_canonical] == header


def test_validate_and_predict_batch_raises_for_missing_required_columns(models, feature_columns):
    df = pd.DataFrame({"Grade": ["CB4848MO"], "MFR": [48.0]})  # XS missing
    with pytest.raises(ValueError, match="Missing required column"):
        batch_utils.validate_and_predict_batch(df, models, feature_columns)


# --- row-level validation & prediction, no Grade Family column ----------

def test_all_valid_rows_produce_predictions_matching_predict_all(models, feature_columns):
    df = pd.DataFrame({
        "Grade": [VALID_GRADE_1, VALID_GRADE_2],
        "MFR": [48.0, 12.5],
        "XS": [16.0, 4.2],
        "C2": [8.6, 1.1],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)

    assert (result[batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK).all()
    assert (result[batch_utils.ERROR_COLUMN] == "").all()

    expected_1 = model_utils.predict_all(models, feature_columns, VALID_GRADE_1, 48.0, 16.0, 8.6)
    expected_2 = model_utils.predict_all(models, feature_columns, VALID_GRADE_2, 12.5, 4.2, 1.1)
    assert result.loc[0, "Predicted Izod Impact"] == pytest.approx(expected_1["Izod Impact"])
    assert result.loc[0, "Predicted Tensile Modulus"] == pytest.approx(expected_1["Tensile Modulus"])
    assert result.loc[1, "Predicted Flexural Modulus"] == pytest.approx(expected_2["Flexural Modulus"])


def test_original_columns_preserved_and_new_columns_appended(models, feature_columns):
    df = pd.DataFrame({
        "Batch ID": ["A1"],
        "Grade": [VALID_GRADE_1],
        "MFR": [48.0],
        "XS": [16.0],
        "C2": [8.6],
        "Notes": ["sample note"],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)

    assert result.loc[0, "Batch ID"] == "A1"
    assert result.loc[0, "Notes"] == "sample note"
    expected_new_columns = [
        "Predicted Izod Impact", "Predicted Tensile Modulus", "Predicted Flexural Modulus",
        batch_utils.STATUS_COLUMN, batch_utils.ERROR_COLUMN,
    ]
    assert list(result.columns) == list(df.columns) + expected_new_columns


def test_production_date_and_batch_number_preserved_as_batch_identity(models, feature_columns):
    # Production Date and Batch Number are treated as the batch's identity
    # for future history/matching work. They're not part of the prediction
    # schema, so they must pass through completely unchanged - same
    # generic passthrough as any other extra column, verified explicitly
    # here because these two specifically matter for future matching.
    df = pd.DataFrame({
        "Production Date": ["2026-08-17", "2026-08-18"],
        "Batch Number": ["B-2026-0817-01", "B-2026-0818-04"],
        "Grade": [VALID_GRADE_1, VALID_GRADE_2],
        "MFR": [48.0, 12.5],
        "XS": [16.0, 4.2],
        "C2": [8.6, 1.1],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)

    assert list(result["Production Date"]) == ["2026-08-17", "2026-08-18"]
    assert list(result["Batch Number"]) == ["B-2026-0817-01", "B-2026-0818-04"]
    # both rows still predicted successfully - identity columns don't
    # interfere with validation or prediction
    assert (result[batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK).all()
    # and they keep their original position ahead of the prediction/status
    # columns, exactly like any other original column
    assert list(result.columns)[:2] == ["Production Date", "Batch Number"]


def test_production_date_and_batch_number_do_not_affect_predictions(models, feature_columns):
    # Same Grade/MFR/XS/C2, different Production Date and Batch Number ->
    # predictions must be identical. These columns are never read by the
    # model - only Grade/MFR/XS/C2 feed build_feature_row/predict_all.
    common = {"Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [8.6]}
    df_a = pd.DataFrame({
        "Production Date": ["2026-01-01"], "Batch Number": ["B-AAA"], **common,
    })
    df_b = pd.DataFrame({
        "Production Date": ["2099-12-31"], "Batch Number": ["B-ZZZ-DIFFERENT"], **common,
    })

    result_a = batch_utils.validate_and_predict_batch(df_a, models, feature_columns)
    result_b = batch_utils.validate_and_predict_batch(df_b, models, feature_columns)

    for column in [
        "Predicted Izod Impact", "Predicted Tensile Modulus", "Predicted Flexural Modulus",
    ]:
        assert result_a.loc[0, column] == pytest.approx(result_b.loc[0, column])


def test_unsupported_grade_row_is_isolated_error(models, feature_columns):
    df = pd.DataFrame({
        "Grade": [UNTRAINED_GRADE, VALID_GRADE_1],
        "MFR": [48.0, 48.0],
        "XS": [16.0, 16.0],
        "C2": [8.6, 8.6],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)

    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "Unsupported grade" in result.loc[0, batch_utils.ERROR_COLUMN]
    assert pd.isna(result.loc[0, "Predicted Izod Impact"])

    # the other row is unaffected
    assert result.loc[1, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK
    assert not pd.isna(result.loc[1, "Predicted Izod Impact"])


# --- Batch accepts every real trained grade, not just SUPPORTED_GRADES ----
#
# Batch Prediction's grade allowlist is model_utils.trained_grades(), the
# model's real 31-grade schema - deliberately wider than the 13-grade
# SUPPORTED_GRADES UI allowlist used by Single Prediction. These grades
# are drawn from the model's real trained schema but are NOT in
# SUPPORTED_GRADES, confirming batch doesn't wrongly narrow to that list.

@pytest.mark.parametrize("grade,family", [
    ("CA0342EX", "HECO"),   # trained HECO grade absent from SUPPORTED_GRADES
    ("HB0356FR", "HOMO"),   # trained HOMO grade absent from SUPPORTED_GRADES
    ("RA0342EX", "RACO"),   # trained RACO grade absent from SUPPORTED_GRADES
])
def test_batch_accepts_trained_grades_outside_supported_grades(models, feature_columns, grade, family):
    assert grade not in model_utils.SUPPORTED_GRADES
    c2 = 0.0 if family == "HOMO" else 8.6
    df = pd.DataFrame({"Grade": [grade], "MFR": [48.0], "XS": [16.0], "C2": [c2]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK, result.loc[0, batch_utils.ERROR_COLUMN]
    assert not pd.isna(result.loc[0, "Predicted Izod Impact"])


def test_batch_rejects_grade_with_unrecognized_family_prefix_as_row_level_error(models, feature_columns):
    # PP0102TR is a real trained grade (present in trained_grades()), but
    # its "P" prefix has no HOMO/RACO/HECO mapping - derive_grade_family
    # raises, and that must surface as an isolated row error, never a
    # crash that stops the rest of the batch.
    ppprefixed_grade = "PP0102TR"
    assert ppprefixed_grade in model_utils.trained_grades(feature_columns)
    df = pd.DataFrame({
        "Grade": [ppprefixed_grade, VALID_GRADE_1],
        "MFR": [48.0, 48.0],
        "XS": [16.0, 16.0],
        "C2": [8.6, 8.6],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "starts with" in result.loc[0, batch_utils.ERROR_COLUMN]
    assert pd.isna(result.loc[0, "Predicted Izod Impact"])
    # the rest of the batch is unaffected
    assert result.loc[1, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK


def test_batch_still_rejects_grade_absent_from_real_trained_schema(models, feature_columns):
    # UNTRAINED_GRADE has no Grade_<code> column anywhere in the real
    # artifact - widening the allowlist to trained_grades() must not
    # accidentally accept it.
    assert UNTRAINED_GRADE not in model_utils.trained_grades(feature_columns)
    df = pd.DataFrame({"Grade": [UNTRAINED_GRADE], "MFR": [48.0], "XS": [16.0], "C2": [8.6]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "Unsupported grade" in result.loc[0, batch_utils.ERROR_COLUMN]


def test_missing_grade_is_error(models, feature_columns):
    df = pd.DataFrame({"Grade": [None], "MFR": [48.0], "XS": [16.0], "C2": [8.6]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "Grade is missing" in result.loc[0, batch_utils.ERROR_COLUMN]


@pytest.mark.parametrize("mfr,xs,c2,expected_fragment", [
    (-1.0, 16.0, 8.6, "MFR must be between"),
    (200.0, 16.0, 8.6, "MFR must be between"),
    (48.0, -5.0, 8.6, "XS must be between"),
    (48.0, 50.0, 8.6, "XS must be between"),
    (48.0, 16.0, -1.0, "C2 must be between"),
    (48.0, 16.0, 100.0, "C2 must be between"),
])
def test_out_of_range_values_are_errors(models, feature_columns, mfr, xs, c2, expected_fragment):
    df = pd.DataFrame({"Grade": [VALID_GRADE_1], "MFR": [mfr], "XS": [xs], "C2": [c2]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert expected_fragment in result.loc[0, batch_utils.ERROR_COLUMN]


def test_non_numeric_mfr_is_error(models, feature_columns):
    df = pd.DataFrame({"Grade": [VALID_GRADE_1], "MFR": ["not-a-number"], "XS": [16.0], "C2": [8.6]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "MFR is not numeric" in result.loc[0, batch_utils.ERROR_COLUMN]


def test_missing_c2_is_error_when_no_family_column(models, feature_columns):
    # CB4848MO derives to HECO regardless of whether a Grade Family column
    # is present, and HECO always requires C2.
    df = pd.DataFrame({"Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [None]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "C2 is required" in result.loc[0, batch_utils.ERROR_COLUMN]


def test_missing_c2_column_entirely_is_error_when_no_family_column(models, feature_columns):
    df = pd.DataFrame({"Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "C2 is required" in result.loc[0, batch_utils.ERROR_COLUMN]


# --- Grade Family logic: derived from grade prefix ------------------------
#
# SUPPORTED_GRADES includes one grade per family - CB4848MO (HECO),
# HB3500GP (HOMO), RB4545MO (RACO) - so all three branches are exercised
# directly through this function's real public API, no monkeypatching.

def test_family_auto_derived_when_column_absent_succeeds(models, feature_columns):
    df = pd.DataFrame({"Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [8.6]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK


def test_family_auto_derived_when_cell_blank_succeeds(models, feature_columns):
    # A blank Grade Family cell (column present) is no longer an error -
    # it's simply derived from the grade instead.
    df = pd.DataFrame({
        "Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [8.6],
        "Grade Family": [None],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK


@pytest.mark.parametrize("family_value", ["HECO", "heco", "ICP", "icp"])
def test_supplied_family_matching_grade_prefix_succeeds(models, feature_columns, family_value):
    # HECO, or its ICP synonym, is consistent with CB4848MO's "C" prefix.
    df = pd.DataFrame({
        "Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [8.6],
        "Grade Family": [family_value],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK


@pytest.mark.parametrize("family_value", ["HOMO", "RACO"])
def test_supplied_family_conflicting_with_grade_prefix_is_error(models, feature_columns, family_value):
    # CB4848MO's prefix derives HECO. Supplying HOMO or RACO for it is
    # inconsistent and must be flagged, not silently corrected.
    df = pd.DataFrame({
        "Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [8.6],
        "Grade Family": [family_value],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "does not match grade" in result.loc[0, batch_utils.ERROR_COLUMN]
    assert pd.isna(result.loc[0, "Predicted Izod Impact"])


def test_invalid_family_value_is_error(models, feature_columns):
    df = pd.DataFrame({
        "Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [8.6],
        "Grade Family": ["SOMETHING_ELSE"],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "Invalid Grade Family" in result.loc[0, batch_utils.ERROR_COLUMN]


def test_homo_grade_blank_c2_defaults_to_zero(models, feature_columns):
    homo_grade = "HB3500GP"
    df = pd.DataFrame({"Grade": [homo_grade], "MFR": [48.0], "XS": [16.0], "C2": [None]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)

    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK
    expected = model_utils.predict_all(models, feature_columns, homo_grade, 48.0, 16.0, 0.0)
    assert result.loc[0, "Predicted Izod Impact"] == pytest.approx(expected["Izod Impact"])


def test_homo_grade_explicit_c2_uses_given_value(models, feature_columns):
    homo_grade = "HB3500GP"
    df = pd.DataFrame({"Grade": [homo_grade], "MFR": [48.0], "XS": [16.0], "C2": [2.0]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)

    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK
    expected = model_utils.predict_all(models, feature_columns, homo_grade, 48.0, 16.0, 2.0)
    assert result.loc[0, "Predicted Izod Impact"] == pytest.approx(expected["Izod Impact"])


def test_homo_grade_family_mismatch_is_still_an_error(models, feature_columns):
    # HB3500GP is HOMO by prefix - supplying RACO for it must still be
    # flagged, exactly like the reverse case for a HECO grade.
    df = pd.DataFrame({
        "Grade": ["HB3500GP"], "MFR": [48.0], "XS": [16.0], "C2": [4.0],
        "Grade Family": ["RACO"],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "does not match grade" in result.loc[0, batch_utils.ERROR_COLUMN]


def test_raco_grade_matching_family_succeeds(models, feature_columns):
    raco_grade = "RB4545MO"
    df = pd.DataFrame({
        "Grade": [raco_grade], "MFR": [48.0], "XS": [16.0], "C2": [4.0],
        "Grade Family": ["RACO"],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK


def test_raco_grade_blank_c2_is_error(models, feature_columns):
    raco_grade = "RB4545MO"
    df = pd.DataFrame({"Grade": [raco_grade], "MFR": [48.0], "XS": [16.0], "C2": [None]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "C2 is required" in result.loc[0, batch_utils.ERROR_COLUMN]


# --- mixed realistic batch -----------------------------------------------

def test_mixed_batch_realistic(models, feature_columns):
    df = pd.DataFrame({
        "Grade": [VALID_GRADE_1, VALID_GRADE_2, UNTRAINED_GRADE, VALID_GRADE_1, VALID_GRADE_1],
        "MFR": [48.0, 12.5, 30.0, 999.0, 48.0],
        "XS": [16.0, 4.2, 10.0, 16.0, 16.0],
        "C2": [8.6, 1.1, 5.0, 8.6, 3.0],
        "Grade Family": ["HECO", "ICP", "HECO", "HECO", None],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    summary = batch_utils.summarize_batch_results(result)

    assert summary["total"] == 5
    assert summary["ok"] == 3  # rows 0, 1, 4
    assert summary["errors"] == 2  # row 2 (bad grade), row 3 (MFR out of range)
    assert result.loc[2, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert result.loc[3, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert result.loc[4, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK  # blank family cell -> auto-derived HECO


# --- serialization ---------------------------------------------------------

def test_dataframe_to_csv_bytes_round_trips():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    csv_bytes = batch_utils.dataframe_to_csv_bytes(df)
    round_tripped = pd.read_csv(io.BytesIO(csv_bytes))
    pd.testing.assert_frame_equal(df, round_tripped)


def test_dataframe_to_excel_bytes_round_trips():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    excel_bytes = batch_utils.dataframe_to_excel_bytes(df)
    round_tripped = pd.read_excel(io.BytesIO(excel_bytes), engine="openpyxl")
    pd.testing.assert_frame_equal(df, round_tripped)
