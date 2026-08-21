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
    df = pd.DataFrame({"Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [None]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "C2 is required" in result.loc[0, batch_utils.ERROR_COLUMN]


def test_missing_c2_column_entirely_is_error_when_no_family_column(models, feature_columns):
    df = pd.DataFrame({"Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0]})
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "C2 is required" in result.loc[0, batch_utils.ERROR_COLUMN]


# --- Grade Family logic --------------------------------------------------

def test_homo_family_with_blank_c2_defaults_to_zero(models, feature_columns):
    df = pd.DataFrame({
        "Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [None],
        "Grade Family": ["HOMO"],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)

    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK
    expected = model_utils.predict_all(models, feature_columns, VALID_GRADE_1, 48.0, 16.0, 0.0)
    assert result.loc[0, "Predicted Izod Impact"] == pytest.approx(expected["Izod Impact"])


def test_homo_family_with_explicit_c2_uses_given_value(models, feature_columns):
    df = pd.DataFrame({
        "Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [2.0],
        "Grade Family": ["HOMO"],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)

    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK
    expected = model_utils.predict_all(models, feature_columns, VALID_GRADE_1, 48.0, 16.0, 2.0)
    assert result.loc[0, "Predicted Izod Impact"] == pytest.approx(expected["Izod Impact"])


@pytest.mark.parametrize("family", ["RACO", "HECO", "ICP", "raco", "heco"])
def test_non_homo_family_with_blank_c2_is_error(models, feature_columns, family):
    df = pd.DataFrame({
        "Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [None],
        "Grade Family": [family],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "C2 is required" in result.loc[0, batch_utils.ERROR_COLUMN]


def test_invalid_family_value_is_error(models, feature_columns):
    df = pd.DataFrame({
        "Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [8.6],
        "Grade Family": ["SOMETHING_ELSE"],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "Invalid Grade Family" in result.loc[0, batch_utils.ERROR_COLUMN]


def test_blank_family_value_with_family_column_present_is_error(models, feature_columns):
    df = pd.DataFrame({
        "Grade": [VALID_GRADE_1], "MFR": [48.0], "XS": [16.0], "C2": [8.6],
        "Grade Family": [None],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    assert result.loc[0, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert "Grade Family column present but value is missing" in result.loc[0, batch_utils.ERROR_COLUMN]


# --- mixed realistic batch -----------------------------------------------

def test_mixed_batch_realistic(models, feature_columns):
    df = pd.DataFrame({
        "Grade": [VALID_GRADE_1, VALID_GRADE_2, UNTRAINED_GRADE, VALID_GRADE_1, VALID_GRADE_1],
        "MFR": [48.0, 12.5, 30.0, 999.0, 48.0],
        "XS": [16.0, 4.2, 10.0, 16.0, 16.0],
        "C2": [8.6, 1.1, 5.0, 8.6, None],
        "Grade Family": ["RACO", "HOMO", "HECO", "RACO", "HOMO"],
    })
    result = batch_utils.validate_and_predict_batch(df, models, feature_columns)
    summary = batch_utils.summarize_batch_results(result)

    assert summary["total"] == 5
    assert summary["ok"] == 3  # rows 0, 1, 4
    assert summary["errors"] == 2  # row 2 (bad grade), row 3 (MFR out of range)
    assert result.loc[2, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert result.loc[3, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_ERROR
    assert result.loc[4, batch_utils.STATUS_COLUMN] == batch_utils.STATUS_OK  # HOMO, blank C2 -> 0.0


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
