"""End-to-end UI tests using Streamlit's AppTest harness, driving the real
app.py + pages_ui/* against the real committed model artifact. These are
UI/navigation/rendering tests; the underlying prediction math is covered
in more depth by test_model_utils.py and test_batch_utils.py.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).parent.parent


def _nav_button(at, key):
    return next(b for b in at.sidebar.button if b.key == key)


@pytest.fixture()
def at():
    (REPO_ROOT / "predictions_log.csv").unlink(missing_ok=True)
    app_test = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=30)
    app_test.run()
    assert not app_test.exception, f"Initial run raised: {app_test.exception}"
    return app_test


# --- Dashboard -------------------------------------------------------------

def test_dashboard_is_default_page_and_shows_performance_cards(at):
    assert at.session_state["active_page"] == "dashboard"
    assert any("Dashboard" in t.value for t in at.title)

    card_values = [m for m in at.markdown if "pp-card-value" in m.value]
    joined = " ".join(m.value for m in card_values)
    assert "0.974" in joined  # Izod Impact R²
    assert "0.574" in joined  # Tensile Modulus R²
    assert "0.599" in joined  # Flexural Modulus R²

    assert any("11 of 31 trained grades exposed" in c.value for c in at.sidebar.caption)


def test_dashboard_quick_action_navigates_to_single(at):
    at.button(key="dash_go_single").click().run()
    assert at.session_state["active_page"] == "single"


def test_dashboard_shows_model_status_and_grade_count(at):
    values = " ".join(m.value for m in at.markdown if "pp-card-value" in m.value)
    assert "Loaded" in values
    assert "11" in values and "of 31 trained" in values


# --- Model Information -------------------------------------------------------

def test_model_information_page_shows_targets_and_grades(at):
    _nav_button(at, "nav_model_info").click().run()
    assert at.session_state["active_page"] == "model_info"
    assert not at.exception, f"App raised on Model Information: {at.exception}"

    assert any("Model Information" in t.value for t in at.title)

    table_html = " ".join(m.value for m in at.markdown if "pp-info-table" in m.value)
    # targets & their real R² values are present
    assert "Izod Impact" in table_html and "0.974" in table_html
    assert "Tensile Modulus" in table_html and "0.574" in table_html
    assert "Flexural Modulus" in table_html and "0.599" in table_html
    # every supported grade is listed with its derived family
    for grade in ["CA0900BM", "CB4848MO", "CB8248MO"]:
        assert grade in table_html
    assert "HECO" in table_html


# --- Single Prediction -------------------------------------------------------
#
# Grade Family is now auto-derived from the grade prefix and shown
# read-only (see pages_ui/single_prediction.py::_family_display). Every
# grade in SUPPORTED_GRADES starts with "C" (-> HECO), so these AppTest
# scenarios can only exercise the HECO/C2-required path through the real
# selectbox. The HOMO/RACO branches (C2 auto-disable/default) are real
# and tested directly against pages_ui.single_prediction's helper
# function in tests/test_pages_ui.py, using real HOMO/RACO grade codes
# from the model's full trained schema that aren't yet exposed here.

def test_single_prediction_default_matches_known_baseline(at):
    _nav_button(at, "nav_single").click().run()
    assert at.session_state["active_page"] == "single"

    grade_box = next(s for s in at.selectbox if s.label == "Grade")
    assert grade_box.value == "CB4848MO"

    family_display = next(t for t in at.text_input if t.label == "Grade family")
    assert family_display.value == "HECO — High Impact Copolymer (ICP)"
    assert family_display.disabled is True

    c2_input = next(n for n in at.number_input if n.label == "C2, wt%")
    assert c2_input.value == 8.6
    assert c2_input.disabled is False

    run_button = next(b for b in at.button if b.label == "Run Prediction")
    run_button.click().run()
    assert not at.exception, f"App raised after Run Prediction: {at.exception}"

    values = {m.value for m in at.markdown if "pp-card-value" in m.value}
    joined = " ".join(values)
    assert "7.71" in joined and "kJ/m²" in joined
    assert "1371" in joined and "MPa" in joined
    assert "1388" in joined

    log_path = REPO_ROOT / "predictions_log.csv"
    assert log_path.exists()


def test_single_prediction_family_reflects_selected_grade(at):
    _nav_button(at, "nav_single").click().run()

    grade_box = next(s for s in at.selectbox if s.label == "Grade")
    grade_box.set_value("CA0900BM").run()

    family_display = next(t for t in at.text_input if t.label == "Grade family")
    assert family_display.value == "HECO — High Impact Copolymer (ICP)"


# --- Batch Prediction --------------------------------------------------------

BATCH_CSV = """Production Date,Batch Number,Grade,MFR,XS,C2,Grade Family
2026-08-17,B-2026-0817-01,CB4848MO,48.0,16.0,8.6,HECO
2026-08-18,B-2026-0818-04,CA0900BM,12.5,4.2,1.1,
2026-08-19,B-2026-0819-02,CB2000GT,30.0,10.0,5.0,HECO
"""


def test_batch_prediction_preserves_production_date_and_batch_number(at):
    _nav_button(at, "nav_batch").click().run()
    assert at.session_state["active_page"] == "batch"

    uploader = at.get("file_uploader")[0]
    uploader.upload("weekly_batch.csv", BATCH_CSV.encode("utf-8"), "text/csv")
    at.run()

    run_button = next(b for b in at.button if b.label == "Run Batch Predictions")
    run_button.click().run()
    assert not at.exception, f"App raised after batch run: {at.exception}"

    # Summary is rendered as stat tiles (Total / Succeeded / Failed), not
    # a sentence - see pages_ui/batch_prediction.py.
    stat_values = [m.value for m in at.markdown if "pp-stat-n" in m.value]
    joined_stats = " ".join(stat_values)
    assert ">3<" in joined_stats  # total rows
    assert ">2<" in joined_stats  # succeeded
    assert ">1<" in joined_stats  # failed

    warning_texts = [w.value for w in at.warning]
    assert any("1 row(s) failed validation" in t for t in warning_texts), warning_texts

    result_df = at.dataframe[0].value
    assert list(result_df["Production Date"]) == ["2026-08-17", "2026-08-18", "2026-08-19"]
    assert list(result_df["Batch Number"]) == [
        "B-2026-0817-01", "B-2026-0818-04", "B-2026-0819-02",
    ]
    assert list(result_df.columns)[:2] == ["Production Date", "Batch Number"]

    download_labels = [d.label for d in at.get("download_button")]
    assert "Download results (CSV)" in download_labels
    assert "Download results (Excel)" in download_labels
