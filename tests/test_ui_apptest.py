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
    (REPO_ROOT / "lab_validation_archive.csv").unlink(missing_ok=True)
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

    assert any("28 of 31 trained grades exposed" in c.value for c in at.sidebar.caption)


def test_dashboard_process_flow_diagram_renders_all_blocks(at):
    badges = [m.value for m in at.markdown if "pp-badge" in m.value]
    assert any("Conceptual PP Quality Workflow" in b for b in badges)

    flow_html = " ".join(m.value for m in at.markdown if "pp-process-flow" in m.value)
    for title in ["Feed", "Polymerization", "Batch", "Laboratory", "PP Virtual Lab", "Quality Results"]:
        assert title in flow_html
    # detail lines from the spec are present
    assert "Propylene" in flow_html and "Comonomer" in flow_html
    assert "Grade · MFR · XS · C2" in flow_html
    assert "Production Date" in flow_html and "Batch Number" in flow_html
    assert "Izod Impact" in flow_html and "Tensile Modulus" in flow_html and "Flexural Modulus" in flow_html
    # live R² values (not hardcoded) appear in the PP Virtual Lab block
    assert "0.974" in flow_html and "0.574" in flow_html and "0.599" in flow_html
    assert "✅ Loaded" in flow_html
    # hover tooltips are present as data-tooltip attributes, one per block
    assert flow_html.count("data-tooltip=") == 6
    # the "you are here" block is visually distinguished
    assert "pp-you-are-here" in flow_html

    caption_texts = [c.value for c in at.caption]
    assert any("Symbolic representation" in t and "P&ID" in t for t in caption_texts)
    assert any("Process" in t and "AI Prediction" in t and "Quality" in t for t in caption_texts)


def test_dashboard_process_flow_does_not_break_model_performance_cards(at):
    # The new diagram must not push out or interfere with the existing
    # Model Performance section below it.
    joined = " ".join(m.value for m in at.markdown if "pp-card-value" in m.value)
    assert "0.974" in joined and "0.574" in joined and "0.599" in joined


def test_dashboard_quick_action_navigates_to_single(at):
    at.button(key="dash_go_single").click().run()
    assert at.session_state["active_page"] == "single"


def test_dashboard_shows_model_status_and_grade_count(at):
    values = " ".join(m.value for m in at.markdown if "pp-card-value" in m.value)
    assert "Loaded" in values
    assert "28" in values and "of 31 trained" in values


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
    for grade in ["CA0900BM", "CB4848MO", "CB8248MO", "HB3500GP", "RB4545MO"]:
        assert grade in table_html
    assert "HECO" in table_html
    assert "HOMO" in table_html
    assert "RACO" in table_html


def test_model_information_explains_how_the_prediction_works(at):
    _nav_button(at, "nav_model_info").click().run()
    assert not at.exception, f"App raised on Model Information: {at.exception}"

    headers = [h.value for h in at.subheader]
    assert "How the Prediction Works" in headers
    assert "What affects the prediction?" in headers
    assert "The algorithm" in headers

    # the pipeline diagram shows all 5 steps in order
    flow_html = " ".join(m.value for m in at.markdown if "pp-flow" in m.value)
    for step in ["Inputs", "Feature construction", "Trained model",
                 "Independent regressions", "Output"]:
        assert step in flow_html

    # input-role table explains all 4 real inputs
    body_text = " ".join(m.value for m in at.markdown)
    assert "Melt Flow Rate" in body_text
    assert "Xylene Solubles" in body_text
    assert "Ethylene comonomer" in body_text

    # what-affects-the-prediction table correctly separates inputs from metadata
    io_html = " ".join(m.value for m in at.markdown if "pp-io-table" in m.value)
    assert "Model input" in io_html
    assert "Production Date" in io_html and "Batch Number" in io_html
    assert "Not a model input" in io_html or "Metadata only" in io_html

    # R² is explicitly framed as validation/performance, never as confidence
    assert "model validation statistic, not a confidence score" in body_text

    # the algorithm section shows the real, live hyperparameters
    algo_html = " ".join(m.value for m in at.markdown if "pp-info-table" in m.value)
    assert "n_estimators" in algo_html and "500" in algo_html
    assert "min_samples_leaf" in algo_html and "2" in algo_html
    assert "random_state" in algo_html and "42" in algo_html

    # limitations cover all four required points
    assert "not laboratory measurements" in body_text
    assert "domain of the training data" in body_text
    assert "Unsupported grades must not be predicted" in body_text
    assert "substantially lower R²" in body_text


# --- Single Prediction -------------------------------------------------------
#
# Grade Family is auto-derived from the grade prefix and shown read-only
# (see pages_ui/single_prediction.py::_family_display). SUPPORTED_GRADES
# now includes one grade per family - CB4848MO (HECO), HB3500GP (HOMO),
# RB4545MO (RACO) - so all three branches, including the HOMO C2
# auto-disable/default behavior, are exercised directly through the real
# selectbox below, not just via the pure-function tests in
# tests/test_pages_ui.py.

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


def test_single_prediction_homo_grade_disables_and_zeroes_c2(at):
    import model_utils

    _nav_button(at, "nav_single").click().run()

    grade_box = next(s for s in at.selectbox if s.label == "Grade")
    grade_box.set_value("HB3500GP").run()

    family_display = next(t for t in at.text_input if t.label == "Grade family")
    assert family_display.value == "HOMO — Homopolymer"

    c2_input = next(n for n in at.number_input if n.label == "C2, wt%")
    assert c2_input.value == 0.0
    assert c2_input.disabled is True

    run_button = next(b for b in at.button if b.label == "Run Prediction")
    run_button.click().run()
    assert not at.exception, f"App raised after Run Prediction (HOMO): {at.exception}"

    artifact = model_utils.load_model_artifact(REPO_ROOT / "pp_virtual_lab_models .joblib")
    expected = model_utils.predict_all(
        artifact["models"], artifact["feature_columns"], "HB3500GP", 48.0, 16.0, 0.0
    )
    values = " ".join(m.value for m in at.markdown if "pp-card-value" in m.value)
    assert f"{expected['Izod Impact']:.2f}" in values
    assert f"{expected['Tensile Modulus']:.0f}" in values
    assert f"{expected['Flexural Modulus']:.0f}" in values


def test_single_prediction_raco_grade_keeps_c2_enabled(at):
    import model_utils

    _nav_button(at, "nav_single").click().run()

    grade_box = next(s for s in at.selectbox if s.label == "Grade")
    grade_box.set_value("RB4545MO").run()

    family_display = next(t for t in at.text_input if t.label == "Grade family")
    assert family_display.value == "RACO — Random Copolymer"

    c2_input = next(n for n in at.number_input if n.label == "C2, wt%")
    assert c2_input.disabled is False
    assert c2_input.value == 8.6  # unchanged default, still editable

    run_button = next(b for b in at.button if b.label == "Run Prediction")
    run_button.click().run()
    assert not at.exception, f"App raised after Run Prediction (RACO): {at.exception}"

    artifact = model_utils.load_model_artifact(REPO_ROOT / "pp_virtual_lab_models .joblib")
    expected = model_utils.predict_all(
        artifact["models"], artifact["feature_columns"], "RB4545MO", 48.0, 16.0, 8.6
    )
    values = " ".join(m.value for m in at.markdown if "pp-card-value" in m.value)
    assert f"{expected['Izod Impact']:.2f}" in values
    assert f"{expected['Tensile Modulus']:.0f}" in values
    assert f"{expected['Flexural Modulus']:.0f}" in values


# --- Batch Prediction --------------------------------------------------------

BATCH_CSV = """Production Date,Batch Number,Grade,MFR,XS,C2,Grade Family
2026-08-17,B-2026-0817-01,CB4848MO,48.0,16.0,8.6,HECO
2026-08-18,B-2026-0818-04,CA0900BM,12.5,4.2,1.1,
2026-08-19,B-2026-0819-02,CB2000GT,30.0,10.0,5.0,HECO
"""


def test_batch_prediction_accepts_a_real_trained_grade_via_the_app(at):
    # CA0342EX is a real trained grade, now also included in the widened
    # SUPPORTED_GRADES (Single Prediction dropdown) - but Batch
    # Prediction validates independently via trained_grades(), not
    # SUPPORTED_GRADES (see tests/test_batch_utils.py for the case that
    # actually distinguishes the two: PP0102TR, trained but excluded
    # from SUPPORTED_GRADES). This confirms the app-level upload path
    # still works end-to-end for it.
    _nav_button(at, "nav_batch").click().run()

    csv_bytes = (
        "Grade,MFR,XS,C2\nCA0342EX,48.0,16.0,8.6\n"
    ).encode("utf-8")
    uploader = at.get("file_uploader")[0]
    uploader.upload("widened_grade.csv", csv_bytes, "text/csv")
    at.run()

    run_button = next(b for b in at.button if b.label == "Run Batch Predictions")
    run_button.click().run()
    assert not at.exception, f"App raised after batch run: {at.exception}"

    result_df = at.dataframe[0].value
    assert result_df.loc[0, "Validation Status"] == "OK"


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


# --- Model Validation ---------------------------------------------------------

VALIDATION_CSV = """Production Date,Batch Number,Grade,MFR,XS,C2,Actual Izod Impact,Actual Tensile Modulus,Actual Flexural Modulus
2026-08-17,B-2026-0817-01,CB4848MO,48.0,16.0,8.6,7.6,1360,1400
2026-08-18,B-2026-0818-04,HB3500GP,48.0,16.0,,3.1,1500,1600
2026-08-19,B-2026-0819-02,CB2000GT,30.0,10.0,5.0,5.0,1200,1200
"""


def test_model_validation_nav_button_exists_and_navigates(at):
    _nav_button(at, "nav_validation").click().run()
    assert at.session_state["active_page"] == "validation"
    assert not at.exception, f"App raised on Model Validation: {at.exception}"
    assert any("Validate against new lab data" in t.value for t in at.title)


def test_model_validation_scores_uploaded_lab_file(at):
    import model_utils

    _nav_button(at, "nav_validation").click().run()

    uploader = at.get("file_uploader")[0]
    uploader.upload("lab_results.csv", VALIDATION_CSV.encode("utf-8"), "text/csv")
    at.run()

    run_button = next(b for b in at.button if b.label == "Run Validation")
    run_button.click().run()
    assert not at.exception, f"App raised after Run Validation: {at.exception}"

    # Summary stat tiles: 3 total, 2 scored, 1 excluded (untrained grade)
    stat_values = [m.value for m in at.markdown if "pp-stat-n" in m.value]
    joined_stats = " ".join(stat_values)
    assert ">3<" in joined_stats
    assert ">2<" in joined_stats
    assert ">1<" in joined_stats

    warning_texts = [w.value for w in at.warning]
    assert any("1 row(s) failed validation" in t for t in warning_texts), warning_texts

    result_df = at.dataframe[0].value
    assert list(result_df["Production Date"]) == ["2026-08-17", "2026-08-18", "2026-08-19"]
    assert result_df.loc[0, "Validation Status"] == "OK"
    assert result_df.loc[1, "Validation Status"] == "OK"  # HOMO, blank C2 -> defaults to 0
    assert result_df.loc[2, "Validation Status"] == "ERROR"  # untrained grade

    # Predicted values for the OK rows match predict_all directly.
    artifact = model_utils.load_model_artifact(REPO_ROOT / "pp_virtual_lab_models .joblib")
    expected_0 = model_utils.predict_all(
        artifact["models"], artifact["feature_columns"], "CB4848MO", 48.0, 16.0, 8.6
    )
    assert result_df.loc[0, "Predicted Izod Impact"] == pytest.approx(expected_0["Izod Impact"])

    # New R²/MAE/RMSE table is rendered, alongside the original training metrics.
    metrics_html = " ".join(m.value for m in at.markdown if "pp-info-table" in m.value)
    assert "Training R²" in metrics_html and "New R²" in metrics_html
    assert "New RMSE" in metrics_html
    assert "0.974" in metrics_html  # original training R² for Izod Impact, unchanged

    download_labels = [d.label for d in at.get("download_button")]
    assert "Download results (CSV)" in download_labels
    assert "Download results (Excel)" in download_labels

    # Predicted vs. Actual charts rendered without raising (AppTest has
    # no chart-content inspector, so this is a smoke check).
    headers = [h.value for h in at.subheader]
    assert "Predicted vs. Actual" in headers

    # "Save to training archive" control is present, separate from the
    # download buttons - a distinct, explicit user action.
    button_labels = [b.label for b in at.button]
    assert "Save to training archive" in button_labels


def test_model_validation_rejects_file_missing_actual_columns(at):
    _nav_button(at, "nav_validation").click().run()

    csv_bytes = b"Grade,MFR,XS,C2\nCB4848MO,48.0,16.0,8.6\n"
    uploader = at.get("file_uploader")[0]
    uploader.upload("incomplete.csv", csv_bytes, "text/csv")
    at.run()

    run_button = next(b for b in at.button if b.label == "Run Validation")
    run_button.click().run()
    assert not at.exception, f"App raised on missing-column upload: {at.exception}"

    error_texts = [e.value for e in at.error]
    assert any("Missing required column" in t for t in error_texts), error_texts


def test_model_validation_save_to_training_archive_accumulates_and_is_visible_on_improvement_page(at):
    # End-to-end: run validation -> save to archive -> Model Improvement
    # reflects the accumulated count. Confirms the "progressively
    # accumulate lab data" workflow actually works across pages.
    _nav_button(at, "nav_validation").click().run()

    uploader = at.get("file_uploader")[0]
    uploader.upload("lab_results.csv", VALIDATION_CSV.encode("utf-8"), "text/csv")
    at.run()

    next(b for b in at.button if b.label == "Run Validation").click().run()
    assert not at.exception, f"App raised after Run Validation: {at.exception}"

    save_button = next(b for b in at.button if b.label == "Save to training archive")
    save_button.click().run()
    assert not at.exception, f"App raised after Save to training archive: {at.exception}"

    success_texts = [s.value for s in at.success]
    assert any("Saved 2 row(s)" in t for t in success_texts), success_texts

    _nav_button(at, "nav_model_improvement").click().run()
    assert at.session_state["active_page"] == "model_improvement"
    assert not at.exception, f"App raised on Model Improvement: {at.exception}"

    stat_values = [m.value for m in at.markdown if "pp-stat-n" in m.value]
    joined_stats = " ".join(stat_values)
    assert ">2<" in joined_stats  # 2 OK rows were saved (the untrained-grade row was excluded)


def test_model_validation_save_button_warns_when_nothing_to_save(at):
    _nav_button(at, "nav_validation").click().run()

    # A file where every row fails validation - nothing to save.
    csv_bytes = (
        b"Grade,MFR,XS,C2,Actual Izod Impact,Actual Tensile Modulus,Actual Flexural Modulus\n"
        b"CB2000GT,48.0,16.0,8.6,7.5,1380,1390\n"
    )
    uploader = at.get("file_uploader")[0]
    uploader.upload("all_bad.csv", csv_bytes, "text/csv")
    at.run()

    next(b for b in at.button if b.label == "Run Validation").click().run()
    save_button = next(b for b in at.button if b.label == "Save to training archive")
    save_button.click().run()
    assert not at.exception

    warning_texts = [w.value for w in at.warning]
    assert any("No successfully validated rows to save" in t for t in warning_texts), warning_texts


# --- Model Improvement -------------------------------------------------------

def test_model_improvement_nav_button_exists_and_navigates(at):
    _nav_button(at, "nav_model_improvement").click().run()
    assert at.session_state["active_page"] == "model_improvement"
    assert not at.exception, f"App raised on Model Improvement: {at.exception}"
    assert any("Model Improvement" in t.value for t in at.title)


def test_model_improvement_shows_zero_accumulated_data_on_clean_state(at):
    _nav_button(at, "nav_model_improvement").click().run()

    stat_values = [m.value for m in at.markdown if "pp-stat-n" in m.value]
    joined_stats = " ".join(stat_values)
    assert ">0<" in joined_stats

    caption_texts = [c.value for c in at.caption]
    assert any("No lab data has been saved" in t for t in caption_texts), caption_texts


def test_model_improvement_documents_the_controlled_roadmap_and_shows_current_metrics(at):
    _nav_button(at, "nav_model_improvement").click().run()

    body_text = " ".join(m.value for m in at.markdown)
    assert "Combine data" in body_text
    assert "Train a candidate model" in body_text
    assert "Compare metrics" in body_text
    assert "Human review and approval" in body_text
    assert "no automatic replacement path" in body_text

    warning_texts = [w.value for w in at.warning]
    assert any("not implemented yet" in t for t in warning_texts), warning_texts

    # Current production model's real, live metrics are shown for future comparison.
    metrics_html = " ".join(m.value for m in at.markdown if "pp-info-table" in m.value)
    assert "0.974" in metrics_html and "0.574" in metrics_html and "0.599" in metrics_html


def test_model_improvement_never_touches_the_model_binary(at):
    import hashlib

    model_path = REPO_ROOT / "pp_virtual_lab_models .joblib"
    before = hashlib.sha256(model_path.read_bytes()).hexdigest()

    _nav_button(at, "nav_model_improvement").click().run()
    assert not at.exception

    after = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert before == after
