"""Unit tests for ui_components.py's pure display-logic helpers."""

import ui_components


def test_performance_tier_strong_for_izod():
    label, key = ui_components.performance_tier(0.974)
    assert key == "high"
    assert label == "Strong fit"


def test_performance_tier_moderate_for_tensile_and_flexural():
    for r2 in (0.574, 0.599):
        label, key = ui_components.performance_tier(r2)
        assert key == "moderate"
        assert label == "Moderate fit"


def test_performance_tier_boundary_is_strong():
    label, key = ui_components.performance_tier(ui_components.STRONG_FIT_R2_THRESHOLD)
    assert key == "high"


def test_performance_tier_just_below_boundary_is_moderate():
    label, key = ui_components.performance_tier(ui_components.STRONG_FIT_R2_THRESHOLD - 0.001)
    assert key == "moderate"


def test_labels_never_use_the_word_confidence():
    # R² is a goodness-of-fit statistic, not a probability or confidence
    # interval - the UI deliberately avoids the word "confidence".
    for r2 in (0.974, 0.574, 0.0, 1.0):
        label, _ = ui_components.performance_tier(r2)
        assert "confidence" not in label.lower()


def test_nav_items_include_model_information():
    labels = [label for _, label in ui_components.NAV_ITEMS]
    assert labels == [
        "Dashboard", "Single Prediction", "Batch Prediction",
        "Model Information", "Model Validation", "Model Improvement",
    ]
