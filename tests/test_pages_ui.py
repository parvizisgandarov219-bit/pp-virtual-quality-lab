"""Direct unit tests for small pure helper functions inside pages_ui/*.

These exist because SUPPORTED_GRADES currently only contains "C"-prefix
(HECO) grades, so the HOMO/RACO branches of the single-prediction page's
family-derivation logic can't be exercised by actually selecting a grade
in the live app (see tests/test_ui_apptest.py). Calling the helper
function directly, with real grade codes drawn from the model's full
31-grade trained schema, verifies that logic is correct and ready for
whenever a HOMO/RACO grade is added to SUPPORTED_GRADES.
"""

from pages_ui.single_prediction import _family_display


def test_family_display_for_heco_grade():
    family, label, c2_disabled = _family_display("CB4848MO")
    assert family == "HECO"
    assert label == "HECO — High Impact Copolymer (ICP)"
    assert c2_disabled is False


def test_family_display_for_homo_grade():
    family, label, c2_disabled = _family_display("HB3500GP")
    assert family == "HOMO"
    assert label == "HOMO — Homopolymer"
    assert c2_disabled is True


def test_family_display_for_raco_grade():
    family, label, c2_disabled = _family_display("RB4545MO")
    assert family == "RACO"
    assert label == "RACO — Random Copolymer"
    assert c2_disabled is False
