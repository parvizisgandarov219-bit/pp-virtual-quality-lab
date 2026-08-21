"""Direct unit tests for small pure helper functions inside pages_ui/*.

These are fast, isolated unit tests of the family-derivation display
logic in pages_ui/single_prediction.py, independent of a running
Streamlit session. The same three grades are also exercised end-to-end
through the real live selectbox in tests/test_ui_apptest.py.
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
