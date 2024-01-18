import os
from os.path import dirname, join, realpath

import pandas as pd
import pytest

from src.ds_risk_mcda.assessments import ATAssessment

simple_criteria = {"Criterion": ["S", "O", "D"], "ParentCriterion": [None, None, None], "CritType": ["max", "max", "min"]}
criteria_with_subcriteria = {"Criterion": ["S", "S_K", "S_Q", "O", "D"], "ParentCriterion": [None, "S", "S", None, None], "CritType": ["max", "max", "max", "max", "min"]}
risks = ["R_01", "R_02", "R_03", "R_04", "R_05", "R_06", "R_07", "R_08", "R_09", "R_10"]
dir_path = dirname(realpath(__file__))
input_file = join(dir_path, "excel_test_files", "test_at_input.xlsx")
output_file = join(dir_path, "excel_test_files", "test_at_output.xlsx")
output_with_subcrits = join(dir_path, "excel_test_files", "test_new_ahp_topsis_with_subcrita.xlsx")
output_without_subcrits = join(dir_path, "excel_test_files", "test_new_ahp_topsis_no_subcriteria.xlsx")


def test_prepare_excel_assessment_with_correct_input():
    crit_df = pd.read_excel(input_file, sheet_name="correct_crit_input")
    risk_df = pd.read_excel(input_file, sheet_name="correct_risk_input")
    ac = ATAssessment(crit_df, risk_df)
    ac.prepare_excel_assessment(output_file)


def test_prepare_excel_assessment_with_bad_input_1():
    with pytest.raises(ValueError, match="Please choose a shorter name for 'PCM_ThisIsAVeryVeryLongCritNameThatShouldBeAvoided'. It is recommended to use IDs."):
        crit_df = pd.read_excel(input_file, sheet_name="crit_name_too_long")
        risk_df = pd.read_excel(input_file, sheet_name="correct_risk_input")
        ac = ATAssessment(crit_df, risk_df)
        ac.prepare_excel_assessment(output_file)


def test_prepare_excel_assessment_with_bad_input_2():
    with pytest.raises(ValueError, match="Risks have to be unique! Please check input data."):
        crit_df = pd.read_excel(input_file, sheet_name="correct_crit_input")
        risk_df = pd.read_excel(input_file, sheet_name="double_risk")
        ac = ATAssessment(crit_df, risk_df)
        ac.prepare_excel_assessment(output_file)


def test_prepare_excel_assessment_with_bad_input_3():
    with pytest.raises(ValueError, match="Criteria have to be unique! Please check input data."):
        crit_df = pd.read_excel(input_file, sheet_name="double_crit")
        risk_df = pd.read_excel(input_file, sheet_name="correct_risk_input")
        ac = ATAssessment(crit_df, risk_df)
        ac.prepare_excel_assessment(output_file)


def test_prepare_excel_assessment_with_bad_input_4():
    with pytest.raises(ValueError, match="A parent criterion must not be it's own parent -> 'D'"):
        crit_df = pd.read_excel(input_file, sheet_name="parent_is_own_parent")
        risk_df = pd.read_excel(input_file, sheet_name="correct_risk_input")
        ac = ATAssessment(crit_df, risk_df)
        ac.prepare_excel_assessment(output_file)


def test_prepare_excel_assessment_with_subcriteria():
    cat_df = pd.DataFrame(criteria_with_subcriteria)
    risk_df = pd.DataFrame(risks)
    atac = ATAssessment(cat_df, risk_df)
    atac.prepare_excel_assessment(output_with_subcrits)
    assert os.path.isfile(output_with_subcrits)


def test_prepare_excel_assessment_without_subcriteria():
    cat_df = pd.DataFrame(simple_criteria)
    risk_df = pd.DataFrame(risks)
    atac = ATAssessment(cat_df, risk_df)
    atac.prepare_excel_assessment(output_without_subcrits)
    assert os.path.isfile(output_without_subcrits)
