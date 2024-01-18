from os.path import dirname, join, realpath

import numpy as np
import pandas as pd
import pytest

from src.ds_risk_mcda.analyzers import AHPAnalyzer
from src.ds_risk_mcda.assessments import AHPAssessment

# sample test data
expected_result = np.array([[1.0, 2.0, 1 / 3, 4], [0.5, 1.0, 7.0, 1 / 9], [3.0, 1 / 7, 1.0, 3], [1 / 4, 9.0, 1 / 3, 1.0]])
df = pd.DataFrame(np.array([[float("nan"), 2.0, 1 / 3, 4], [0.5, float("nan"), 7.0, 1 / 9], [3.0, 1 / 7, float("nan"), 3], [1 / 4, 9.0, 1 / 3, 1.0]]))
empty_df = np.full((4, 4), np.nan)
dir_path = dirname(realpath(__file__))
input_file = join(dir_path, "excel_test_files", "test_ahp_input.xlsx")
output_file = join(dir_path, "excel_test_files", "test_ahp_output.xlsx")
df_from_excel = pd.read_excel(join(dir_path, "excel_test_files", "test_input_pcm.xlsx"), index_col=0)
filled_test_file = join(dir_path, "excel_test_files", "test_assessment.xlsx")


def test_prepare_excel_assessment_with_correct_input():
    input_df = pd.read_excel(input_file, sheet_name="correct_input")
    ac = AHPAssessment(input_df)
    ac.prepare_excel_assessment(output_file)


def test_prepare_excel_assessment_with_bad_input_1():
    with pytest.raises(ValueError, match="Risks have to be unique! Please check input data."):
        input_df = pd.read_excel(input_file, sheet_name="double_risk")
        ac = AHPAssessment(input_df)
        ac.prepare_excel_assessment(output_file)


def test_prepare_excel_assessment_with_bad_input_2():
    with pytest.raises(ValueError, match=r"Please choose a shorter name for 'Level_1_CriteriaWithFarToLongNamesAreNotGood'. It is recommended to use IDs."):
        input_df = pd.read_excel(input_file, sheet_name="name_too_long")
        ac = AHPAssessment(input_df)
        ac.prepare_excel_assessment(output_file)


def test_prepare_excel_assessment_with_no_input():
    with pytest.raises(ValueError, match="Cannot create assessment file when there is no input given in the class constructor."):
        ac = AHPAssessment(risk_data=None)
        ac.prepare_excel_assessment("output.xlsx")


def test_analyze_excel_assessment():
    category_pcm, risk_pcms, add_pcms = AHPAssessment.read_excel_assessment(filled_test_file)
    analyzer = AHPAnalyzer(category_pcm, risk_pcms, add_pcms)
    _, _, additional_analysis = analyzer.perform_analysis(weight_derivation="mean")
    result = additional_analysis.iloc[:, -1].to_numpy()
    expected_analysis_result = np.array([np.nan, 0.361, 0.104, 0.535])
    np.testing.assert_array_almost_equal(result, expected_analysis_result, decimal=3)


def test_read_excel_assessment():
    category_pcm, risk_pcms, add_pcms = AHPAssessment.read_excel_assessment(filled_test_file)
    result = category_pcm.to_numpy()
    expected_excel_result = np.array([[1, 1 / 5], [5, 1]])
    np.testing.assert_array_almost_equal(result, expected_excel_result, decimal=6)
    result = add_pcms[2].to_numpy()
    expected_excel_result = np.array([[1, 4, 7], [1 / 4, 1, 3], [1 / 7, 1 / 3, 1]])
    np.testing.assert_array_almost_equal(result, expected_excel_result, decimal=6)
    result = risk_pcms[1].to_numpy()
    expected_excel_result = np.array([[1, 1 / 6], [6, 1]])
    np.testing.assert_array_almost_equal(result, expected_excel_result, decimal=6)
    result = add_pcms[2].to_numpy()
    expected_excel_result = np.array([[1, 4, 7], [1 / 4, 1, 3], [1 / 7, 1 / 3, 1]])
    np.testing.assert_array_almost_equal(result, expected_excel_result, decimal=6)


def test_complete_pcm_with_valid_data():
    test_pcm = df.to_numpy(copy=True)
    result = AHPAssessment.complete_pcm(test_pcm)
    np.testing.assert_array_almost_equal(result, expected_result, decimal=6)


def test_complete_pcm_with_valid_excel_data():
    test_pcm = df_from_excel.to_numpy(copy=True)
    result = AHPAssessment.complete_pcm(test_pcm)
    np.testing.assert_array_almost_equal(result, expected_result, decimal=6)


def test_complete_pcm_with_empty_data():
    with pytest.raises(ValueError):
        AHPAssessment.complete_pcm(empty_df)


def test_complete_pcm_with_wrong_input_type():
    with pytest.raises(ValueError):
        AHPAssessment.complete_pcm([1, 2, 3])
