from os.path import dirname, join, realpath

import numpy as np
import pandas as pd

from src.ds_risk_mcda.analyzers import ATAnalyzer
from src.ds_risk_mcda.assessments import ATAssessment

# saple data
dir_path = dirname(realpath(__file__))
input_file_alpha = join(dir_path, "excel_test_files", "test_at_analyzer_alpha.xlsx")
input_file_beta = join(dir_path, "excel_test_files", "test_at_analyzer_beta.xlsx")
simple_criteria_data = {"Criterion": ["S", "O", "D"], "ParentCriterion": [None, None, None], "CritType": ["max", "max", "min"]}
simple_criteria_df = pd.DataFrame(simple_criteria_data)
risks_and_freqs = {"Risks": ["R_01", "R_02", "R_03", "R_04"]}
risks_and_freqs_df = pd.DataFrame(risks_and_freqs)
complex_criteria_data = {"Criterion": ["S_K", "S", "O", "D", "S_Q"], "ParentCriterion": ["S", None, None, None, "S"], "CritType": ["max", "max", "max", "min", "max"]}
complex_criteria_df = pd.DataFrame(complex_criteria_data)

# expected results (simple hierarchy)
expected_weight_result = {"Weight": [0.5, 0.25, 0.25]}
expected_weight_result = np.array(list(expected_weight_result.values()))
expected_topsis_result = {"Rank": [1, 2, 3, 4], "Score": [0.803, 0.644, 0.356, 0.197]}
expected_topsis_result = np.array(list(expected_topsis_result.values()))

# expected results (complex hierarchy)
expected_complex_weight_result = {"Weight": [0.25, 0.25, 0.666667, 0.333333], "GlobalWeight": [0.25, 0.25, 0.333333, 0.166667]}
expected_complex_weight_result = np.array(list(expected_complex_weight_result.values()))
expected_complex_topsis_result = {"Rank": [1, 2, 3, 4], "Score": [0.889, 0.739, 0.261, 0.111]}
expected_complex_topsis_result = np.array(list(expected_complex_topsis_result.values()))


# tests
def test_perform_simple_analysis():
    ahp_pcms, topsis_df, input_df = ATAssessment.read_excel_assessment(input_file_alpha)
    atan = ATAnalyzer(ahp_pcms, topsis_df, input_df)
    ahp_results, _, topsis_results = atan.perform_analysis()
    np.testing.assert_array_almost_equal(expected_weight_result, ahp_results.transpose().to_numpy(), decimal=3)
    np.testing.assert_array_almost_equal(expected_topsis_result, topsis_results.transpose().to_numpy(), decimal=3)
    np.testing.assert_almost_equal(ahp_results[ahp_results.columns[0]].sum(), 1, decimal=3)


def test_perform_complex_analysis():
    ahp_pcms, topsis_df, input_df = ATAssessment.read_excel_assessment(input_file_beta)
    atan = ATAnalyzer(ahp_pcms, topsis_df, input_df)
    ahp_results, _, topsis_results = atan.perform_analysis()
    np.testing.assert_array_almost_equal(expected_complex_weight_result, ahp_results.transpose().to_numpy(), decimal=3)
    np.testing.assert_array_almost_equal(expected_complex_topsis_result, topsis_results.transpose().to_numpy(), decimal=3)
    np.testing.assert_almost_equal(ahp_results[ahp_results.columns[1]].sum(), 1, decimal=3)
