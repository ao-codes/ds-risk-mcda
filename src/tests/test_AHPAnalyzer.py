import pandas as pd
import pytest

from src.ds_risk_mcda.analyzers import AHPAnalyzer
from src.ds_risk_mcda.simulators import AHPExpertSimulator

# saple data
correct_ahp_input_data_dict = {
    "Risks": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"],
    "Categories": ["C_A", "C_B", "C_C", "C_A", "C_B", "C_C", "C_A", "C_B", "C_C", "C_A"],
    "Frequency": [5, 1, 7, 9, 5, 2, 7, 4, 9, 6],
}
correct_ahp_input_data = pd.DataFrame(correct_ahp_input_data_dict)

bad_ahp_input_data_dict = {"Risks": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]}
bad_ahp_input_data = pd.DataFrame(bad_ahp_input_data_dict)

simulator = AHPExpertSimulator(risks_categories_and_frequencies=correct_ahp_input_data)
cat_pcm, risk_pcms, _ = simulator.simulate_assessment("saaty_scale")


def test_ahp_analyzer_init():
    AHPAnalyzer(category_pcm=cat_pcm, risk_pcms=risk_pcms)


def test_ahp_analyzer_perform_analysis_with_correct_data():
    analyzer = AHPAnalyzer(category_pcm=cat_pcm, risk_pcms=risk_pcms)
    try:
        analyzer.perform_analysis()
    except ValueError as ex:
        pytest.fail(f"Warning -> unexpected exception! {ex}")
