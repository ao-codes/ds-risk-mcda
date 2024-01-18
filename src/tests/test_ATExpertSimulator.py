import numpy as np
import pandas as pd

from src.ds_risk_mcda.simulators import ATExpertSimulator

# test data
criteria_data = {
    "Criterion": ["S", "S_K", "S_Q", "O", "D"],
    "ParentCriterion": [None, "S", "S", None, None],
    "Types": ["max", "max", "max", "max", "min"],
    "ScaleMax": [[1, 5], [1, 500000], [1, 5], None, [1, 5]],
}
criteria_df = pd.DataFrame(criteria_data)
risks_and_freqs = {"Risks": ["R_01", "R_02", "R_03", "R_04"], "Frequencies": [5, 2, 7, 4]}
risks_and_freqs_df = pd.DataFrame(risks_and_freqs)


def test_simulate_assessment_with_frequencies():
    expected_column_with_frequencies = pd.DataFrame(risks_and_freqs)
    expected_column_with_frequencies = expected_column_with_frequencies["Frequencies"].to_numpy()
    ates = ATExpertSimulator(criteria_data, risks_and_freqs, "O")
    ahp_pcms, topsis_df, origin_data = ates.simulate_assessment("saaty_scale")
    np.testing.assert_array_almost_equal(expected_column_with_frequencies, topsis_df["O"].transpose().to_numpy(), decimal=6)
    ates = ATExpertSimulator(criteria_df, risks_and_freqs_df, "O")
    ahp_pcms, topsis_df, origin_data = ates.simulate_assessment("saaty_scale")
    np.testing.assert_array_almost_equal(expected_column_with_frequencies, topsis_df["O"].transpose().to_numpy(), decimal=6)
    assert len(topsis_df.columns) == 4
    assert len(ahp_pcms) == 2
    assert origin_data.equals(criteria_df)


def test_simulate_assessment_no_frequencies():
    ates = ATExpertSimulator(criteria_data, risks_and_freqs)
    ahp_pcms, topsis_df, origin_data = ates.simulate_assessment("saaty_scale")
    assert len(topsis_df.columns) == 4
    assert len(ahp_pcms) == 2
    assert origin_data.equals(criteria_df)
