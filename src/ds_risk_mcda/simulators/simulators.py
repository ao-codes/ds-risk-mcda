from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..assessments import AHPAssessment, ATAssessment
from ..util import DsRiskMcdaClass


class ExpertSimulator(ABC):
    @abstractmethod
    def simulate_assessment(self, mode: str) -> Any:
        ...

    @abstractmethod
    def _check_input(self, input_data: list[Any]) -> None:
        ...


class SimulatorWithAHPPart(DsRiskMcdaClass):
    def __init__(self) -> None:
        with open(self._config_file_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
            self._sim_mode_saaty = config["simulators"]["sim_mode_saaty"]
            self._sim_mode_simple = config["simulators"]["sim_mode_simple"]

    def _fill_additional_pcm(self, pcm: pd.DataFrame, mode: str) -> pd.DataFrame:
        if mode == self._sim_mode_saaty:
            scale_max_value = 9
        elif mode == self._sim_mode_simple:
            scale_max_value = 2
        else:
            raise ValueError("the simulation parameter 'mode' should be either 'simple_comparison' or 'saaty_scale'")
        num_rows, num_cols = pcm.shape
        for i in range(num_rows - 1):
            j = i + 1
            for x in range(j, num_cols):
                rand_int = np.random.randint(1, scale_max_value + 1)
                element_state = np.random.choice(["smaller", "bigger", "equal"])
                if element_state == "smaller":
                    pcm.iloc[i, x] = 1 / rand_int
                elif element_state == "bigger":
                    pcm.iloc[i, x] = rand_int
                else:
                    pcm.iloc[i, x] = 1
        return pcm


class AHPExpertSimulator(ExpertSimulator, SimulatorWithAHPPart):
    """
    A simulator class for generating synthetic data in the context of Analytic Hierarchy Process
    (AHP) assessments.

    Parameters
    ----------
        - risks_categories_and_frequencies (dict[str, list[str]] | pd.DataFrame):
        A dictionary or DataFrame containing information about risks, categories, and their
        corresponding frequencies.

        - additional_hierarchy (dict[str, list[str]] | pd.DataFrame |
        None, optional): A dictionary or DataFrame specifying an additional hierarchy level for the
        assessment. Default is None.

    Notes
    -----
        risks_categories_and_frequencies must be data (in the form of a dict oder a DataFrame) which
        has 3 columns in the following order: column 1: criteria (names or IDs), column 2: parents
        (parent of the criterion if required), column 3: criterion type ("max" or "min"), column 4:
        criterion scale (Lists of minimum and maxium values, see example).

        risks must be data (in the form of a dict oder a DataFrame) which has 1 column (2 if empiric
        frequency values for a criterion should be used).

        frequency_criterion musst be the name or ID (str) of the criterion, for which the given
        frequencies should be used in the simulation.

    Example
    -------
        raw_data = {
            "Risks": ["R-01", "R-02", "R-03", "R-04", "R-05"], "Categories": ["C_A", "C_A", "C_A",
            "C_B", "C_B"], "Frequencies": [5, 1, 7, 9, 5],
        }

        additional_level = {"Phases": ["Phase_1", "Phase_2", "Phase_3"]}

        simulator = AHPExpertSimulator(raw_data, additional_level)
    """

    def __init__(self, risks_categories_and_frequencies: dict[str, list[str]] | pd.DataFrame, additional_hierarchy: dict[str, list[str]] | pd.DataFrame | None = None) -> None:
        super().__init__()
        self.__prompt = "AHP-ExpertSimulator:"
        risks_categories_and_frequencies = self._ensure_input_parameter(risks_categories_and_frequencies)
        self.__additional_hierarchy = self._ensure_input_parameter(additional_hierarchy)
        self._check_input([risks_categories_and_frequencies, self.__additional_hierarchy])
        self.__risks_and_categories = risks_categories_and_frequencies.iloc[:, [0, 1]].copy()
        self.__risks_and_frequencies = risks_categories_and_frequencies.iloc[:, [0, 2]].copy()
        self.__risks_and_frequencies.set_index(self.__risks_and_frequencies.columns[0], inplace=True)
        self.__ac = AHPAssessment(self.__risks_and_categories, self.__additional_hierarchy)

    @property
    def _prompt(self) -> str:
        return self.__prompt

    def _check_input(self, input_data: list[pd.DataFrame]) -> None:
        if len(input_data[0].columns) < 3:
            raise ValueError("Incomplete data! Please specifiy a DataFrame (or dictionary) with the following 3 columns: risks, categories and frequencies.")
        if input_data[1] is not None:
            if len(input_data[1].columns) > 1:
                raise ValueError("Too much additional levels! Currently only 1 additional level is allowed.")

    @staticmethod
    def __simulate_rating_by_simple_comparison(risk_rating: float, comp_rating: float) -> float:
        result: float
        if risk_rating == comp_rating:
            result = 1
        elif risk_rating < comp_rating:
            result = 1 / 2
        else:
            result = 2
        return result

    @staticmethod
    def __simulate_rating_by_saaty_scale(risk_rating: float, comp_rating: float) -> float:
        result: float
        if risk_rating == comp_rating:
            result = 1
        elif risk_rating < comp_rating:
            result = 1 / (comp_rating - risk_rating)
        else:
            result = risk_rating - comp_rating
        return result

    @staticmethod
    def __map_vector_to_scale(vector: "pd.Series[int]", max_output: int) -> Any:
        max_input = np.max(vector)
        mapped_vector = np.round(1 + (vector - 1) * (max_output - 1) / (max_input - 1))
        return mapped_vector

    def __fill_pcm(self, pcm: pd.DataFrame, scores: "pd.Series[int]", mode: str) -> pd.DataFrame:
        if mode == self._sim_mode_saaty:
            scores = self.__map_vector_to_scale(scores, 9)
        for i in range(len(pcm)):
            for j in range(i + 1, len(pcm)):
                score = scores.iloc[i]
                comp_score = scores.iloc[j]
                if mode == self._sim_mode_simple:
                    rating = self.__simulate_rating_by_simple_comparison(score, comp_score)
                elif mode == self._sim_mode_saaty:
                    rating = self.__simulate_rating_by_saaty_scale(score, comp_score)
                else:
                    raise ValueError("the simulation parameter 'mode' should be either 'simple_comparison' or 'saaty_scale'")
                pcm.iloc[i, j] = float(rating)
        completed_pcm = AHPAssessment.complete_pcm(pcm.to_numpy())
        filled_df = AHPAssessment.fill_pcm_df(completed_pcm, pcm)
        return filled_df

    def __get_risk_frequencies_from_risk_pcm(self, risk_pcm: pd.DataFrame) -> Any:
        selected_frequencies = []
        for risk in risk_pcm.columns:
            freq = self.__risks_and_frequencies.loc[risk, self.__risks_and_frequencies.columns[0]]
            selected_frequencies.append(freq)
        result = pd.DataFrame()
        result[len(result.columns)] = risk_pcm.columns
        result[len(result.columns)] = selected_frequencies
        return result[result.columns[1]].squeeze()

    def simulate_assessment(self, mode: str) -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame] | None]:
        """
        Simulates an AHP-risk-assessment by generating synthetic data for RBS-category and risk
        pairwise comparison matrices (PCMs).

        Parameters
        ----------
            mode (str): The simulation mode, specifying which scale should be used when the
            simulator generates the values for the pairwise comparison matrices ('saaty_scale' or
            'simple_comparison').

        Returns
        -------
            tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame] | None]: A tuple containing
            the following:
                - filled_category_pcm (pd.DataFrame): A pairwise comparison matrix for categories
                  filled with simulated values.
                - filled_risk_pcms (list[pd.DataFrame]): A List of pairwise comparison matrices for
                  risks, each filled with simulated values.
                - filled_additional_pcms (list[pd.DataFrame] | None): A List of pairwise comparison
                  matrices for additional hierarchy levels, if applicable. None if no additional
                  hierarchy is provided.

        Notes
        -----
            The assessment involves building Risk and Category PCMs, counting category frequencies,
            and filling PCMs with simulated values. If an additional hierarchy is provided, PCMs for
            additional levels are also simulated and included in the output.

        Example
        -------
            category_pcm, risk_pcms, add_pcms = instance.simulate_assessment("saaty_scale")
        """
        category_pcm, risk_pcms = self.__ac.build_risk_and_category_pcms(self.__risks_and_categories)
        print(f"{self._prompt} counting category frequencies")
        category_frequencies = self.__risks_and_categories[self.__risks_and_categories.columns[1]].value_counts()
        print(f"{self._prompt} filling pcms with simulated values")
        filled_category_pcm = self.__fill_pcm(category_pcm, category_frequencies, mode)
        filled_risk_pcms: list[pd.DataFrame] = []
        for pcm in risk_pcms:
            risk_frequencies = self.__get_risk_frequencies_from_risk_pcm(pcm)
            filled_risk_pcm = self.__fill_pcm(pcm, risk_frequencies, mode)
            filled_risk_pcms.append(filled_risk_pcm)
        if self.__additional_hierarchy is not None:
            additional_pcms = self.__ac.build_additional_level_pcms(self.__risks_and_categories.iloc[:, 0], self.__additional_hierarchy.iloc[:, 0])
            filled_additional_pcms = []
            for pcm in additional_pcms:
                filled_pcm = self._fill_additional_pcm(pcm, mode)
                completed_pcm = AHPAssessment.complete_pcm(pcm.to_numpy())
                filled_pcm = AHPAssessment.fill_pcm_df(completed_pcm, pcm)
                filled_additional_pcms.append(filled_pcm)
            return filled_category_pcm, filled_risk_pcms, filled_additional_pcms
        return filled_category_pcm, filled_risk_pcms, None


class ATExpertSimulator(ExpertSimulator, SimulatorWithAHPPart):
    """
    A class representing an AHP-TOPSIS Expert Simulator, combining AHP and TOPSIS methodologies for
    risk assessment simulations.

    Parameters
    ----------
        - criteria_data (dict[str, list[str | int]] | pd.DataFrame): Data describing decision
          criteria.
        - risk_data (dict[str, list[str]] | pd.DataFrame): Data describing risks.
        - frequency_criterion (str | None, optional): Name of the frequency criterion if applicable.
          Default is None.

    Notes
    -----
        criteria_data must be data (in the form of a dict oder a DataFrame) which has 4 columns in
        the following order: column 1: criteria (names or IDs), column 2: parents (parent of the
        criterion if required), column 3: criterion type ("max" or "min"), column 4: criterion scale
        (Lists of minimum and maxium values, see example).

        risks must be data (in the form of a dict oder a DataFrame) which has 1 column (2 if
        empiric frequency values for a criterion should be used).

        frequency_criterion musst be the name or ID (str) of the criterion, for which the given frequencies
        should be used in the simulation.

    Example
    -------
        criteria_data = {
            "Criterion": ["S", "S_K", "S_Q", "O", "D", "AddCrit_1", "AddCrit_2", "AddCrit_3", "AddCrit_4"],
            "ParentCriterion": [None, "S", "S", None, None, "D", "D", "D", "D"],
            "CriterionType": ["max", "max", "max", "max", "min", "max", "max", "max", "min"],
            "CritScale": [[1, 5], [30000, 500000], [1, 100], None, [1, 5], [1, 5], [1, 5], [-2001, 50], [1, 5]],
            }

        risk_data = {"Risks": ["R_01", "R_02", "R_03", "R_04"], "Frequencies": [5, 2, 7, 4]}

        simulator = ATExpertSimulator(criteria_data, risk_data, frequency_criterion="O")
    """

    def __init__(self, criteria_data: dict[str, list[str | int]] | pd.DataFrame, risk_data: dict[str, list[str]] | pd.DataFrame, frequency_criterion: str | None = None) -> None:
        super().__init__()
        self.__prompt = "AHP-Topsis-ExpertSimulator:"
        self.__criteria_data = self._ensure_input_parameter(criteria_data)
        self.__risk_data = self._ensure_input_parameter(risk_data)
        self.__frequency_criterion = frequency_criterion
        self._check_input([self.__criteria_data, self.__risk_data, self.__frequency_criterion])
        with open(self._config_file_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
            self.__ahp_base_level_name = config["ahp_topsis_simulator"]["ahp_base_level_name"]

    @property
    def _prompt(self) -> str:
        return self.__prompt

    def _check_input(self, input_data: list[Any]) -> None:
        crit_data = input_data[0]
        risk_data = input_data[1]
        freq_crit = input_data[2]
        if len(crit_data.columns) < 4:
            raise ValueError(
                "Incomplete criteria data! Please specifiy a DataFrame (or dictionary) with 4 columns in the following order: criteria, criteria parents, criteria types and criteria scale."
            )
        if freq_crit is not None and len(risk_data.columns) < 2:
            raise ValueError("You have given risk data which should have empiric frequency data, but it seems you have not specified frequency data in the constructor.")
        if freq_crit is not None and freq_crit in crit_data[crit_data.columns[1]].values:
            raise ValueError("A parent criterion cannot be the fequency criterion. Please check your input data.")
        for element in set(crit_data[crit_data.columns[1]]):
            elementIsString = element is not None and element is not np.nan
            if elementIsString and element not in crit_data[crit_data.columns[0]].values:
                raise ValueError(f"You have specified a parent that is no criterion! -> {element}")
        for _, row in crit_data.iterrows():
            expected_list = row.iloc[3]
            if not isinstance(expected_list, list | type(None)):
                raise ValueError("You have to specifiy lists of integers or None in the list of criteria scales.")
            if not isinstance(expected_list, type(None)):
                x = (isinstance(e, int) for e in expected_list)  # type: ignore
                if not all(x):
                    raise ValueError("The values of the given criteria scales have to be integer values.")
                if len(expected_list) != 2:  # type: ignore
                    raise ValueError(
                        f"You have given less or more than 2 values for the criteria scales -> {expected_list}. Criteria scales should look like this list: [minimum value, maximum value]."
                    )
                if expected_list[0] == expected_list[1]:  # type: ignore
                    raise ValueError(f"Scale minimum matches scale maximum! {expected_list}")

    def __get_simulated_pcms(self, mode: str) -> list[pd.DataFrame]:
        hierarchy_levels = ATAssessment.get_hierarchy_lists(self.__criteria_data)
        simulated_pcms = []
        print(f"{self._prompt} starting pcm simulation")
        print(f"{self._prompt} using simulation mode '{mode}'")
        for level in hierarchy_levels:
            level_name = level[0]
            if level_name is None:
                level_name = self.__ahp_base_level_name
            print(f"{self._prompt} simulating PCM for {level_name}")
            empty_pcm = ATAssessment.create_empty_pcm_from_list(level[1])
            filled_pcm = self._fill_additional_pcm(empty_pcm, mode)
            completed_pcm = ATAssessment.complete_pcm(filled_pcm.to_numpy())
            filled_pcm = ATAssessment.fill_pcm_df(completed_pcm, empty_pcm)
            simulated_pcms.append(filled_pcm)
        return simulated_pcms

    def __fill_additional_columns_randomly(self, df: pd.DataFrame) -> pd.DataFrame:
        crit_data = self.__criteria_data.set_index(self.__criteria_data.columns[0])
        crit_data.replace(np.nan, None, inplace=True)
        for i, crit in enumerate(df.columns):
            crit_scale = crit_data.loc[crit, crit_data.columns[2]]
            if crit_scale is None:
                df.iloc[:, i] = None
            else:
                scale_min = crit_scale[0]  # type:ignore
                scale_max = crit_scale[1] + 1  # type:ignore
                random_values = np.random.randint(scale_min, scale_max, size=df.shape[0])  # type:ignore
                df.iloc[:, i] = pd.Series(random_values)
        return df

    def __get_simulated_decision_matrix(self) -> pd.DataFrame:
        risks = pd.DataFrame(self.__risk_data.iloc[:, 0])
        topsis_criteria = ATAssessment.get_topsis_criteria(self.__criteria_data)
        topsis_df = ATAssessment.get_topsis_df(topsis_criteria, risks)
        filled_topsis_df = self.__fill_additional_columns_randomly(topsis_df)
        if self.__frequency_criterion is not None:
            print(f"{self._prompt} using given frequencies in topsis decision matrix")
            frequency_column = self.__risk_data[self.__risk_data.columns[1]]
            filled_topsis_df[self.__frequency_criterion] = frequency_column.to_numpy()
        filled_topsis_df.index.rename(None, inplace=True)
        return filled_topsis_df

    def simulate_assessment(self, mode: str) -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        """
        Simulates an AHP-TOPSIS risk assessment by generating synthetic data for pairwise comparison
        matrices (PCMs) and a (TOPSIS)-Decision Matrix.

        Parameters
        ----------
            mode (str): The simulation mode, specifying which scale should be used when the
            simulator generates the values for the pairwise comparison matrices ('saaty_scale' or
            'simple_comparison').

        Returns
        -------
            tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]: A tuple containing the following:
                - simulated_pcms (list[pd.DataFrame]): A list of simulated pairwise comparison
                  matrices.
                - simulated_decision_matrix (pd.DataFrame): A simulated decision matrix.
                - data_origin (pd.DataFrame): The original criteria data which was given by the
                  user.

        Notes
        -----
            The assessment involves generating simulated PCMs based on the specified simulation mode
            and simulating a decision matrix. The original criteria data is also included in the
            output.

        Example
        -------
            pcms, decision_matrix, org_data = instance.simulate_assessment("saaty_scale")
        """
        simulated_pcms = self.__get_simulated_pcms(mode)
        simulated_decision_matrix = self.__get_simulated_decision_matrix()
        org_data = self.__criteria_data.replace(np.nan, None)
        org_data.replace(np.nan, None)
        return simulated_pcms, simulated_decision_matrix, org_data
