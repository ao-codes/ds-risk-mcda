import warnings
from abc import abstractmethod
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from pyDecision.algorithm import ahp_method, topsis_method

from ..util import DsRiskMcdaClass


class Analyzer:
    @abstractmethod
    def _check_input(self) -> None:
        ...

    @abstractmethod
    def perform_analysis(self) -> Any:
        ...


class AnalyzerWithAHPPart(DsRiskMcdaClass):
    def __init__(self) -> None:
        with open(self._config_file_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
            self._decimals = config["analyzer"]["decimals"]
            self._consistency_df_col_1 = config["ahp_analyzer"]["consistency_df_col_1"]
            self._consistency_df_col_2 = config["ahp_analyzer"]["consistency_df_col_2"]
            self.__consistency_df_cat_pcm_name = config["ahp_analyzer"]["consistency_df_cat_pcm_name"]
            decimal_format = config["analyzer"]["decimal_format"]
        pd.options.display.float_format = str(decimal_format).format

    def _calculate_weights_and_consitency(self, pcm: pd.DataFrame, weight_derivation: str) -> tuple[list[float], float]:
        pcm_is_from_excel = isinstance(pcm.index, pd.RangeIndex)
        if pcm_is_from_excel:
            pcm.set_index(pcm.columns[0], inplace=True)
        pcm = pcm.astype(float)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            weights, consistency = ahp_method(pcm, weight_derivation)
        if np.isnan(consistency) or consistency == 0.0:
            consistency = "perfect"
        else:
            consistency = round(consistency, self._decimals)
        return weights, consistency

    def _create_weight_table(self, pcm: pd.DataFrame, weight_derivation: str, columns: list[str]) -> tuple[pd.DataFrame, float]:
        weights, consistency = self._calculate_weights_and_consitency(pcm, weight_derivation)
        result_df = pd.DataFrame(columns=columns)
        categories = []
        weight_list = []
        for i, cat in enumerate(pcm.index):
            categories.append(cat)
            weight_list.append(weights[i])
        result_df[result_df.columns[0]] = categories
        result_df[result_df.columns[1]] = weight_list
        return result_df, consistency

    def _get_list_of_criteria(self, category_pcm: pd.DataFrame) -> list[str]:
        pcm_is_from_excel = isinstance(category_pcm.index, pd.RangeIndex)
        if pcm_is_from_excel:
            list_of_categories = category_pcm[category_pcm.columns[0]].to_list()
        else:
            list_of_categories = category_pcm.index.to_list()
        return list_of_categories

    def _print_verbose_pcm_info(self, pcm: pd.DataFrame, consistency: float, result_df: pd.DataFrame, pcm_name: str) -> None:
        df_string = pcm.to_string()
        consistency_string = "consistency ratio: " + str(consistency)
        line_length = int(len(df_string) / (df_string.count("\n") + 1))
        if line_length < len(consistency_string):
            line_length = len(consistency_string)
        line = "\u2500" * line_length
        headline = pcm_name
        headline = f"PCM: {headline}"
        digits = line_length - len(headline) - 1
        short_line = int(digits) * "\u2550"
        headline = headline + " " + short_line
        print(headline)
        print(pcm)
        print(line)
        print(consistency_string)
        print(line)
        result_df = result_df.set_index(result_df.columns[0])
        result_df.index.rename(None, inplace=True)
        print(result_df)
        print(line)

    def _create_consistency_result_df(self, list_of_consistencies: list[float], pcm_names: list[str], base_level_consistency: float | None = None) -> pd.DataFrame:
        if base_level_consistency is not None:
            consistencies = {self._consistency_df_col_1: [], self._consistency_df_col_2: []}  # type: dict[str, list[Any]]
            consistencies[self._consistency_df_col_1].append(self.__consistency_df_cat_pcm_name)
            consistencies[self._consistency_df_col_2].append(base_level_consistency)
        for i, name in enumerate(pcm_names):
            consistencies[self._consistency_df_col_1].append(name)
            consistency = list_of_consistencies[i]
            consistencies[self._consistency_df_col_2].append(consistency)
        return pd.DataFrame(consistencies)


class AHPAnalyzer(Analyzer, AnalyzerWithAHPPart):
    """
    A class for analyzing and processing data using the Analytic Hierarchy Process (AHP).

    Parameters
    ----------
        - category_pcm (pd.DataFrame): The pairwise comparison matrix (PCM) for categories.
        - risk_pcms (list[pd.DataFrame]): A List of PCMs for risks.
        - additional_pcms (list[pd.DataFrame] | None): A List of PCMs for additional hierarchy
          levels, if applicable. Default value is None.

    Notes
    -----
        The parameters should be the results from the AHPAssessment.
    """

    def __init__(self, category_pcm: pd.DataFrame, risk_pcms: list[pd.DataFrame], additional_pcms: list[pd.DataFrame] | None = None):
        super().__init__()
        self.__promt = "AHP-Analyzer:"
        self.__risks_and_categories = self.__get_risks_and_categories_from_pcms(category_pcm, risk_pcms)
        self.__category_pcm = category_pcm
        self.__risk_pcms = risk_pcms
        self.__additional_pcms = additional_pcms
        self._check_input()
        with open(self._config_file_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
            self.__risk_col = config["ahp_analyzer"]["risk_column"]
            self.__category_col = config["ahp_analyzer"]["category_column"]
            self.__local_weight_col = config["ahp_analyzer"]["local_weight_column"]
            self.__category_weight_col = config["ahp_analyzer"]["category_weight_column"]
            self.__consistency_df_col_1 = config["ahp_analyzer"]["consistency_df_col_1"]
            self.__consistency_df_col_2 = config["ahp_analyzer"]["consistency_df_col_2"]
            self.__consistency_df_cat_pcm_name = config["ahp_analyzer"]["consistency_df_cat_pcm_name"]
            self.__global_weight_col = config["ahp_analyzer"]["global_weight_col"]
            self.__add_weight_col = config["ahp_analyzer"]["add_weight_col"]

    @property
    def _prompt(self) -> str:
        return self.__promt

    def _check_input(self) -> None:
        self.__check_input_for_rbs_ahp(self.__risks_and_categories)

    def __finalize_result_and_consistency_dfs(self, final_df: pd.DataFrame, consistency_df: pd.DataFrame) -> None:
        final_df.set_index(final_df.columns[0], inplace=True)
        final_df.index.rename(None, inplace=True)
        consistency_df.set_index(consistency_df.columns[0], inplace=True)
        consistency_df.index.rename(None, inplace=True)

    def __get_risks_and_categories_from_pcms(self, category_pcm: pd.DataFrame, risk_pcms: list[pd.DataFrame]) -> pd.DataFrame:
        list_of_categories = self._get_list_of_criteria(category_pcm)
        risks_and_categories: dict[str, list[str]] = {"Risks": [], "Categories": []}
        i = 0
        for df in risk_pcms:
            risk_list = df.index.to_list()
            for risk in risk_list:
                risks_and_categories["Risks"].append(risk)
                risks_and_categories["Categories"].append(list_of_categories[i])
            i += 1
        return pd.DataFrame(risks_and_categories)

    def __create_additional_results(
        self, risk_weight_df: pd.DataFrame, additional_pcms: list[pd.DataFrame], weight_derivation: str, verbose: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        list_of_additional_elements = additional_pcms[0].index
        list_of_risks = self.__risks_and_categories.iloc[:, 0].to_list()
        rows = [self.__global_weight_col]
        rows.extend(list_of_additional_elements)
        results = pd.DataFrame(index=rows, columns=list_of_risks)
        additional_weight_dfs, additional_consistencies = self.__calculate_local_weights(list_of_risks, additional_pcms, weight_derivation, verbose)
        self.__fill_additional_result_df(risk_weight_df, additional_weight_dfs, results)
        self.__finish_additional_result_df(results)
        consistencies = {self.__consistency_df_col_1: [], self.__consistency_df_col_2: []}  # type: dict[list[str], list[float]]
        for i, risk in enumerate(list_of_risks):
            consistencies[self.__consistency_df_col_1].append(risk)
            consistencies[self.__consistency_df_col_2].append(additional_consistencies[i])
        return results, pd.DataFrame(consistencies)

    def __finish_additional_result_df(self, result_df: pd.DataFrame) -> None:
        weighted_sums = []
        for i in range(1, len(result_df)):
            ws = 0
            for _, col in enumerate(result_df):
                grw = result_df[col].iloc[0]
                element_weight = result_df[col].iloc[i]
                ws += element_weight * grw
            weighted_sums.append(ws)
        result_df[self.__add_weight_col] = np.nan
        for i in range(1, len(result_df)):
            result_df.iloc[i, len(result_df.columns) - 1] = weighted_sums[i - 1]

    def __fill_additional_result_df(self, risk_weight_df: pd.DataFrame, additional_weight_dfs: list[pd.DataFrame], result_df: pd.DataFrame) -> None:
        for i, row in risk_weight_df.iterrows():
            for col_name, col in result_df.items():
                if col_name == row[self.__risk_col]:
                    col.iloc[0] = row[self.__global_weight_col]
        for j, col_name in enumerate(result_df.columns):
            df = additional_weight_dfs[j]
            for i, row in df.iterrows():
                index = cast(int, i) + 1
                result_df[col_name].iloc[index] = row.iloc[1]

    def __check_input_for_rbs_ahp(self, risks_and_categories: pd.DataFrame) -> None:
        if not isinstance(risks_and_categories, pd.DataFrame):
            raise ValueError("The provided input is not a pandas DataFrame!")
        if len(risks_and_categories.columns) != 2:
            raise ValueError("You have to provide 2 columns in input data. 1 column for risks and 1 column for categories!")

    def __calculate_category_weights(self, weight_derivation: str, verbose: bool = False) -> tuple[pd.DataFrame, float]:
        result_columns = [self.__category_col, self.__category_weight_col]
        result_df, consistency = self._create_weight_table(self.__category_pcm, weight_derivation, result_columns)
        if verbose is True:
            print("\n")
            self._print_verbose_pcm_info(self.__category_pcm, consistency, result_df, self.__consistency_df_cat_pcm_name)
            print("\n")
        return result_df, consistency

    def __calculate_local_weights(
        self, list_of_categories: list[str], risk_pcms: list[pd.DataFrame], weight_derivation: str, verbose: bool = True
    ) -> tuple[list[pd.DataFrame], list[float]]:
        result_columns = [self.__risk_col, self.__local_weight_col]
        weight_dfs = []
        consitencies = []
        for i, category in enumerate(list_of_categories):
            result_df, consistency = self._create_weight_table(risk_pcms[i], weight_derivation, result_columns)
            if verbose is True:
                print("\n")
                self._print_verbose_pcm_info(risk_pcms[i], consistency, result_df, category)
            weight_dfs.append(result_df)
            consitencies.append(consistency)
        if verbose is True:
            print("\n")
        return weight_dfs, consitencies

    def __add_columns_to_risk_weight_dfs(self, risk_weight_dfs: list[pd.DataFrame]) -> None:
        for weight_df in risk_weight_dfs:
            weight_df[self.__category_col] = ""
            weight_df[self.__category_weight_col] = np.nan
            weight_df.insert(0, self.__category_col, weight_df.pop(self.__category_col))
            for i in range(len(weight_df)):
                for j in range(len(self.__risks_and_categories)):
                    risk = weight_df.iloc[i, 1]
                    comp_risk = self.__risks_and_categories.iloc[j, 0]
                    if risk == comp_risk:
                        category = self.__risks_and_categories.iloc[j, 1]
                        weight_df.iloc[i, 0] = category

    def __merge_weight_dfs(self, risk_df_list: list[pd.DataFrame], category_df: pd.DataFrame) -> pd.DataFrame:
        combined_df = pd.concat(item for item in risk_df_list)
        for i in range(len(combined_df)):
            for j in range(len(category_df)):
                if combined_df.iloc[i, 0] == category_df.iloc[j, 0]:
                    combined_df.iloc[i, 3] = category_df.iloc[j, 1]
        combined_df[self.__global_weight_col] = combined_df[self.__local_weight_col] * combined_df[self.__category_weight_col]
        combined_df.sort_values(by=self.__global_weight_col, ascending=False, inplace=True)
        combined_df = pd.DataFrame(combined_df.iloc[:, [1, 2, 0, 3, 4]])
        num_of_risks = combined_df[combined_df.columns[0]].count()
        rank_list = list(range(1, num_of_risks + 1))
        combined_df.insert(1, "Rank", rank_list)
        return combined_df

    def perform_analysis(
        self, verbose: bool = False, weight_derivation: str = "max_eigen"
    ) -> tuple[pd.DataFrame, pd.DataFrame, None] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs a comprehensive risk analysis of the assessment data, calculating category and risk
        weights.

        Parameters
        ----------
            - verbose (bool, optional): If True, additional information will be printed during the
              analysis. Default is False.
            - weight_derivation (str, optional): Method for deriving weights, 'max_eigen' for the
              original maximum eigenvalue approach from Thomas L. Saaty. Other valid methods are
              "mean" and "geometric". Default is 'max_eigen'.

        Returns
        -------
            tuple[pd.DataFrame, pd.DataFrame, None] | tuple[pd.DataFrame, pd.DataFrame,
            pd.DataFrame]:

                A tuple containing the following:

                - final_df (pd.DataFrame): The final result DataFrame containing both category and
                  risk weights.
                - consistency_df (pd.DataFrame): A DataFrame summarizing the consistencies of all
                  pairwise comparison matrices. Note: a value of > 0,1 should be considered as not
                  satisfying.
                - additional_results (pd.DataFrame | None): Additional results DataFrame for an
                  extra hierarchy level, if provided. None if no additional hierarchy is provided.

        Notes
        -----
            The analysis involves calculating category weights, risk weights, and consistencies for
            the pariwise comparison matrices. If an additional hierarchy is provided, weights for
            that hierarchy level are also calculated and included in the output.

        Example
        -------
            final_results, consistency_results, additional_results =
            instance.perform_analysis(verbose=True)
        """
        print(f"{self._prompt} calculating category weights")
        category_weights, pcm_consistency = self.__calculate_category_weights(weight_derivation, verbose)
        print(f"{self._prompt} calculating risk weights")
        list_of_categories = self._get_list_of_criteria(self.__category_pcm)
        risk_weight_dfs, pcm_consistencies = self.__calculate_local_weights(list_of_categories, self.__risk_pcms, weight_derivation, verbose)
        self.__add_columns_to_risk_weight_dfs(risk_weight_dfs)
        final_df = self.__merge_weight_dfs(risk_weight_dfs, category_weights)
        final_df.reset_index(drop=True, inplace=True)
        consistency_df = self._create_consistency_result_df(pcm_consistencies, list_of_categories, pcm_consistency)
        if self.__additional_pcms is not None:
            print(f"{self._prompt} calculating weights for additional hierarchy level")
            additional_results, additional_consistencies = self.__create_additional_results(final_df, self.__additional_pcms, weight_derivation, verbose)
            consistency_df = pd.concat([consistency_df, additional_consistencies], axis=0)
            self.__finalize_result_and_consistency_dfs(final_df, consistency_df)
            return final_df, consistency_df, additional_results
        self.__finalize_result_and_consistency_dfs(final_df, consistency_df)
        return final_df, consistency_df, None


class ATAnalyzer(Analyzer, AnalyzerWithAHPPart):
    """
    A class for analyzing and processing data using a combined AHP-TOPSIS method.

    Parameters
    ----------
        - ahp_pcms (list[pd.DataFrame]): A List of pairwise comparison matrices (PCMs) for criteria
          comparisons.
        - decision_matrix (pd.DataFrame): A DataFrame representing the decision matrix for the
          TOPSIS method.
        - crit_data (pd.DataFrame): The original criteria data that was given by the user.

    Notes
    -----
        The parameters should be the results from the ATAssessment.
    """

    def __init__(self, ahp_pcms: list[pd.DataFrame], decision_matrix: pd.DataFrame, crit_data: pd.DataFrame) -> None:
        super().__init__()
        self.__prompt = "AHP-TOPSIS-Analyzer:"
        self.__input_pcms = ahp_pcms
        self.__input_dm = decision_matrix
        self.__crit_data = crit_data
        self.__crit_data.set_index(self.__crit_data.columns[0], inplace=True)
        self.__parents = self.__get_parents(self.__crit_data)
        self._check_input()
        with open(self._config_file_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
            self.__col_rank = config["ahp_topsis_analyzer"]["col_rank"]
            self.__col_risk = config["ahp_topsis_analyzer"]["col_risk"]
            self.__col_result = config["ahp_topsis_analyzer"]["col_result"]
            self.__col_crit = config["ahp_topsis_analyzer"]["col_crit"]
            self.__col_crit_weight = config["ahp_topsis_analyzer"]["col_crit_weight"]
            self.__col_global_weight = config["ahp_topsis_analyzer"]["col_global_weight"]
            self.__col_global_weight = config["ahp_topsis_analyzer"]["col_global_weight"]
            self.__ahp_base_level_name = config["ahp_topsis_analyzer"]["ahp_base_level_name"]

    @property
    def _prompt(self) -> str:
        return self.__prompt

    def _check_input(self) -> None:
        if not isinstance(self.__input_dm, pd.DataFrame):
            raise ValueError("The provided TOPSIS-input is not a pandas DataFrame!")
        if len(self.__input_dm.columns) < 2:
            raise ValueError("You have to provide 2 or more columns in input data!")
        if self.__input_dm.isnull().values.any():
            raise ValueError("Some values in the given TOPSIS-table are 'None'!")

    def __create_ahp_weight_entries(self, weight_results: pd.DataFrame) -> list[float]:
        ahp_weights: list[float] = []
        for crit in self.__input_dm.columns:
            ahp_weights.append(weight_results.loc[crit, self.__col_crit_weight])
        return ahp_weights

    def __create_criteria_type_entries(self) -> list[str]:
        crit_types: list[Any] = []
        crit_type_column = self.__crit_data.columns[1]
        for crit in self.__input_dm.columns:
            crit_types.append(self.__crit_data.loc[crit, crit_type_column])
        return crit_types

    def __build_result_data_frame(self, topsis_results: list[float]) -> pd.DataFrame:
        result_df = pd.DataFrame(columns=[self.__col_rank, self.__col_risk, self.__col_result])
        result_df[self.__col_risk] = self.__input_dm.index
        result_df[self.__col_result] = topsis_results
        result_df.sort_values(by=self.__col_result, ascending=False, inplace=True)
        result_df[self.__col_rank] = list(range(1, len(topsis_results) + 1))
        result_df.reset_index(drop=True, inplace=True)
        result_df.set_index(self.__col_risk, inplace=True)
        return result_df

    def __get_parents(self, crit_df: pd.DataFrame) -> pd.DataFrame:
        result = {"Element": [], "Parent": []}  # type: dict[str, list[Any]]
        for crit, _ in crit_df.iterrows():
            parent = crit_df.loc[str(crit), crit_df.columns[0]]
            if parent is None:
                result["Element"].append(crit)
                result["Parent"].append(None)
            else:
                i = 0
                for crit_comp, _ in crit_df.iterrows():
                    i += 1
                    if parent == crit_comp:
                        result["Element"].append(crit)
                        result["Parent"].append(parent)
                        break
        result_df = pd.DataFrame(result)
        result_df.set_index(result_df.columns[0], inplace=True)
        return result_df

    def __compute_global_weights(self, weight_results: pd.DataFrame, max_level: int, criteria_levels: dict[str, int]) -> pd.DataFrame:
        global_weights = {self.__col_crit: [], self.__col_global_weight: []}  # type: dict[str, list[Any]]
        weight_results[self.__col_global_weight] = None
        for level in range(1, max_level + 1):
            criteria_in_level = {key: value for key, value in criteria_levels.items() if value == level}
            for criterion in criteria_in_level:
                global_weights[self.__col_crit].append(criterion)
                local_weight: Any = weight_results.loc[criterion, weight_results.columns[0]]
                if level == 1:
                    gw = local_weight
                else:
                    parent = self.__parents.loc[criterion, self.__parents.columns[0]]
                    parent_weight: Any = weight_results.loc[str(parent), weight_results.columns[1]]
                    gw = parent_weight * local_weight
                weight_results.loc[criterion, weight_results.columns[1]] = gw
        return weight_results

    def __compute_pcm_weights(self, weight_derivation: str, verbose: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        results = {self.__col_crit: [], self.__col_crit_weight: []}  # type: dict[str, list[Any]]
        consistencies = {self._consistency_df_col_1: [], self._consistency_df_col_2: []}  # type: dict[str, list[Any]]
        # compute local weights and put them in a dataframe ...
        result_columns = [self.__col_crit, self.__col_crit_weight]
        for _, pcm in enumerate(self.__input_pcms):
            criteria = self._get_list_of_criteria(pcm)
            parent = self.__parents.loc[criteria[0], self.__parents.columns[0]]
            if parent is None:
                pcm_name = self.__ahp_base_level_name
            else:
                pcm_name = parent
            weights, cr = self._create_weight_table(pcm, weight_derivation, result_columns)
            consistencies[self._consistency_df_col_1].append(pcm_name)
            consistencies[self._consistency_df_col_2].append(cr)
            if verbose:
                print("\n")
                self._print_verbose_pcm_info(pcm, cr, weights, pcm_name)  # type: ignore
            for i, criterion in enumerate(criteria):
                results[self.__col_crit].append(criterion)
                results[self.__col_crit_weight].append(weights.iloc[i, 1])
        if verbose:
            print("\n")
        # compute global weights ...
        weight_results = pd.DataFrame(results)
        weight_results.set_index(weight_results.columns[0], inplace=True)
        consistency_results = pd.DataFrame(consistencies)
        consistency_results.set_index(consistency_results.columns[0], inplace=True)
        criteria_levels = self.__get_criteria_levels(self.__crit_data)
        max_level = max(criteria_levels.values())
        if max_level > 1:
            weight_results = self.__compute_global_weights(weight_results, max_level, criteria_levels)
        return weight_results, consistency_results

    def __get_criteria_levels(self, data: pd.DataFrame) -> dict[str, int]:
        levels_dict = {}
        for element, _ in data.iterrows():
            level = self.__find_level(str(element), data)
            levels_dict[str(element)] = level
        return levels_dict

    def __find_level(self, element: str | None, data: pd.DataFrame, level: int = 1) -> int:
        if element is not None:
            parent = data.loc[element, data.columns[0]]
        if parent is None:
            return level
        return self.__find_level(str(parent), data, level + 1)

    def perform_analysis(self, verbose: bool = False, weight_derivation: str = "max_eigen") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform a comprehensive risk analysis using Analytic Hierarchy Process (AHP) and Technique
        for Order of Preference by Similarity to Ideal Solution (TOPSIS).

        Parameters
        ----------
            - verbose (bool, optional): If True, additional information will be printed during the
              analysis. Default is False.
            - weight_derivation (str, optional): Method for deriving weights, 'max_eigen' for the
              original maximum eigenvalue approach from Thomas L. Saaty. Other valid methods are
              "mean" and "geometric". Default is 'max_eigen'.

        Returns
        -------
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the following result
            DataFrames:

                - ahp_weights (pd.DataFrame): A DataFrame containing the AHP weights for criteria.
                - consistencies (pd.DataFrame): A DataFrame with consistency measures for the given
                  pairwise comparison matrices (PCMs).
                - topsis_results (pd.DataFrame): Results of the TOPSIS algorithm, including rankings
                  by descending order and scores (closeness to the so called 'positive ideal
                  solution').

        Notes
        -----
            The analysis involves computing criteria weights for the given by using the AHP
            algorithm on the given PCMs, and applying the TOPSIS algorithm by using the before
            computed criteria weights and the given decision matrix. The result DataFrames provide
            insights into the criteria weights, consistencies, and the overall performance ranking
            based on TOPSIS.

        Example
        -------
            ahp_weights, consistencies, topsis_results = instance.perform_analysis(verbose=True)
        """
        print(f"{self._prompt} computing weights for given pcms")
        pcm_weights, consistencies = self.__compute_pcm_weights(weight_derivation, verbose)
        ahp_weights = pcm_weights[pcm_weights.index.isin(self.__input_dm.columns)]
        print(f"{self._prompt} creating entries for AHP weights")
        ahp_weight_list = self.__create_ahp_weight_entries(pcm_weights)
        print(f"{self._prompt} creating entries for criteria types")
        crit_types = self.__create_criteria_type_entries()
        print(f"{self._prompt} starting TOPSIS algorithm")
        topsis_results = topsis_method(self.__input_dm, ahp_weight_list, crit_types, verbose=False, graph=False)
        print(f"{self._prompt} creating result dataframes")
        topsis_results = self.__build_result_data_frame(topsis_results)
        if verbose:
            print("\n")
        ahp_weights.index.rename(None, inplace=True)
        consistencies.index.rename(None, inplace=True)
        topsis_results.index.rename(None, inplace=True)
        return ahp_weights, consistencies, topsis_results
