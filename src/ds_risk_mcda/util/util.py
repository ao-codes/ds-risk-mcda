from abc import ABC, abstractmethod
from os.path import dirname, join, realpath
from typing import Any

import pandas as pd


class DsRiskMcdaClass(ABC):
    """This abstract class can be inherited by other classes of this package to get the promt, the
    config_file_path functions and other useful things."""

    __dir_path = dirname(dirname(realpath(__file__)))
    _config_file_path = join(__dir_path, "config.yml")

    @property
    @abstractmethod
    def _prompt(self) -> str:
        ...

    def _ensure_input_parameter(self, parameter: Any) -> pd.DataFrame:
        if isinstance(parameter, dict):
            result = pd.DataFrame(parameter)
        elif isinstance(parameter, pd.DataFrame | None):
            result = parameter
        else:
            raise ValueError(f"Wrong data type for given function parameter {parameter}. Must be a dictionary or a pandas DataFrame.")
        return result
