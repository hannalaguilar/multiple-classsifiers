from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd


@dataclass
class DataCSV:
    """
    This is the main class for handle data.
    """
    data_path: Union[str, Path]

    @property
    def df(self) -> str:
        return pd.read_csv(self.data_path, index_col=0)

    @property
    def target_name(self) -> str:
        """

        Returns: the name of the target column (i.e., the last column).

        """
        return self.df.columns[-1]

    @property
    def n_classes(self) -> int:
        return self.df.iloc[:, -1].nunique()
