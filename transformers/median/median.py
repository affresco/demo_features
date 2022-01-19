import numpy as np
import logging

# Scientific
import pandas as pd

# Base class for all features
from transformers.abstract.mixin import AbstractFeature


# ##################################################################
# AVERAGE TRUE RANGE
# ##################################################################

class MedianFeature(AbstractFeature):
    #
    # Columns created will be prefixed with this keyword
    PREFIX = "median"

    # ##################################################################
    # INIT
    # ##################################################################

    def __init__(self,
                 columns: list = None,
                 prefix: str = None):
        """
        Feature a computation of the median of provided columns.

        :param columns: columns on which computation will be performed
        :param prefix: Corresponding to the name of the column output
        """

        # Init super feature class
        if prefix is None:
            prefix = self.PREFIX
        super(MedianFeature, self).__init__(prefix=prefix)

        # Target columns on which we operate
        self.columns = self._sanitize_columns(columns=columns)

        # Initial span of data to be discarded
        self.warmup = 0

    # ##################################################################
    # SKLEARN TRANSFORMER MIXIN IMPLEMENTATION
    # ##################################################################

    def fit(self, *args, **kwargs):
        return self  # nothing to do here

    def transform(self, X, y=None, **fit_params):
        assert isinstance(X, pd.DataFrame), "Transform must be a DataFrame"
        if y is not None:
            logging.warning(f"Argument 'y' of median transformation will be ignored.")
        return self(df=X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y, **fit_params)

    def get_params(self, *args, **kwargs):
        params = {
            "prefix": self.prefix,
            "columns": self.columns,
        }
        return params

    # ##################################################################
    # CORE METHODS
    # ##################################################################

    def __call__(self, df: pd.DataFrame):
        return self.apply(df=df, prefix=self.PREFIX, columns=self.columns)

    @classmethod
    def apply(cls, df: pd.DataFrame, prefix: str = None, columns: list = None):

        # Ensure that parameters are present
        prefix = prefix or cls.PREFIX

        # Make some assertions
        cls._assert_before_applying(df=df, columns=columns, spans=None)

        return cls.__compute_median(df=df,
                                    prefix=prefix,
                                    columns=columns)

    @classmethod
    def __compute_median(cls, df: pd.DataFrame, columns: list, prefix: str):
        df[prefix] = np.median(df.loc[:, columns], axis=1)
        return df
