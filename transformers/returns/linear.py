import logging

import pandas as pd

from transformers.abstract.mixin import AbstractFeature
from transformers.returns.mixin import ReturnsMixin


# ##################################################################
# ESTIMATOR
# ##################################################################

class LinearReturnFeature(AbstractFeature, ReturnsMixin):
    #
    PREFIX_LINEAR = "lin"
    SUFFIX_RETURNS = "ret"

    # ##################################################################
    # INIT
    # ##################################################################

    def __init__(self, columns: list = None, prefix: str = None, fill_na: bool = True):

        # Init super feature class
        if prefix is None:
            prefix = self.PREFIX_LINEAR
        super(LinearReturnFeature, self).__init__(prefix=prefix)

        # Target columns on which we operate
        self.columns = self._sanitize_ret_columns(columns=columns)

        # Initial span of data to be discarded
        self.warmup = 1

        # Filling missing data
        #
        # Fill first N/A
        self.fill_na = bool(fill_na)

    # ##################################################################
    # SKLEARN TRANSFORMER MIXIN IMPLEMENTATION
    # ##################################################################

    def fit(self, *args, **kwargs):
        return self  # nothing to do here

    def transform(self, X, y=None, **fit_params):
        assert isinstance(X, pd.DataFrame), "Transform must be a DataFrame"
        if y is not None:
            logging.warning(f"Argument 'y' of linear returns transformation will be ignored.")
        return self(df=X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y, **fit_params)

    def get_params(self, *args, **kwargs):
        params = {
            "prefix": self.prefix,
            "columns": self.columns,
            "fill_na": self.fill_na,
        }
        return params

    # ##################################################################
    # FACILITATION FOR ADDING STANDARD RETURN COLUMNS/SERIES
    # ##################################################################

    def __call__(self, df: pd.DataFrame):
        return self.apply(df=df, prefix=self.PREFIX_LINEAR, columns=self.columns)

    @classmethod
    def apply(cls, df: pd.DataFrame, prefix: str = None, columns: list = None, spans: list = None, fill: bool = True):

        # Ensure that parameters are present
        prefix = prefix or cls.PREFIX_LINEAR

        # Make some assertions
        cls._assert_before_applying(df=df, columns=columns, spans=spans)

        return cls.__compute_linear_return(df=df, columns=columns, prefix=prefix, fill=fill)

    # ##################################################################
    # CORE METHODS
    # ##################################################################

    @classmethod
    def __compute_linear_return(cls, df: pd.DataFrame, columns: list, prefix: str, fill: bool = True):
        #
        # Pre-existing columns:
        # we will not overwrite/compute the pre-existing columns
        already_present = df.columns

        # Our target columns
        for col in columns:

            col_out = cls.prefix_lin(col)

            # Already there...
            if col_out in already_present:
                logging.warning(f"Column '{col_out}' is already present, skipping transformation.")
                continue

            res = df[col] / df[col].shift(1) - 1.0
            res.name = col_out
            df = df.merge(res, left_index=True, right_index=True)

            if fill:
                df[col_out].fillna(0.0, inplace=True)

        return df

    # ##################################################################
    # ANCILLARY
    # ##################################################################

    @classmethod
    def prefix_lin(cls, target):
        return "_".join([cls.PREFIX_LINEAR, cls.SUFFIX_RETURNS, target])
