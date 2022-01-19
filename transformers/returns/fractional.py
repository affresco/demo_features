import logging

import numpy as np
import pandas as pd

from fracdiff import fdiff

from transformers.abstract.mixin import AbstractFeature
from transformers.returns.mixin import ReturnsMixin

# ##################################################################
# FRACTIONS
# ##################################################################

MINUTES_PER_DAY = 24 * 60
DAYS_TO_ESTIMATE_FRAC_DIFF_ORDER = 7


# ##################################################################
# ESTIMATOR
# ##################################################################

class FractionalReturnFeature(AbstractFeature, ReturnsMixin):
    #
    PREFIX_FRAC = "frac"
    SUFFIX_RETURNS = "ret"

    # ##################################################################
    # INIT
    # ##################################################################

    def __init__(self, order: float, columns: list = None, prefix: str = None, fill_na: bool = True):

        # Init super feature class
        if prefix is None:
            prefix = self.PREFIX_FRAC
        super(FractionalReturnFeature, self).__init__(prefix=prefix)

        # Order of fractional differentiation
        self.order = order
        assert 0.0 < self.order < 1.0, "Order of fractional differentiation must be between 0 and 1."

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
        return self

    def transform(self, X, y=None, **fit_params):
        assert isinstance(X, pd.DataFrame), "Transform must be a DataFrame"
        if y is not None:
            logging.warning(f"Argument 'y' of fractional returns transformation will be ignored.")
        return self(df=X)

    def fit_transform(self, X, y=None, **fit_params):
        if self.order is not None:
            return self.transform(X, y, **fit_params)

    def get_params(self, *args, **kwargs):
        params = {
            "prefix": self.prefix,
            "columns": self.columns,
            "fill_na": self.fill_na,
            "order": self.order,
        }
        return params

    # ##################################################################
    # FACILITATION FOR ADDING STANDARD RETURN COLUMNS/SERIES
    # ##################################################################

    def __call__(self, df: pd.DataFrame):
        return self.apply(df=df, order=self.order, prefix=self.PREFIX_FRAC, columns=self.columns)

    @classmethod
    def apply(cls, df: pd.DataFrame, order: float, prefix: str = None,
              columns: list = None, spans: list = None, fill: bool = True):

        # Ensure that parameters are present
        prefix = prefix or cls.PREFIX_FRAC

        # Make some assertions
        cls._assert_before_applying(df=df, columns=columns, spans=spans)

        return cls.__compute_frac_return(df=df, order=order, columns=columns, prefix=prefix, fill=fill)

    # ##################################################################
    # CORE METHODS
    # ##################################################################

    @classmethod
    def __compute_frac_return(cls, df: pd.DataFrame, order: float, columns: list, prefix: str,
                              fill: bool = True, normalize: bool = False):
        #
        # Pre-existing columns:
        # we will not overwrite/compute the pre-existing columns
        already_present = df.columns

        # Our target columns
        for col in columns:

            # target output column name
            col_out = cls.prefix_frac(col, prefix=prefix)

            # Already there...
            if col_out in already_present:
                logging.warning(f"Column '{col_out}' is already present, skipping transformation.")
                continue

            # Make computation (return numpy array)
            res_array = fdiff(df[col], n=order)
            res = pd.Series(res_array, index=df.index, name=col_out)

            if normalize:
                res /= df[col]

            # Merge back into the df
            df = df.merge(res, left_index=True, right_index=True)

            if fill:
                df[col_out].fillna(0.0, inplace=True)

        return df

    # ##################################################################
    # ANCILLARY
    # ##################################################################

    @classmethod
    def prefix_frac(cls, target, prefix: str = None):
        prefix = prefix or cls.PREFIX_FRAC
        return "_".join([prefix, cls.SUFFIX_RETURNS, target])
