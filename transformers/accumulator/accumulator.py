import numpy as np
import logging

# Scientific
import pandas as pd

# Base class for all features
from transformers.abstract.mixin import AbstractFeature

# Default parameters for this feature
from .parameters import *


# ##################################################################
# PERFORMANCE ACCUMULATOR
# ##################################################################

class PerformanceAccumulatorFeature(AbstractFeature):
    #
    # Columns created will be prefixed with this keyword
    PREFIX = "perf_acc"

    # ##################################################################
    # INIT
    # ##################################################################

    def __init__(self,
                 columns: list,
                 windows: list = None,
                 references: list = None,
                 prefix: str = None):
        """
        Feature an accumulator of the performance of 1 time span relative
        to another one. For example the return over the last 5 minutes

        :param columns: columns on which computation will be performed
        :param windows: rolling performance total (sum) computed over this window (short)
        :param references: EWMA are computed based on these reference decay rates
        :param prefix: Corresponding to the name of the column output
        """

        # Init super feature class
        if prefix is None:
            prefix = self.PREFIX
        super(PerformanceAccumulatorFeature, self).__init__(prefix=prefix)

        # Target columns on which we operate
        self.columns = self._sanitize_columns(columns=columns)

        # Spans over which calculations will be made
        self.windows = self._sanitize_spans(windows, default=PERF_ACC_WINDOWS)
        self.references = self._sanitize_spans(references, default=PERF_ACC_REFERENCES)

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
            "windows": self.windows,
            "references": self.references,
        }
        return params

    # ##################################################################
    # CORE METHODS
    # ##################################################################

    def __call__(self, df: pd.DataFrame):
        return self.apply(df=df, prefix=self.PREFIX, columns=self.columns)

    @classmethod
    def apply(cls, df: pd.DataFrame, prefix: str = None, columns: list = None,
              windows: list = None, references: list = None):

        # Ensure that parameters are present
        prefix = prefix or cls.PREFIX

        # Make some assertions
        all_spans = list(set(references + windows))
        cls._assert_before_applying(df=df, columns=columns, spans=all_spans)

        return cls.__compute_performance_accumulator(df=df,
                                                     prefix=prefix,
                                                     columns=columns,
                                                     windows=windows,
                                                     references=references)

    @classmethod
    def __compute_performance_accumulator(cls,
                                          df: pd.DataFrame,
                                          columns: list,
                                          prefix: str,
                                          windows: list,
                                          references: list):
        #
        # Pre-existing columns:
        # we will not overwrite/compute the pre-existing columns
        already_present = df.columns

        # Our target columns
        for col in columns:

            for w in windows:

                for r in references:

                    # target output column name
                    col_out = cls._prefixed(prefix, col, w, r)

                    # Already there...
                    if col_out in already_present:
                        logging.warning(f"Column '{col_out}' is already present, skipping transformation.")
                        continue

                    # Levels
                    levels = df[col].copy()

                    # Trend indicator
                    ewma_reference = df[col].ewm(com=r).mean()

                    # Difference in level
                    levels -= ewma_reference

                    # Add over the window size
                    res = levels.rolling(w).sum() / ewma_reference * (1.0 / w)

                    # Merge back into the df
                    res.name = col_out
                    df = df.merge(res, left_index=True, right_index=True)

        return df
