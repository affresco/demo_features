import copy
import logging

# Scientific
import pandas as pd

# Base class for all features
from transformers.abstract.mixin import AbstractFeature

# Local modules relating to averaging
from .mixin import AverageMixin


# ##################################################################
# EXPONENTIALLY WEIGHTED MOVING AVERAGES (EWMA)
# ##################################################################

class EwmaFeature(AbstractFeature, AverageMixin):
    #
    # Columns created will be prefixed with this keyword
    PREFIX_EWMA = "ewma"

    # ##################################################################
    # INIT
    # ##################################################################

    def __init__(self,
                 spans: list,
                 columns: list = None,
                 prefix: str = None,
                 ffill: bool = True,
                 bfill: bool = True):

        # Init super feature class
        if prefix is None:
            prefix = self.PREFIX_EWMA
        super(EwmaFeature, self).__init__(prefix=prefix)

        # Target columns on which we operate
        self.columns = self._sanitize_avg_columns(columns=columns)

        # Averages spans
        self.spans = self._sanitize_avg_spans(spans=spans)
        #
        # Initial span of data to be discarded
        self.warmup = max(self.spans)

        # Filling missing data
        #
        # Forward fill first
        self.ffill = bool(ffill)
        #
        # Then backfill (see warmup parameter)
        self.bfill = bool(bfill)

    # ##################################################################
    # SKLEARN TRANSFORMER MIXIN IMPLEMENTATION
    # ##################################################################

    def fit(self, *args, **kwargs):
        return self  # nothing to do here

    def transform(self, X, y=None, **fit_params):
        assert isinstance(X, pd.DataFrame), "Transform must be a DataFrame"
        if y is not None:
            logging.warning(f"Argument 'y' of EWMA transformation will be ignored.")
        return self(df=X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y, **fit_params)

    def get_params(self, *args, **kwargs):
        params = {
            "prefix": self.prefix,
            "spans": self.spans,
            "columns": self.columns,
            "ffill": self.ffill,
            "bfill": self.bfill,
        }
        return params

    # ##################################################################
    # CORE METHODS
    # ##################################################################

    def __call__(self, df: pd.DataFrame):
        return self.apply(df=df, prefix=self.PREFIX_EWMA, columns=self.columns, spans=self.spans,
                          ffill=self.ffill, bfill=self.bfill)

    @classmethod
    def apply(cls, df: pd.DataFrame, prefix: str = None, columns: list = None, spans: list = None,
              ffill: bool = True, bfill: bool = True):

        # Ensure that parameters are present
        prefix = prefix or cls.PREFIX_EWMA

        # Make some assertions
        cls._assert_before_applying(df=df, columns=columns, spans=spans)

        return cls.__compute_rolling_ewma(df=df, columns=columns, spans=spans, prefix=prefix, ffill=ffill, bfill=bfill)

    @classmethod
    def __compute_rolling_ewma(cls, df: pd.DataFrame, columns: list, spans: list, prefix: str,
                               ffill: bool = True, bfill: bool = True):
        #
        # Current columns: we will not overwrite the pre-existing columns
        already_present = df.columns

        # Our target columns
        for col in columns:

            spans = [s for s in spans if cls._prefixed(prefix, col, s) not in already_present]

            # Our target spans
            for s in spans:

                # Projected column name
                ewma_col = cls._prefixed(prefix, col, s)

                if ewma_col in already_present:
                    logging.warning(f"Column '{ewma_col}' is already present, skipping transformation.")
                    continue

                # Compute transformation
                res = df[col].ewm(com=s, adjust=True).mean()

                # Add to existing df (this avoids copy warnings)
                res.name = ewma_col
                df = df.merge(res, left_index=True, right_index=True)

                # Forward fill
                if ffill:
                    df[ewma_col].ffill(inplace=True)

                # Backward fill
                if bfill:
                    df[ewma_col].bfill(inplace=True)

        return df
