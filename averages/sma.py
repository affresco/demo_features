import logging
import pandas as pd

# Base class for all features
from _transformer.scikit_mixin import BaseFeature

# Local modules relating to averaging
from .parameters import SMA_SPANS
from .mixin import AverageMixin


# ##################################################################
# SIMPLE MOVING AVERAGES (SMA)
# ##################################################################

class SmaFeature(BaseFeature, AverageMixin):
    #
    # Columns created will be prefixed with this keyword
    PREFIX_SMA = "sma"

    # ##################################################################
    # INIT
    # ##################################################################

    def __init__(self, spans: list = SMA_SPANS, columns: list = None, ffill: bool = True, bfill: bool = True):

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
            logging.warning(f"Argument 'y' of SMA transformation will be ignored.")
        return self(df=X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y, **fit_params)

    # ##################################################################
    # CORE
    # ##################################################################

    def __call__(self, df: pd.DataFrame):
        return self.apply(df=df, prefix=self.PREFIX_SMA, columns=self.columns, spans=self.spans)

    @classmethod
    def apply(cls, df: pd.DataFrame, columns: list, spans: list, prefix: str = None,
              ffill: bool = True, bfill: bool = True):

        # Ensure that parameters are present
        prefix = prefix or cls.PREFIX_SMA

        # Make some assertions
        cls._assert_before_applying(df=df, columns=columns, spans=spans)

        return cls.__add_rolling_sma(df=df, columns=columns, spans=spans, prefix=prefix, ffill=ffill, bfill=bfill)

    @classmethod
    def __add_rolling_sma(cls, df: pd.DataFrame, columns: list, spans: list, prefix: str,
                          ffill: bool = True, bfill: bool = True):
        #
        # Current columns: we will not overwrite the pre-existing columns
        already_present = df.columns

        # Our target columns
        for col in columns:

            # Our target spans
            for s in spans:

                # Projected column name
                sma_col = cls._prefixed(col, prefix, s)

                if sma_col in already_present:
                    logging.warning(f"Column '{sma_col}' is already present, skipping transformation.")
                    continue

                # Compute transformation
                df[sma_col] = df[col].rolling(s).mean()

                # Forward fill
                if ffill:
                    df[sma_col].ffill(inplace=True)

                # Backward fill
                if bfill:
                    df[sma_col].bfill(inplace=True)

        return df
