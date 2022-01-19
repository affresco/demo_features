import logging
import random

# Scientific
import pandas as pd

# Base class for all features
from transformers.abstract.mixin import AbstractFeature

# Local modules relating to boll bands
from .mixin import AverageTrueRangeMixin


# ##################################################################
# AVERAGE TRUE RANGE
# ##################################################################

class AverageTrueRangeFeature(AbstractFeature, AverageTrueRangeMixin):
    #
    # Columns created will be prefixed with this keyword
    PREFIX = "atr"

    # ##################################################################
    # INIT
    # ##################################################################

    def __init__(self,
                 spans: list,
                 columns: dict = None,
                 prefix: str = None,
                 ffill: bool = True,
                 bfill: bool = True):
        """
        Feature a normalized computation of the Average True Range (ATR) of Wilder suitable for machine learning.

        :param columns: columns on which computation will be performed
        :param prefix: Added to output col for identifications
        :param ffill: Fill forward after computation
        :param bfill: Fill backward after computation
        """

        # Init super feature class
        if prefix is None:
            prefix = self.PREFIX
        super(AverageTrueRangeFeature, self).__init__(prefix=prefix)

        # Target columns on which we operate
        self.columns = self._sanitize_atr_columns_mapping(columns=columns)

        # Computation spans
        self.spans = self._sanitize_spans(spans=spans)

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
            logging.warning(f"Argument 'y' of ATR transformation will be ignored.")
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
        return self.apply(df=df,
                          prefix=self.PREFIX,
                          columns=self.columns,
                          spans=self.spans,
                          ffill=self.ffill,
                          bfill=self.bfill)

    @classmethod
    def apply(cls, df: pd.DataFrame, prefix: str = None, columns: dict = None, spans: list = None,
              ffill: bool = True, bfill: bool = True):

        # Ensure that parameters are present
        prefix = prefix or cls.PREFIX

        # Make some assertions
        # cls._assert_before_applying(df=df, columns=columns, spans=spans)

        return cls.__compute_atr(df=df,
                                 columns=columns,
                                 spans=spans,
                                 prefix=prefix,
                                 ffill=ffill,
                                 bfill=bfill)

    @classmethod
    def __compute_true_range(cls, df: pd.DataFrame, high_col: str, low_col: str, close_col: str):
        df['atr0'] = abs(df[high_col] - df[low_col])
        df['atr1'] = abs(df[high_col] - df[close_col].shift())
        df['atr2'] = abs(df[low_col] - df[close_col].shift())
        return df[['atr0', 'atr1', 'atr2']].max(axis=1)

    @classmethod
    def __compute_atr(cls, df: pd.DataFrame, columns: dict, spans: list, prefix: str,
                      ffill: bool = True, bfill: bool = True):

        # Extract columns from mapping
        high_col = columns.get("high", "high")
        assert high_col in df.columns

        low_col = columns.get("low", "low")
        assert low_col in df.columns

        close_col = columns.get("close", "close")
        assert close_col in df.columns

        # Our column
        true_range = f"tr_tmp_{random.randint(11111111111, 99999999999)}"

        # First compute the True Range for perpetual
        df.loc[:, true_range] = cls.__compute_true_range(df=df[[high_col, low_col, close_col]].copy(),
                                                         high_col=high_col,
                                                         low_col=low_col,
                                                         close_col=close_col)
        df.loc[:, true_range] /= df[close_col]

        # Current columns: we will not overwrite the pre-existing columns
        already_present = df.columns

        # Our target spans
        for s in spans:

            # Projected column name
            col_out = cls._prefixed(prefix, s)

            if col_out in already_present:
                logging.warning(f"Column '{col_out}' is already present, skipping transformation.")
                continue

            # Compute transformation
            df.loc[:, col_out] = df[true_range].ewm(alpha=1 / s, adjust=True).mean()

            # Forward fill
            if ffill:
                df[col_out].ffill(inplace=True)

            # Backward fill
            if bfill:
                df[col_out].bfill(inplace=True)

        # Clean up after use...
        if true_range in df.columns:
            df.drop(columns=[true_range, ], inplace=True)
        return df
