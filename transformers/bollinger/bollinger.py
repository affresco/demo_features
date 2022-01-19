import logging

# Scientific
import pandas as pd

# Base class for all features
from transformers.abstract.mixin import AbstractFeature

# Local modules relating to boll bands
from .mixin import BollingerBandsMixin


# ##################################################################
# BOLLINGER BANDS
# ##################################################################

class BollingBandsFeature(AbstractFeature, BollingerBandsMixin):
    #
    # Columns created will be prefixed with this keyword
    PREFIX = "boll"

    # ##################################################################
    # INIT
    # ##################################################################

    def __init__(self,
                 spans: list,
                 std_devs: list,
                 columns: list = None,
                 prefix: str = None,
                 ffill: bool = True,
                 bfill: bool = True):
        """
        Feature a normalized computation of the Bollinger band suitable for machine learning.

        :param columns: columns on which computation will be performed
        :param spans: List of spans as integers
        :param std_devs: Number of standard deviations (half-width of the band)
        :param prefix: Added to output col for identifications
        :param ffill: Fill forward after computation
        :param bfill: Fill backward after computation
        """

        # Init super feature class
        if prefix is None:
            prefix = self.PREFIX
        super(BollingBandsFeature, self).__init__(prefix=prefix)

        # Target columns on which we operate
        self.columns = self._sanitize_columns(columns=columns)

        # Computation spans
        self.spans = self._sanitize_spans(spans=spans)

        # Computation standard deviation
        self.std_devs = self._sanitize_std_devs(std_devs=std_devs)

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
            logging.warning(f"Argument 'y' of Bollinger bands transformation will be ignored.")
        return self(df=X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y, **fit_params)

    def get_params(self, *args, **kwargs):
        params = {
            "prefix": self.prefix,
            "spans": self.spans,
            "std_devs": self.std_devs,
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
                          std_devs=self.std_devs,
                          ffill=self.ffill,
                          bfill=self.bfill)

    @classmethod
    def apply(cls, df: pd.DataFrame, prefix: str = None, columns: list = None, spans: list = None,
              std_devs: list = None,
              ffill: bool = True, bfill: bool = True):

        # Ensure that parameters are present
        prefix = prefix or cls.PREFIX

        # Make some assertions
        cls._assert_before_applying(df=df, columns=columns, spans=spans)

        return cls.__compute_bollinger_bands(df=df,
                                             columns=columns,
                                             spans=spans,
                                             std_devs=std_devs,
                                             prefix=prefix,
                                             ffill=ffill,
                                             bfill=bfill)

    @classmethod
    def __compute_bollinger_bands(cls, df: pd.DataFrame, columns: list, spans: list, std_devs: list, prefix: str,
                                  ffill: bool = True, bfill: bool = True):
        """
        Compute a normalized version of the Bollinger band indicator for machine learning applications.

        The result is a normalized position inside the bands, that is

                           target_column(t) - SMA_target_column(t)
            boll_norm(t) = ----------------------------------------
                              2 x target_column_std_deviation(t)

        :param df: pd.DataFrame
        :param columns: columns on which computation will be performed
        :param spans: List of spans as integers
        :param std_devs: Number of standard deviations (half-width of the band)
        :param prefix: Added to output col for identifications
        :param ffill: Fill forward after computation
        :param bfill: Fill backward after computation
        :return: pd.DataFrame with added columns
        """
        #
        # Current columns: we will not overwrite the pre-existing columns
        already_present = df.columns

        # Our target columns
        for col in columns:

            # Our target spans
            for s in spans:

                # Our target std deviations
                for nb_std in std_devs:

                    # Projected column name
                    boll_col = cls._prefixed(prefix, col, s, nb_std)

                    if boll_col in already_present:
                        logging.warning(f"Column '{boll_col}' is already present, skipping transformation.")
                        continue

                    # Compute transformation
                    prices = df[col]
                    sma = prices.rolling(s).mean()
                    one_std = prices.rolling(s).std()
                    boll_normalized = (df[col] - sma) / (one_std * 2.0 * nb_std)

                    # Add to existing df (this avoids copy warnings)
                    boll_normalized.name = boll_col
                    df = df.merge(boll_normalized, left_index=True, right_index=True)

                    # Forward fill
                    if ffill:
                        df[boll_col].ffill(inplace=True)

                    # Backward fill
                    if bfill:
                        df[boll_col].bfill(inplace=True)

        return df
