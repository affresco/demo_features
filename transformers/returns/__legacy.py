import math
import pandas as pd
import numpy as np

import statsmodels.tsa.stattools as stat_tools

# Fractional Differentiation Package
from fracdiff import FracdiffStat, fdiff

from transform.base.estimator import BaseFeature

# ##################################################################
# FRACTIONS
# ##################################################################

FRACTIONS = [0.4, 0.5, 0.6, 0.7, 0.8]

ESTIMATE_DEFAULT_ROUNDING = 0.01

# This is safe: order 1.0 has
# to achieve stationarity...
ESTIMATE_FRAC_ORDER_FLOOR = 0.6
ESTIMATE_FRAC_ORDER_ERROR = 0.9

MINUTES_PER_DAY = 24 * 60
DAYS_TO_ESTIMATE_FRAC_DIFF_ORDER = 7


# ##################################################################
# ESTIMATOR
# ##################################################################

class ReturnFeature(BaseFeature):
    #
    PREFIX_FRACTIONAL = "frac"
    PREFIX_LINEAR = "lin"
    PREFIX_LOG = "log"
    SUFFIX_RETURNS = "ret"
    RETURN_TYPES = {
        "lin": ["lin", "linear", ],
        "log": ["log", ],
        "frac": ["frac", "fractional", ],
    }

    # ##################################################################
    # ESTIMATE THE MOST MEMORY-PRESERVING ORDER OF FRAC DIFFERENTIATION
    # ##################################################################

    @classmethod
    def find_max_preserving_order(cls, df: pd.DataFrame,
                                  target: str,
                                  from_idx: int = None,
                                  to_idx: int = None,
                                  rounding: float = ESTIMATE_DEFAULT_ROUNDING):

        # From index (start)
        from_idx = from_idx or 0
        from_idx = max(0, from_idx)

        # To index (end)
        to_idx = to_idx or len(df)
        assert to_idx > from_idx

        # Sanity check: col must be present
        assert target in df.columns

        # Extract the data related to the target column
        data = df[target].iloc[from_idx:to_idx].to_numpy().reshape(-1, 1)

        # Compute the minimum order preserving the stationarity
        fs = FracdiffStat(mode="valid")
        data_diff = fs.fit_transform(data)

        # Some validations
        _, p_value, _, _, _, _ = stat_tools.adfuller(data_diff.reshape(-1))
        corr = np.corrcoef(data[-data_diff.size:, 0], data_diff.reshape(-1))[0][1]

        print("* Order: {:.4f}".format(fs.d_[0]))
        print("* ADF p-value: {:.2f} %".format(100 * p_value))
        print("* Correlation with the original time-series: {:.2f}".format(corr))

        # Extract the data from the class
        frac_order = fs.d_[0]

        # We keep a certain level of diff to account for
        # estimation problems / errors
        if frac_order < ESTIMATE_FRAC_ORDER_FLOOR:
            print(f"Estimate of the fractional differentiation order was too low ({frac_order}), "
                  f"returning default error level ({ESTIMATE_FRAC_ORDER_ERROR}).")
            return ESTIMATE_FRAC_ORDER_ERROR

        # Returns frac diff order (0.0 < ord < 1.0)
        # Without any rounding specified
        if rounding is None:
            return frac_order

        # Sanitize rounding and round
        rounding = max(min(1.0, rounding), ESTIMATE_DEFAULT_ROUNDING)

        # return max(ESTIMATE_FRAC_ORDER_FLOOR, min(1.0, math.ceil(fs.d_[0] / rounding) * rounding))
        return max(ESTIMATE_FRAC_ORDER_FLOOR, min(1.0, math.ceil(fs.d_[0] / rounding) * rounding))

    # ##################################################################
    # FACILITATION FOR ADDING STANDARD RETURN COLUMNS/SERIES
    # ##################################################################

    @classmethod
    def estimate(cls, df: pd.DataFrame, target: str, method: str = "linear"):

        method = cls.selector(method)

        if "lin" in method:
            return cls.__compute_single_linear_return(df=df, target=target)

        if "log" in method:
            return cls.__compute_single_log_return(df=df, target=target)

        if "frac" in method:
            return cls.__compute_single_frac_return(df=df, target=target)

        raise NotImplemented(f"Cannot compute return {method} not implemented.")

    # ##################################################################
    # SELECTOR
    # ##################################################################

    @classmethod
    def selector(cls, method: str):
        method = str(method).lower()
        for rt, synonyms in cls.RETURN_TYPES.items():
            if rt in method:
                return rt
            for s in synonyms:
                if s in method:
                    return rt
        raise NotImplemented(f"Method {method} not implemented.")

    # ##################################################################
    # PRE-DEFINED PREFIXES
    # ##################################################################

    @classmethod
    def prefix_lin(cls, target):
        return "_".join([cls.PREFIX_LINEAR, cls.SUFFIX_RETURNS, target])

    @classmethod
    def prefix_log(cls, target):
        return "_".join([cls.PREFIX_LOG, cls.SUFFIX_RETURNS, target])

    @classmethod
    def prefix_frac(cls, target, n):
        return "_".join([cls.PREFIX_FRACTIONAL, str(n), cls.SUFFIX_RETURNS, target])

    # ##################################################################
    # ESTIMATOR FOR HIGHS
    # ##################################################################

    @classmethod
    def rescale(cls, df: pd.DataFrame, target: str = "close", inception: float = 1.0, name: str = None):
        df = cls.__rescale(df=df, target=target, inception=inception, name=name)
        return df

    @classmethod
    def __rescale(cls, df: pd.DataFrame, target: str, inception: float = 1.0, ohlc: bool = True, name: str = None):
        # Find a suitable name for the new column
        if name:
            col, tmp = name, "tmp"
        else:
            col, tmp = cls._prefixed(target, 100), "tmp"

        # Start by re-scaling on the closing levels
        df[tmp] = df[target] / df[target].shift(1)
        df[tmp] = df[tmp].fillna(1.0)
        df[col] = df[tmp].cumprod() * inception
        df.drop(columns=[tmp, ], inplace=True)

        # Scale the other columns
        if not ohlc:
            return df

        # Find which ols this is applicable to
        other_cols = ["open", "high", "low", "close"]
        other_cols = [oc for oc in other_cols if target not in oc]

        for oc in other_cols:
            df[cls._prefixed(oc, 100)] = (df[oc] / df[target]) * df[col]

        return df

    # ##################################################################
    # CORE METHODS
    # ##################################################################

    @classmethod
    def __compute_single_linear_return(cls, df: pd.DataFrame, target: str):

        # col = cls._prefixed(cls.PREFIX_LINEAR, target)
        col = cls.prefix_lin(target)

        # Already there...
        if col in df.columns:
            print(f"WARNING: Column already present in dataframe: {col}. Skipping linear returns computation.")
            return df

        res = df[target] / df[target].shift(1) - 1.0
        res.name = col
        df = df.merge(res, left_index=True, right_index=True)
        df[col].fillna(0.0, inplace=True)
        return df

    @classmethod
    def __compute_single_log_return(cls, df: pd.DataFrame, target: str):
        #
        # col = cls._prefixed(cls.PREFIX_LOG, target)
        col = cls.prefix_log(target)

        # Already there...
        if col in df.columns:
            print(f"WARNING: Column already present in dataframe: {col}. Skipping log returns computation.")
            return df

        res = np.log(df[target] / df[target].shift(1))
        res.name = col
        df = df.merge(res, left_index=True, right_index=True)
        df[col].fillna(0.0, inplace=True)
        return df

    @classmethod
    def __compute_single_frac_return(cls, df: pd.DataFrame, target: str, n: list = None):

        if n is None:
            k, days = len(df), DAYS_TO_ESTIMATE_FRAC_DIFF_ORDER
            st, nd = max(0, k - days * MINUTES_PER_DAY), k
            print(f"Estimating fractional differentiation order from last {(nd-st)/MINUTES_PER_DAY} days of history.")
            n = cls.find_max_preserving_order(df=df, target=target, from_idx=st, to_idx=nd)

        if not isinstance(n, list):
            n = [n, ]

        tgt_series, results = df[target], {}

        for ni in n:

            # col = cls._prefixed(cls.PREFIX_LOG, target)
            col = cls.prefix_frac(target, ni)

            # Already there...
            if col in df.columns:
                print(f"WARNING: Column already present in dataframe: {col}. Skipping fractional returns computation.")
                continue

            try:
                diff = fdiff(tgt_series, n=ni, mode="valid")
                res = pd.Series(diff, index=df.index[-diff.size:], name=col)
                results[ni] = res

            except Exception as exc:
                print(exc)

        # Concatenate the series and merge with our dataset
        df = df.merge(pd.concat(results.values(), axis=1), left_index=True, right_index=True)
        return df
