import logging
from abc import ABC, abstractmethod

# Scientific
import pandas as pd

# Machine Learning
from sklearn.base import TransformerMixin

# Local
from utilities.container import Spans


# ##################################################################
# ABSTRACT FEATURE CLASS
# ##################################################################


class AbstractFeature(TransformerMixin, ABC):

    def __init__(self, prefix: str):
        self.__prefix: str = str(prefix)

    # ##################################################################
    # PROPERTIES
    # ##################################################################

    @property
    def prefix(self):
        return self.__prefix

    # ##################################################################
    # SKLEARN TRANSFORMER MIXIN IMPLEMENTATION
    # ##################################################################

    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError(f"Scikit Transformer 'fit' method must be implemented by derived classes.")

    @abstractmethod
    def transform(self, X, y=None, **fit_params):
        raise NotImplementedError(f"Scikit Transformer 'transform' method must be implemented by derived classes.")

    @abstractmethod
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform()

    # ##################################################################
    # COLUMN PREFIX METHODS
    # ##################################################################

    @classmethod
    def _prefixed(cls, *args):
        return "_".join([str(a) for a in args])

    # ##################################################################
    # HELPER METHODS: SANITIZERS
    # ##################################################################

    @classmethod
    def _sanitize_spans(cls, spans):
        if isinstance(spans, int) or isinstance(spans, float):
            spans = [spans, ]

        assert isinstance(spans, list) or isinstance(spans, Spans), "Spans must be passed as a list of integers."
        output = []
        for s in spans:
            try:
                output.append(int(s))  # cast should avoid typing issues
            except Exception as exc:
                logging.warning(f"Span type casting produced an exception: {exc}")
                output.append(int(s[0]))  # Scikit issue with clone sanity checks (see keras issue 13586)

        return sorted(list(set(output)))

    @classmethod
    def _sanitize_columns(cls, columns=None, df: pd.DataFrame = None):

        # No target columns means applied to all columns
        if columns is None and df is None:
            logging.warning(f"Transformation will be applied to all columns (no target specified).")
            return None

        if columns is None and df is not None:
            logging.warning(f"Transformation will be applied to all columns (dataframe was provided).")
            return df.columns

        # Make sure the type is legitimate
        assert isinstance(columns, list), "Target columns must be passed as a list of strings."

        # Clean
        output = list(set([str(t) for t in columns]))

        if df is not None:
            for c in output:
                assert c in df.columns, f"Required column {c} missing from dataframe"

        return output

    @classmethod
    def _assert_before_applying(cls, df: pd.DataFrame, columns: list, spans: list = None):

        # Must provide some columns
        columns = columns or df.columns

        # Take a reference
        df_columns = df.columns
        for c in columns:
            assert c in df_columns, f"Transformation cannot be applied on missing column {c}."

        if spans is not None:
            for s in spans:
                assert s > 0, f"Transformation cannot be applied on negative span {s}."

    # ##################################################################
    # LEGACY METHODS
    # ##################################################################

    @classmethod
    def _legacy_sanitize_columns(cls, df: pd.DataFrame, columns: list = None):

        # Marshall to list of strings
        columns = [str(columns), ] if not isinstance(columns, list) else [str(c) for c in columns]

        # Check that the columns are valid
        if columns:
            already_present = df.columns
            missing = [c for c in columns if c not in already_present]
            assert not len(missing)
        else:
            columns = df.columns

        return list(set(columns))
