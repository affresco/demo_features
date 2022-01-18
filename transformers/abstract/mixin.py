from abc import ABC, abstractmethod

# Scientific
import pandas as pd

# Machine Learning
from sklearn.base import TransformerMixin


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
    # HELPER METHODS
    # ##################################################################

    @classmethod
    def _sanitize_columns(cls, df: pd.DataFrame, columns: list = None):

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

    @classmethod
    def _prefixed(cls, *args):
        return "_".join([str(a) for a in args])
