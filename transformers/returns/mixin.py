import logging
import json
from abc import ABC

import pandas as pd


class ReturnsMixin(ABC):
    """
    Basic returns-specific mixin class:
    only holds duplicate code snippets for the returns-related features.

    """
    @classmethod
    def _sanitize_ret_spans(cls, spans):
        if isinstance(spans, int) or isinstance(spans, float):
            spans = [spans, ]

        assert isinstance(spans, list), "Spans must be passed as a list of integers."
        output = []
        for s in spans:
            output.append(int(s))  # cast should avoid typing issues
        return sorted(list(set(output)))

    @classmethod
    def _sanitize_ret_columns(cls, columns=None):

        # No target columns means applied to all columns
        if columns is None:
            logging.warning(f"SMA Transformation will be applied to all columns (no target specified).")
            return None

        assert isinstance(columns, list), "Target columns must be passed as a list of strings."
        output = []
        for t in columns:
            output.append(str(t))  # cast should avoid typing issues
        return list(set(output))

    @classmethod
    def _assert_before_applying(cls, df: pd.DataFrame, columns: list, spans: list):

        # Must provide some columns
        columns = columns or df.columns

        # Take a reference
        df_columns = df.columns
        for c in columns:
            assert c in df_columns, f"Transformation cannot be applied on missing column {c}."
