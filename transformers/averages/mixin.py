import logging
import json
from abc import ABC

import pandas as pd

from utilities.container import Spans


class AverageMixin(ABC):
    """
    Basic Average specific mixin class:
    only holds duplicate code snippets for the averages features.

    """

    @classmethod
    def json_loads(cls, payload):
        payload = "{ 'data': " + f"{payload}" + "}"
        return json.loads(str(payload).replace("'", '"')).get("data", [])

    @classmethod
    def _sanitize_avg_spans(cls, spans):
        if isinstance(spans, int) or isinstance(spans, float):
            spans = [spans, ]

        if isinstance(spans, str):
            #
            # When using column transformers, the lists
            # are pointing to the same ref, but strings do..
            #
            spans = cls.json_loads(spans)

        assert isinstance(spans, list) or isinstance(spans, Spans), "Spans must be passed as a list of integers."
        output = []
        for s in spans:
            try:
                output.append(int(s))  # cast should avoid typing issues
            except:
                output.append(int(s[0]))  # Scikit issue with clone sanity checks

        # return sorted(list(set(output)))
        return sorted(set(output))

    @classmethod
    def _sanitize_avg_columns(cls, columns=None):

        # No target columns means applied to all columns
        if columns is None:
            logging.warning(f"SMA Transformation will be applied to all columns (no target specified).")
            return None

        if isinstance(columns, str):
            #
            # When using column transformers, the lists
            # are pointing to the same ref, but strings do..
            #
            columns = cls.json_loads(columns)

        assert isinstance(columns, list), "Target columns must be passed as a list of strings."
        output = []
        for t in columns:
            output.append(str(t))  # cast should avoid typing issues
        # return list(set(output))
        return set(output)

    @classmethod
    def _assert_before_applying(cls, df: pd.DataFrame, columns: list, spans: list):

        # Must provide some columns
        columns = columns or df.columns

        # Take a reference
        df_columns = df.columns
        for c in columns:
            assert c in df_columns, f"Transformation cannot be applied on missing column {c}."

        for s in spans:
            assert s > 0, f"Transformation cannot be applied on negative span {s}."
