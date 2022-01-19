import logging
from abc import ABC

from .parameters import ATR_COLUMNS


class AverageTrueRangeMixin(ABC):

    @classmethod
    def _sanitize_atr_columns_mapping(cls, columns: list = None):
        if columns is None:
            return ATR_COLUMNS

        logging.warning(f"ATR cannot verify column accuracy, only their types.")
        for c in columns:
            assert isinstance(c, str), f"Columns must be provided as a list of strings."
        return columns

