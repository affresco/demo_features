import logging
import json
from abc import ABC

import pandas as pd

from utilities.container import Spans


class BollingerBandsMixin(ABC):
    """
    Basic Bollinger Bands specific mixin class:
    only holds duplicate code snippets for the averages features.

    """

    # ##################################################################
    # HELPER METHODS: SANITIZERS
    # ##################################################################

    @classmethod
    def _sanitize_std_devs(cls, std_devs):
        if isinstance(std_devs, int) or isinstance(std_devs, float):
            std_devs = [std_devs, ]
        error_msg = "Standard deviations must be passed as a list of floats."
        assert isinstance(std_devs, list) or isinstance(std_devs, Spans), error_msg
        output = [float(s) for s in std_devs]
        return sorted(list(set(output)))

