from typing import Callable

from gesund.core._managers.base import GenericPMManager


class MetricManager(GenericPMManager[Callable]):
    pass


metric_manager = MetricManager()
