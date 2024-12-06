from typing import Callable

from gesund.core._managers.base import GenericPMManager


class PlotManager(GenericPMManager[Callable]):
    pass


plot_manager = PlotManager()
