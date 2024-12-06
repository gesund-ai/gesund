import pytest

from gesund import PlotManager
from gesund.core._exceptions import FunctionNotFoundError


@pytest.fixture
def plot_manager():
    """
    Fixture to initialize and return a plot manager instance

    :return: Instance of plot manager
    """
    return PlotManager()


@pytest.mark.parametrize(
    "plot_name",
    [("bar_plot"), ("line_chart")],
)
def test_register(plot_manager, plot_name):
    """
    Test registering a plot function for the given plot name

    :param plot_manager: The plot manager instance
    :type plot_manager: PlotManager
    :param plot_name: name of the plot
    :type plot_name: str

    :return: None
    """

    @plot_manager.register(plot_name)
    def mock_plot():
        pass

    assert plot_name in plot_manager.get_names()
    assert plot_manager[plot_name] == mock_plot


def test_non_registered(plot_manager):
    """
    Test a non registered function

    :param plot_manager: the plot manager instance
    :type plot_manager: PlotManager

    :return: None
    """
    with pytest.raises(FunctionNotFoundError):
        _ = plot_manager["random_fxn"]
