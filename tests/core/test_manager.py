import pytest

from typing import Callable

from gesund.core._managers import GenericPMManager
from gesund.core._exceptions import FunctionNotFoundError


@pytest.fixture
def generic_manager():
    return GenericPMManager[Callable]()


def test_register_and_get(generic_manager):
    """
    A function to test register and get function of the generic manager class

    :param generic_manager: Generic Manager Class instance
    :type generic_manager: GenericPMManager

    :return: None
    """

    @generic_manager.register("test_func")
    def test_func():
        pass

    assert "test_func" in generic_manager.get_names()
    assert generic_manager["test_func"] == test_func


def test_non_existent_fxn(generic_manager):
    """
    A function to test raise keyerror function of the generic manager class

    :param  generic_manager: Generic manager class instance
    :type generic_manager: GenericPMManager

    :return: None
    """

    with pytest.raises(FunctionNotFoundError):
        generic_manager["random_fxn"]
