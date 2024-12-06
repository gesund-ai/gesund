import pytest

from gesund.core import MetricManager
from gesund.core._exceptions import FunctionNotFoundError, RegistrationNotAllowed


@pytest.fixture
def metrics_manager():
    """
    Fixture to initialize and return a metricsmanager instance

    :return: Instance of MetricsManager
    """
    return MetricManager()


@pytest.mark.parametrize(
    "problem_type, metric_name, fail_case",
    [
        ("classification", "mock_metric", False),
        ("instance_segmentation", "mock_metric", False),
        ("object_detection", "mock_metric", False),
        ("invalid_problem_type", "mock_metric", True),
    ],
)
def test_register_metrics(metrics_manager, problem_type, metric_name, fail_case):
    """
    Test registering a metric function for the given problem type

    :param metrics_manager: The MetricManager instance
    :type metrics_manager: MetricsManager
    :param problem_type: type of the problem
    :type problem_type: str
    :param metric_name: name of the metric
    :type metric_name: str
    :param fail_case: case if it should fail
    :type fail_case: bool

    :return: None
    """
    register_key = f"{problem_type}.{metric_name}"
    if fail_case:
        with pytest.raises(RegistrationNotAllowed):

            @metrics_manager.register(register_key)
            def mock_metric():
                pass

        with pytest.raises(FunctionNotFoundError):
            _ = metrics_manager[register_key]
    else:

        @metrics_manager.register(register_key)
        def mock_metric():
            pass

        assert metrics_manager[register_key] == mock_metric
        assert metric_name in metrics_manager.get_names(problem_type)
