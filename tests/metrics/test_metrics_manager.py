import pytest

from gesund import MetricsManager


@pytest.fixture
def metrics_manager():
    """
    Fixture to initialize and return a metricsmanager instance

    :return: Instance of MetricsManager
    """
    return MetricsManager()


@pytest.mark.parametrize(
    "problem_type, metric_name",
    [
        ("classification", "mock_metric"),
        ("instance_segmentation", "mock_metric"),
        ("object_detection", "mock_metric"),
        pytest.param(
            "invalid_problem_type",
            "mock_metric",
            marks=pytest.mark.xfail(raises=KeyError),
        ),
    ],
)
def test_register_metrics(metrics_manager, problem_type, metric_name):
    """
    Test registering a metric function for the classification problem type

    :param metrics_manager: The MetricManager instance
    :type metrics_manager: MetricsManager
    :param problem_type: type of the problem
    :type problem_type: str
    :param metric_name: name of the metric
    :type metric_name: str

    :return: None
    """

    @metrics_manager.register_metrics(problem_type, metric_name)
    def mock_metric():
        pass

    if problem_type != "invalid_problem_type":
        assert metric_name in metrics_manager.metrics_store[problem_type]
        assert metrics_manager.metrics_store[problem_type][metric_name] == mock_metric
    else:
        pytest.fail("Expected Keyerror not raised")
