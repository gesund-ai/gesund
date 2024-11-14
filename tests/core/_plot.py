# FILE: tests/core/test_plot.py

import pytest
import os
from unittest.mock import MagicMock
from typing import Dict, List, Any

from gesund.core._plot import CommonPlots
from gesund.core import UserInputParams, UserInputData


@pytest.fixture
def common_plots() -> CommonPlots:
    return CommonPlots()


@pytest.fixture
def mock_cls_driver(common_plots: CommonPlots) -> MagicMock:
    mock = MagicMock()
    common_plots.cls_driver = mock
    return mock


@pytest.fixture
def mock_obj_driver(common_plots: CommonPlots) -> MagicMock:
    mock = MagicMock()
    common_plots.obj_driver = mock
    return mock


@pytest.fixture
def mock_seg_driver(common_plots: CommonPlots) -> MagicMock:
    mock = MagicMock()
    common_plots.seg_driver = mock
    return mock


def test_common_plots_initialization(common_plots: CommonPlots) -> None:
    """Test that all driver attributes are initialized to None."""
    assert common_plots.cls_driver is None
    assert common_plots.obj_driver is None
    assert common_plots.seg_driver is None


def test_class_distribution_plot(common_plots: CommonPlots, mock_cls_driver: MagicMock) -> None:
    """Test the _class_distribution method calls the cls_driver's _plot_class_distributions correctly."""
    metrics: Dict[str, int] = {"classA": 100, "classB": 150}
    threshold: float = 0.5
    save_path: str = "./plots/class_distribution.png"

    common_plots._class_distribution(metrics, threshold, save_path)

    mock_cls_driver._plot_class_distributions.assert_called_once_with(metrics, threshold, save_path)


def test_classification_blind_spot_plot(common_plots: CommonPlots, mock_cls_driver: MagicMock) -> None:
    """Test the _classification_blind_spot method calls the cls_driver's _plot_blind_spot correctly."""
    metrics: Dict[str, float] = {"accuracy": 0.95}
    class_types: List[str] = ["classA", "classB"]
    save_path: str = "./plots/classification_blind_spot.png"

    common_plots._classification_blind_spot(metrics, class_types, save_path)

    mock_cls_driver._plot_blind_spot.assert_called_once_with(metrics, class_types, save_path)


def test_class_performance_by_threshold_plot(common_plots: CommonPlots, mock_cls_driver: MagicMock) -> None:
    """Test the _class_performance_by_threshold method calls the cls_driver's _plot_class_performance_by_threshold correctly."""
    metrics: Dict[str, float] = {"precision": 0.8, "recall": 0.75}
    graph_type: str = "graph_1"
    threshold: float = 0.6
    save_path: str = "./plots/class_performance_by_threshold.png"

    common_plots._class_performance_by_threshold(metrics, graph_type, threshold, save_path)

    mock_cls_driver._plot_class_performance_by_threshold.assert_called_once_with(
        graph_type, metrics, threshold, save_path
    )


def test_roc_statistics_plot(common_plots: CommonPlots, mock_cls_driver: MagicMock) -> None:
    """Test the _roc_statistics method calls the cls_driver's _plot_roc_statistics correctly."""
    roc_class: List[str] = ["classA", "classB"]
    save_path: str = "./plots/roc_statistics.png"

    common_plots._roc_statistics(roc_class, save_path)

    mock_cls_driver._plot_roc_statistics.assert_called_once_with(roc_class, save_path)


def test_precision_recall_statistics_plot(common_plots: CommonPlots, mock_cls_driver: MagicMock) -> None:
    """Test the _precision_recall_statistics method calls the cls_driver's _plot_precision_recall_statistics correctly."""
    pr_class: List[str] = ["classA", "classB"]
    save_path: str = "./plots/precision_recall_statistics.png"

    common_plots._precision_recall_statistics(pr_class, save_path)

    mock_cls_driver._plot_precision_recall_statistics.assert_called_once_with(pr_class, save_path)


def test_classification_confidence_histogram_plot(common_plots: CommonPlots, mock_cls_driver: MagicMock) -> None:
    """Test the _classification_confidence_histogram method calls the cls_driver's _plot_confidence_histogram correctly."""
    confidence_histogram_args: List[Any] = ["arg1", "arg2"]
    save_path: str = "./plots/classification_confidence_histogram.png"

    common_plots._classification_confidence_histogram(confidence_histogram_args, save_path)

    mock_cls_driver._plot_confidence_histogram.assert_called_once_with(confidence_histogram_args, save_path)


def test_classification_overall_metrics_plot(common_plots: CommonPlots, mock_cls_driver: MagicMock) -> None:
    """Test the _classification_overall_metrics method calls the cls_driver's _plot_overall_metrics correctly."""
    overall_metrics_args: List[str] = ["metric1", "metric2"]
    save_path: str = "./plots/classification_overall_metrics.png"

    common_plots._classification_overall_metrics(overall_metrics_args, save_path)

    mock_cls_driver._plot_overall_metrics.assert_called_once_with(overall_metrics_args, save_path)


def test_confusion_matrix_plot(common_plots: CommonPlots, mock_cls_driver: MagicMock) -> None:
    """Test the _confusion_matrix method calls the cls_driver's _plot_confusion_matrix correctly."""
    save_path: str = "./plots/confusion_matrix.png"

    common_plots._confusion_matrix(save_path)

    mock_cls_driver._plot_confusion_matrix.assert_called_once_with(save_path)


def test_prediction_dataset_distribution_plot(common_plots: CommonPlots, mock_cls_driver: MagicMock) -> None:
    """Test the _prediction_dataset_distribution method calls the cls_driver's _plot_prediction_dataset_distribution correctly."""
    save_path: str = "./plots/prediction_dataset_distribution.png"

    common_plots._prediction_dataset_distribution(save_path)

    mock_cls_driver._plot_prediction_dataset_distribution.assert_called_once_with(save_path)


def test_most_confused_bar_plot(common_plots: CommonPlots, mock_cls_driver: MagicMock) -> None:
    """Test the _most_confused_bar method calls the cls_driver's _plot_most_confused_bar correctly."""
    save_path: str = "./plots/most_confused_bar.png"

    common_plots._most_confused_bar(save_path)

    mock_cls_driver._plot_most_confused_bar.assert_called_once_with(save_path)


def test_confidence_histogram_scatter_distribution_plot(common_plots: CommonPlots, mock_cls_driver: MagicMock) -> None:
    """Test the _confidence_histogram_scatter_distribution method calls the cls_driver's _plot_confidence_histogram_scatter_distribution correctly."""
    save_path: str = "./plots/confidence_histogram_scatter_distribution.png"

    common_plots._confidence_histogram_scatter_distribution(save_path)

    mock_cls_driver._plot_confidence_histogram_scatter_distribution.assert_called_once_with(save_path)


def test_lift_chart_plot(common_plots: CommonPlots, mock_cls_driver: MagicMock) -> None:
    """Test the _lift_chart method calls the cls_driver's _plot_lift_chart correctly."""
    save_path: str = "./plots/lift_chart.png"

    common_plots._lift_chart(save_path)

    mock_cls_driver._plot_lift_chart.assert_called_once_with(save_path)


def test_object_detection_blind_spot_plot(common_plots: CommonPlots, mock_obj_driver: MagicMock) -> None:
    """Test the _object_detection_blind_spot method calls the obj_driver's _plot_blind_spot correctly."""
    blind_spot_args: List[str] = ["arg1", "arg2"]
    save_path: str = "./plots/object_detection_blind_spot.png"

    common_plots._object_detection_blind_spot(blind_spot_args, save_path)

    mock_obj_driver._plot_blind_spot.assert_called_once_with(blind_spot_args, save_path)


def test_object_detection_overall_metrics_plot(common_plots: CommonPlots, mock_obj_driver: MagicMock) -> None:
    """Test the _object_detection_overall_metrics method calls the obj_driver's _plot_overall_metrics correctly."""
    overall_metrics_args: List[str] = ["metric1", "metric2"]
    save_path: str = "./plots/object_detection_overall_metrics.png"

    common_plots._object_detection_overall_metrics(overall_metrics_args, save_path)

    mock_obj_driver._plot_overall_metrics.assert_called_once_with(overall_metrics_args, save_path)


def test_top_misses_plot(common_plots: CommonPlots, mock_obj_driver: MagicMock) -> None:
    """Test the _top_misses method calls the obj_driver's _plot_top_misses correctly."""
    top_misses_args: List[str] = ["miss1", "miss2"]
    save_path: str = "./plots/top_misses.png"

    common_plots._top_misses(top_misses_args, save_path)

    mock_obj_driver._plot_top_misses.assert_called_once_with(top_misses_args, save_path)


def test_classbased_table_metrics_plot(common_plots: CommonPlots, mock_obj_driver: MagicMock) -> None:
    """Test the _classbased_table_metrics method calls the obj_driver's _plot_classbased_table_metrics correctly."""
    classbased_table_args: List[str] = ["table_arg1", "table_arg2"]
    save_path: str = "./plots/classbased_table_metrics.png"

    common_plots._classbased_table_metrics(classbased_table_args, save_path)

    mock_obj_driver._plot_classbased_table_metrics.assert_called_once_with(classbased_table_args, save_path)


def test_mixed_metrics_plot(common_plots: CommonPlots, mock_obj_driver: MagicMock) -> None:
    """Test the _mixed_metrics method calls the obj_driver's _plot_mixed_metrics correctly."""
    mixed_metrics_args: List[str] = ["mixed_arg1", "mixed_arg2"]
    save_path: str = "./plots/mixed_metrics.png"

    common_plots._mixed_metrics(mixed_metrics_args, save_path)

    mock_obj_driver._plot_mixed_metrics.assert_called_once_with(mixed_metrics_args, save_path)