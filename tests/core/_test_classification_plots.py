"""
Test suite for ClassificationPlots in the gesund.core._plot module.
"""

import pytest
from unittest.mock import Mock, patch
from gesund.core._plot import ClassificationPlots

@pytest.fixture
def classification_plots():
    return ClassificationPlots()

@pytest.fixture
def mock_metrics():
    return {
        "class_metrics": {
            "class1": {"accuracy": 0.95},
            "class2": {"accuracy": 0.85}
        }
    }

@pytest.fixture
def mock_class_types():
    return ["class1", "class2"]

@pytest.fixture
def mock_save_path():
    return "test/path/plots"

@patch('gesund.core._metrics.classification.classification_metric_plot.Classification_Plot')
def test_classification_blind_spot_success(mock_classification_plot, classification_plots, mock_metrics, mock_class_types, mock_save_path):
    mock_plot_instance = Mock()
    mock_classification_plot.return_value = mock_plot_instance
    classification_plots.cls_driver = mock_plot_instance

    classification_plots._classification_blind_spot(
        metrics=mock_metrics,
        class_types=mock_class_types,
        save_path=mock_save_path
    )

    mock_plot_instance._plot_blind_spot.assert_called_once_with(
        mock_metrics,
        mock_class_types,
        mock_save_path
    )

@patch('gesund.core._metrics.classification.classification_metric_plot.Classification_Plot')
def test_classification_blind_spot_with_none_save_path(mock_classification_plot, classification_plots, mock_metrics, mock_class_types):
    mock_plot_instance = Mock()
    mock_classification_plot.return_value = mock_plot_instance
    classification_plots.cls_driver = mock_plot_instance

    classification_plots._classification_blind_spot(
        metrics=mock_metrics,
        class_types=mock_class_types,
        save_path=None
    )

    mock_plot_instance._plot_blind_spot.assert_called_once_with(
        mock_metrics,
        mock_class_types,
        None
    )