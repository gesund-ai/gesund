import sys
import os
import unittest
from unittest.mock import Mock, patch

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from gesund.core._plot import ClassificationPlots


class TestClassificationBlindSpot(unittest.TestCase):
    def setUp(self):
        self.classification_plots = ClassificationPlots()
        self.mock_metrics = {
            "class_metrics": {
                "class1": {"accuracy": 0.95},
                "class2": {"accuracy": 0.85},
            }
        }
        self.mock_class_types = ["class1", "class2"]
        self.mock_save_path = "test/path/plots"

    @patch(
        "gesund.core._metrics.classification.classification_metric_plot.Classification_Plot"
    )
    def test_classification_blind_spot_success(self, mock_classification_plot):
        mock_plot_instance = Mock()
        mock_classification_plot.return_value = mock_plot_instance
        self.classification_plots.cls_driver = mock_plot_instance

        self.classification_plots._classification_blind_spot(
            metrics=self.mock_metrics,
            class_types=self.mock_class_types,
            save_path=self.mock_save_path,
        )

        mock_plot_instance._plot_blind_spot.assert_called_once_with(
            self.mock_metrics, self.mock_class_types, self.mock_save_path
        )

    @patch(
        "gesund.core._metrics.classification.classification_metric_plot.Classification_Plot"
    )
    def test_classification_blind_spot_with_none_save_path(
        self, mock_classification_plot
    ):
        mock_plot_instance = Mock()
        mock_classification_plot.return_value = mock_plot_instance
        self.classification_plots.cls_driver = mock_plot_instance

        self.classification_plots._classification_blind_spot(
            metrics=self.mock_metrics, class_types=self.mock_class_types, save_path=None
        )

        mock_plot_instance._plot_blind_spot.assert_called_once_with(
            self.mock_metrics, self.mock_class_types, None
        )

    @patch(
        "gesund.core._metrics.classification.classification_metric_plot.Classification_Plot"
    )
    def test_classification_blind_spot_error_handling(self, mock_classification_plot):
        mock_plot_instance = Mock()
        mock_plot_instance._plot_blind_spot.side_effect = Exception("Plot error")
        mock_classification_plot.return_value = mock_plot_instance
        self.classification_plots.cls_driver = mock_plot_instance

        with self.assertRaises(Exception):
            self.classification_plots._classification_blind_spot(
                metrics=self.mock_metrics,
                class_types=self.mock_class_types,
                save_path=self.mock_save_path,
            )


if __name__ == "__main__":
    unittest.main()
