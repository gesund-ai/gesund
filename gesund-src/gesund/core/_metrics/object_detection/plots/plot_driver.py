import os
import warnings
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Optional, Union, Any

from .average_precision import PlotAveragePrecision
from .top_losses import PlotTopLosses
from .object_stats import PlotObjectStats
from .confidence_distribution import PlotConfidenceGraphs
from .dataset_stats import PlotDatasetStats


class ObjectDetectionPlotDriver:
    def __init__(
        self,
        coco_,
        class_mappings,
        ground_truth_dict,
        prediction_dict,
        batch_job_id,
        meta_data_dict=None,
        loss_dict=None,
        filtering_meta=None,
    ):
        # Create validation variables
        self.true = ground_truth_dict
        self.pred = prediction_dict
        self.class_mappings = class_mappings
        self.meta = None
        self.batch_job_id = batch_job_id
        self.coco_ = coco_
        self.filtering_meta = filtering_meta
        if meta_data_dict:
            self.meta = pd.DataFrame(meta_data_dict).T
        if loss_dict:
            self.loss = pd.DataFrame(loss_dict, index=[0])

        self.sample_size = len(self.true)
        self.class_order = list(range(len(class_mappings.keys())))
        # Check, add breakpoint

        # Import Classes
        self.plot_average_precision = PlotAveragePrecision(
            class_mappings=self.class_mappings,
            meta_data_dict=meta_data_dict,
            coco_=coco_,
        )

        self.plot_object_stats = PlotObjectStats(
            coco_=coco_,
            class_mappings=self.class_mappings,
            meta_data_dict=meta_data_dict,
        )

        self.plot_confidence_graphs = PlotConfidenceGraphs(
            class_mappings=self.class_mappings,
            meta_data_dict=meta_data_dict,
            coco_=coco_,
        )

        self.plot_dataset_stats = PlotDatasetStats(
            class_mappings=self.class_mappings,
            meta_data_dict=meta_data_dict,
        )

        self.plot_loss = PlotTopLosses(
            coco_=coco_,
            class_mappings=self.class_mappings,
            meta_dict=self.meta,
        )

    def plot_highlighted_overall_metrics(self) -> Dict[str, Any]:
        """
        Plot highlighted overall metrics at IoU threshold 0.1.

        :return: Dictionary containing overall metrics data
        :rtype: Dict[str, Any]
        """
        return self.plot_average_precision._plot_highlighted_overall_metrics(
            threshold=0.1
        )

    def plot_performance_by_iou_threshold(
        self, threshold: float = 0.5, return_points: bool = False
    ) -> Dict[str, Any]:
        """
        Plot performance metrics at specified IoU threshold.

        :param threshold: IoU threshold value
        :type threshold: float
        :param return_points: Whether to return coordinate points
        :type return_points: bool
        :return: Dictionary containing performance metrics
        :rtype: Dict[str, Any]
        """
        return self.plot_average_precision._plot_performance_by_iou_threshold(
            threshold, return_points
        )

    def plot_statistics_classbased_table(
        self, target_attribute_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Plot statistics table based on classes.

        :param target_attribute_dict: Optional dictionary for filtering attributes
        :type target_attribute_dict: Optional[Dict[str, Any]]
        :return: Dictionary containing class-based statistics
        :rtype: Dict[str, Any]
        """
        return self.plot_average_precision._plot_statistics_classbased_table(
            threshold=0.1, target_attribute_dict=self.filtering_meta
        )

    def plot_object_counts(
        self,
        confidence: float = 0,
        target_attribute_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Plot object count distributions.

        :param confidence: Confidence threshold for filtering
        :type confidence: float
        :param target_attribute_dict: Optional dictionary for filtering attributes
        :type target_attribute_dict: Optional[Dict[str, Any]]
        :return: Dictionary containing object count data
        :rtype: Dict[str, Any]
        """
        return self.plot_object_stats._plot_object_counts(
            confidence=confidence, target_attribute_dict=self.filtering_meta
        )

    def plot_top_misses(self, top_k: int = 100) -> Dict[str, Any]:
        """
        Plot top misclassified samples.

        :param top_k: Number of top misses to plot
        :type top_k: int
        :return: Dictionary containing top misses data
        :rtype: Dict[str, Any]
        """
        return self.plot_loss._plot_top_misses(top_k=top_k)

    def plot_confidence_histogram_scatter_distribution(
        self, predicted_class: Optional[str] = None, n_samples: int = 300
    ) -> Dict[str, Any]:
        """
        Plot confidence distribution as histogram and scatter plot.

        :param predicted_class: Class name to filter results
        :type predicted_class: Optional[str]
        :param n_samples: Number of samples to plot
        :type n_samples: int
        :return: Dictionary containing distribution plots data
        :rtype: Dict[str, Any]
        """
        return (
            self.plot_confidence_graphs._plot_confidence_histogram_scatter_distribution(
                predicted_class, n_samples
            )
        )

    def plot_prediction_distribution(
        self, target_attribute_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Plot distribution of predictions.

        :param target_attribute_dict: Optional dictionary for filtering attributes
        :type target_attribute_dict: Optional[Dict[str, Any]]
        :return: Dictionary containing prediction distribution data
        :rtype: Dict[str, Any]
        """
        return self.plot_object_stats._plot_prediction_distribution(
            target_attribute_dict=self.filtering_meta
        )

    def plot_meta_distribution(self) -> Dict[str, Any]:
        """
        Plot distribution of metadata attributes.

        :return: Dictionary containing metadata distribution plots
        :rtype: Dict[str, Any]
        """
        return self.plot_dataset_stats._plot_meta_distributions()

    def plot_training_validation_comparison_classbased_table(self) -> Dict[str, Any]:
        """
        Plot comparison table between training and validation results.

        :return: Dictionary containing comparison table data
        :rtype: Dict[str, Any]
        """
        return (
            self.plot_average_precision._plot_training_validation_comparison_classbased_table()
        )

    def main_metric(self) -> Dict[str, float]:
        """
        Calculate main evaluation metric.

        :return: Dictionary containing main metric value
        :rtype: Dict[str, float]
        """
        return self.plot_average_precision._main_metric(threshold=0.1)

    # Blind Spots
    def plot_blind_spot_metrics(
        self, target_attribute_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Plot metrics for model blind spots.

        :param target_attribute_dict: Optional dictionary for filtering attributes
        :type target_attribute_dict: Optional[Dict[str, Any]]
        :return: Dictionary containing blind spot metrics
        :rtype: Dict[str, Any]
        """
        return self.plot_average_precision.blind_spot_metrics(
            target_attribute_dict=self.filtering_meta, threshold=0.1
        )

    def _calling_all_plots(self) -> Dict[str, Any]:
        """
        Execute all plotting methods in the class.

        :return: Dictionary containing results from all plotting methods
        :rtype: Dict[str, Any]
        """
        # Getting all methods that do not start with '_'
        methods = [
            method_name
            for method_name in dir(self)
            if callable(getattr(self, method_name)) and not method_name.startswith("_")
        ]
        results = {}

        for method_name in tqdm(methods, desc="Calling all plot functions"):
            method = getattr(self, method_name)
            try:
                # Suppress warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    print(f"Calling method: {method_name}...")
                    # Attempt to call the method, handle cases where no arguments are required
                    result = method()
                    results[method_name] = result
            except TypeError as e:
                results[method_name] = f"Could not call {method_name}: {str(e)}"
            except Exception as e:
                results[method_name] = f"Error in {method_name}: {str(e)}"

        return results
