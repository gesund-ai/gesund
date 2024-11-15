import os
import warnings
from tqdm import tqdm
import pandas as pd

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

    def plot_highlighted_overall_metrics(self):
        return self.plot_average_precision._plot_highlighted_overall_metrics(
            threshold=0.1
        )

    def plot_performance_by_iou_threshold(self, threshold=0.5, return_points=False):
        return self.plot_average_precision._plot_performance_by_iou_threshold(
            threshold, return_points
        )

    def plot_statistics_classbased_table(self, target_attribute_dict=None):
        return self.plot_average_precision._plot_statistics_classbased_table(
            threshold=0.1, target_attribute_dict=self.filtering_meta
        )

    def plot_object_counts(self, confidence=0, target_attribute_dict=None):
        return self.plot_object_stats._plot_object_counts(
            confidence=confidence, target_attribute_dict=self.filtering_meta
        )

    def plot_top_misses(self, top_k=100):
        return self.plot_loss._plot_top_misses(top_k=top_k)

    def plot_confidence_histogram_scatter_distribution(
        self, predicted_class=None, n_samples=300
    ):
        return (
            self.plot_confidence_graphs._plot_confidence_histogram_scatter_distribution(
                predicted_class, n_samples
            )
        )

    def plot_prediction_distribution(self, target_attribute_dict=None):
        return self.plot_object_stats._plot_prediction_distribution(
            target_attribute_dict=self.filtering_meta
        )

    def plot_meta_distribution(
        self,
    ):
        return self.plot_dataset_stats._plot_meta_distributions()

    def plot_training_validation_comparison_classbased_table(self):
        return (
            self.plot_average_precision._plot_training_validation_comparison_classbased_table()
        )

    def main_metric(self):
        return self.plot_average_precision._main_metric(threshold=0.1)

    # Blind Spots
    def plot_blind_spot_metrics(self, target_attribute_dict=None):
        return self.plot_average_precision.blind_spot_metrics(
            target_attribute_dict=self.filtering_meta, threshold=0.1
        )

    def _calling_all_plots(self):
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
