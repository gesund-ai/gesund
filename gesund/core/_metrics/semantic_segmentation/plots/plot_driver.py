import os
import pandas as pd
import warnings
from tqdm import tqdm

from .coco_metrics import PlotCOCOMetrics
from .top_misses import PlotTopMisses
from .iou_distribution import PlotIoUDistribution
from .violin_plot import PlotViolin
from .dataset_stats import PlotDatasetStats


class SemanticSegmentationPlotDriver:
    def __init__(
        self,
        class_mappings,
        ground_truth_dict,
        prediction_dict,
        batch_job_id,
        meta_data_dict=None,
        loss_dict=None,
        study_list=None,
        filtering_meta=None,
    ):

        # Create validation variables
        self.true = ground_truth_dict
        self.pred = prediction_dict
        self.class_mappings = class_mappings
        self.meta = None
        self.batch_job_id = batch_job_id
        self.filtering_meta = filtering_meta
        if meta_data_dict:
            self.meta = pd.DataFrame(meta_data_dict).T

        self.sample_size = len(self.true)
        self.class_order = list(range(len(class_mappings.keys())))

        self.artifacts_path = None  

        # Import Classes
        self.plot_coco_metrics = PlotCOCOMetrics(
            ground_truth_dict=self.true,
            meta=self.meta,
            prediction_dict=self.pred,
            class_mappings=class_mappings,
            artifacts_path=self.artifacts_path,
            study_list=study_list,
        )

        self.plot_top_misses_ = PlotTopMisses(
            ground_truth_dict=self.true,
            prediction_dict=self.pred,
            class_mappings=self.class_mappings,
            artifacts_path=self.artifacts_path,
            study_list=study_list,
        )

        self.plot_iou_distribution_ = PlotIoUDistribution(
            ground_truth_dict=self.true,
            prediction_dict=self.pred,
            class_mappings=self.class_mappings,
            artifacts_path=self.artifacts_path,
            study_list=study_list,
        )

        self.plot_violin_ = PlotViolin(
            ground_truth_dict=self.true,
            prediction_dict=self.pred,
            class_mappings=self.class_mappings,
            artifacts_path=self.artifacts_path,
            study_list=study_list,
        )

        self.plot_dataset_stats = PlotDatasetStats(meta=self.meta)

    def plot_highlighted_overall_metrics(self):
        return self.plot_coco_metrics.highlighted_overall_metrics()

    def plot_main_metric(self):
        return self.plot_coco_metrics.main_metric()

    def plot_statistics_classbased_table(self, target_attribute_dict=None):
        return self.plot_coco_metrics.statistics_classbased_table(
            target_attribute_dict=self.filtering_meta
        )

    def plot_top_misses(self, metric="IoU", sort_by="Ascending", top_k=150):
        return self.plot_top_misses_.top_misses(
            metric=metric, sort_by=sort_by, top_k=top_k
        )

    def plot_metrics_by_meta_data(self, target_attribute_dict=None):
        return self.plot_coco_metrics.metrics_by_meta_data(
            target_attribute_dict=self.filtering_meta
        )

    def plot_iou_distribution(self, n_samples=300):
        return self.plot_iou_distribution_.iou_distribution(n_samples=n_samples)

    def plot_meta_distribution(self):
        return self.plot_dataset_stats.meta_distributions()

    def plot_violin_graph(self):
        return self.plot_violin_.violin_graph()

    def plot_blind_spot_metrics(self, target_attribute_dict=None):
        return self.plot_coco_metrics.blind_spot_metrics(
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