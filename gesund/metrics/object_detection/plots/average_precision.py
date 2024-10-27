import pandas as pd
import numpy as np

from ..metrics.average_precision import AveragePrecision
from gesund.utils.validation_data_utils import ValidationUtils

class PlotAveragePrecision:
    def __init__(
        self,
        class_mappings,
        coco_,
        meta_data_dict=None,
    ):
        """
        Initialize the PlotAveragePrecision class with class mappings, COCO data, and optional metadata.

        :param class_mappings: Dictionary mapping class IDs to class names.
        :param coco_: COCO dataset object containing annotations and predictions.
        :param meta_data_dict: (optional) Dictionary containing metadata information.
        """
        self.class_mappings = class_mappings
        self.coco_ = coco_

        if bool(meta_data_dict):
            self.is_meta_exists = True
            meta_df = pd.DataFrame(meta_data_dict).T
            self.validation_utils = ValidationUtils(meta_df)

    def _plot_performance_by_iou_threshold(self, threshold, return_points=False):
        """
        Plot performance metrics based on the specified IoU threshold.

        This function calculates the average precision metrics and IoU threshold graph for the provided
        threshold and returns the results in a structured format.

        :param threshold: Float representing the Intersection over Union (IoU) threshold.
        :param return_points: Boolean indicating whether to return detailed metric points.
        
        :return: Dictionary containing metrics or overall metrics based on the return_points flag.
        """
        payload_dict = dict()
        payload_dict["type"] = "mixed"

        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )

        metrics = average_precision.calculate_ap_metrics(threshold=threshold)
        coordinates = average_precision.calculate_iou_threshold_graph(
            threshold=threshold
        )

        response = average_precision.calculate_highlighted_overall_metrics(threshold)

        if return_points:
            payload_dict["data"] = {
                "metrics": metrics,
                "coordinates": coordinates,
            }
        else:
            payload_dict["data"] = {"ap_results": response}
        return payload_dict

    def _plot_highlighted_overall_metrics(self, threshold):
        """
        Plot highlighted overall metrics at the specified IoU threshold.

        This function renames the metrics for better readability and returns the overall validation
        metrics in a structured format.

        :param threshold: Float representing the Intersection over Union (IoU) threshold.
        
        :return: Dictionary containing renamed overall metrics.
        """
        rename_dict = {
            f"map{int(threshold*100)}": f"mAP@{int(threshold*100)}",
            f"map{int(threshold*100)}_11": f"mAP11@{int(threshold*100)}",
            "map50": "mAP@50",
            "map75": "mAP@75",
            "map5095": "mAP@[.50,.95]",
            "mar1": "mAR@max=1",
            "mar10": "mAR@max=10",
            "mar100": "mAR@max=100",
        }
        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )
        overall_metrics = average_precision.calculate_highlighted_overall_metrics(
            threshold
        )
        for metric in list(overall_metrics.keys()):
            overall_metrics[rename_dict[metric]] = overall_metrics.pop(metric)

        val_train_dict = {}
        for value in rename_dict.values():
            val_train_dict[value] = {"Validation": overall_metrics[value]}

        payload_dict = {"type": "overall", "data": val_train_dict}
        return payload_dict

    def _filter_ap_metrics(self, target_attribute_dict):
        """
        Filter average precision metrics based on the target attribute dictionary.

        This function extracts the indices of the metrics to be filtered based on the provided
        target attributes.

        :param target_attribute_dict: Dictionary of target attributes used for filtering.
        
        :return: List of indices corresponding to the filtered metrics.
        """
        if target_attribute_dict:
            idxs = self.validation_utils.filter_attribute_by_dict(
                target_attribute_dict
            ).index.tolist()

        return idxs

    def _plot_statistics_classbased_table(
        self, threshold=None, target_attribute_dict=None
    ):
        """
        Plot a statistics table for average precision metrics based on class.

        This function calculates the class-wise average precision metrics and returns them in
        a structured format.

        :param threshold: (optional) Float representing the IoU threshold for metrics calculation.
        :param target_attribute_dict: (optional) Dictionary of target attributes for filtering metrics.
        
        :return: Dictionary containing average precision metrics by class.
        """
        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )

        rename_dict = {
            f"ap10": f"AP@10",
            f"ap10_11": f"AP11@10",
            "ap50": "AP@50",
            "ap75": "AP@75",
            "ap5095": "AP@[.50,.95]",
            "ar1": "AR@max=1",
            "ar10": "AR@max=10",
            "ar100": "AR@max=100",
        }
        if target_attribute_dict:
            idxs = self._filter_ap_metrics(target_attribute_dict)
        else:
            idxs = None

        ap_metrics = average_precision.calculate_ap_metrics(idxs=idxs)

        class_ap_metrics = dict()

        for class_ in ap_metrics["ap50"].keys():
            class_dict = dict()
            for metric in ap_metrics:
                class_dict[rename_dict[metric]] = ap_metrics[metric][class_]
            class_ap_metrics[self.class_mappings[str(class_)]] = class_dict

        payload_dict = {"type": "table", "data": {"Validation": class_ap_metrics}}
        return payload_dict

    def _plot_training_validation_comparison_classbased_table(self):
        """
        Plot a comparison table of training and validation average precision metrics by class.

        This function retrieves the overall metrics and filters them to include only relevant
        metrics for comparison between training and validation.

        :return: Dictionary containing filtered training and validation comparison metrics.
        """
        threshold = 0.1
        payload_dict = self._plot_highlighted_overall_metrics(threshold)

        keys_to_be_included = ["mAP@50", "mAP@75", "mAP@[.50,.95]"]
        all_keys = payload_dict["data"].keys()

        for key in list(set(all_keys) - set(keys_to_be_included)):
            payload_dict["data"].pop(key)

        payload_dict["type"] = "bar"
        return payload_dict

    def _main_metric(self, threshold):
        """
        Calculate and return the main average precision metric at the specified threshold.

        This function computes the mean average precision given a threshold and returns it in
        a structured format.

        :param threshold: Float representing the Intersection over Union (IoU) threshold.
        
        :return: Dictionary containing the mean average precision metric.
        """
        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )
        mean_map_given = average_precision.calculate_ap_metrics(threshold=threshold)[
            "mAP"
        ].round(4)
        payload_dict = {f"mAP@{int(threshold*100)}": mean_map_given}
        return payload_dict

    def blind_spot_metrics(self, target_attribute_dict, threshold):
        """
        Calculate and return blind spot metrics for the specified target attributes.

        This function computes average precision metrics and overall metrics for the specified
        target attributes, returning them in a structured format.

        :param target_attribute_dict: Dictionary of target attributes for filtering metrics.
        :param threshold: Float representing the Intersection over Union (IoU) threshold.
        
        :return: Dictionary containing average precision metrics for each class and overall metrics.
        """
        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )
        avg_rename_dict = {
            f"map{int(threshold*100)}": f"mAP@{int(threshold*100)}",
            f"map{int(threshold*100)}_11": f"mAP11@{int(threshold*100)}",
            "map50": "mAP@50",
            "map75": "mAP@75",
            "map5095": "mAP@[.50,.95]",
            "mar1": "mAR@max=1",
            "mar10": "mAR@max=10",
            "mar100": "mAR@max=100",
        }

        rename_dict = {
            "ap10": "AP@10",
            "ap10_11": "AP11@10",
            "ap50": "AP@50",
            "ap75": "AP@75",
            "ap5095": "AP@[.50,.95]",
            "ar1": "AR@max=1",
            "ar10": "AR@max=10",
            "ar100": "AR@max=100",
        }
        idxs = (
            self._filter_ap_metrics(target_attribute_dict)
            if bool(target_attribute_dict)
            else None
        )

        class_metrics = average_precision.calculate_ap_metrics(idxs=idxs)
        overall_metrics = average_precision.calculate_highlighted_overall_metrics(threshold)

        for key in list(class_metrics.keys()):
            class_metrics[rename_dict[key]] = class_metrics.pop(key)

        for metric in list(overall_metrics.keys()):
            overall_metrics[avg_rename_dict[metric]] = overall_metrics.pop(metric)

        blind_spot_metrics_dict = pd.DataFrame(class_metrics).T.to_dict()
        blind_spot_metrics_dict = {
            str(k): v for k, v in blind_spot_metrics_dict.items()
        }

        blind_spot_metrics_dict["Average"] = overall_metrics
        
        return blind_spot_metrics_dict
