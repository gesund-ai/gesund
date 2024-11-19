import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

from ..metrics.average_precision import AveragePrecision
from gesund.core._utils import ValidationUtils


class PlotAveragePrecision:
    def __init__(
        self,
        class_mappings,
        coco_,
        meta_data_dict=None,
    ):
        self.class_mappings = class_mappings
        self.coco_ = coco_

        if bool(meta_data_dict):
            self.is_meta_exists = True
            meta_df = pd.DataFrame(meta_data_dict).T
            self.validation_utils = ValidationUtils(meta_df)

    def _plot_performance_by_iou_threshold(
        self, threshold: float, return_points: bool = False
    ) -> Dict[str, Any]:
        """
        Plot performance metrics at specific IoU threshold.

        :param threshold: IoU threshold value
        :type threshold: float
        :param return_points: Whether to return coordinate points
        :type return_points: bool
        :return: Dictionary containing performance metrics data
        :rtype: Dict[str, Any]
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

    def _plot_highlighted_overall_metrics(self, threshold: float) -> Dict[str, Any]:
        """
        Plot highlighted overall metrics at specified threshold.

        :param threshold: IoU threshold value
        :type threshold: float
        :return: Dictionary containing overall metrics data
        :rtype: Dict[str, Any]
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

    def _filter_ap_metrics(
        self, target_attribute_dict: Optional[Dict[str, Any]]
    ) -> List[int]:
        """
        Filter average precision metrics based on target attributes.

        :param target_attribute_dict: Dictionary of target attributes for filtering
        :type target_attribute_dict: Optional[Dict[str, Any]]
        :return: List of filtered indices
        :rtype: List[int]
        """
        if target_attribute_dict:
            idxs = self.validation_utils.filter_attribute_by_dict(
                target_attribute_dict
            ).index.tolist()

        return idxs

    def _plot_statistics_classbased_table(
        self,
        threshold: Optional[float] = None,
        target_attribute_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Plot class-based statistics table.

        :param threshold: IoU threshold value
        :type threshold: Optional[float]
        :param target_attribute_dict: Dictionary for filtering by attributes
        :type target_attribute_dict: Optional[Dict[str, Any]]
        :return: Dictionary containing class-based statistics
        :rtype: Dict[str, Any]
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

        threshold = 0.1
        payload_dict = self._plot_highlighted_overall_metrics(threshold)

        keys_to_be_included = ["mAP@50", "mAP@75", "mAP@[.50,.95]"]
        all_keys = payload_dict["data"].keys()

        for key in list(set(all_keys) - set(keys_to_be_included)):
            payload_dict["data"].pop(key)

        payload_dict["type"] = "bar"
        return payload_dict

    def _main_metric(self, threshold):
        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )
        mean_map_given = average_precision.calculate_ap_metrics(threshold=threshold)[
            "mAP"
        ].round(4)
        payload_dict = {f"mAP@{int(threshold*100)}": mean_map_given}
        return payload_dict

    def blind_spot_metrics(
        self, target_attribute_dict: Optional[Dict[str, Any]], threshold: float
    ) -> Dict[str, Any]:
        """
        Plots ROC Curve for target_class.
        References:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        https://plotly.com/python/roc-and-pr-curves/
        :param target_class: target class to produce ROC plot
        :return: payload_dict
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
        overall_metrics = average_precision.calculate_highlighted_overall_metrics(
            threshold
        )

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
