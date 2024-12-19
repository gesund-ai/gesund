from typing import Union, Optional
import os
import itertools
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import sklearn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager
from .average_precision import AveragePrecision
from .iou import IoUCalc


class Classification:
    pass


class SemanticSegmentation:
    pass


class ObjectDetection:
    def __init__(self):
        self.iou = IoUCalc()
        self.class_mapping = {}

    def _validate_data(self, data: dict) -> bool:
        """
        A function to validate the given data used for calculating metrics for object detection validation

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict
        """
        # check for the important keys in the data
        check_keys = ("ground_truth", "prediction", "class_mapping", "metric_args")
        for _key in check_keys:
            if _key not in data:
                raise ValueError(f"Missing {_key} in the data dictionary")

        # check the common set of images
        common_ids = set(list(data["prediction"].keys())).difference(
            set(list(data["ground_truth"].keys()))
        )

        if common_ids:
            raise ValueError(
                "prediction and ground truth does not have corresponding samples"
            )

    def __preprocess(self, data: dict) -> tuple:
        """
        A function to preprocess

        :param data: dictionary data
        :type data: dict

        :return: gt, pred
        :rtype: tuple(dict, dict)
        """
        gt_boxes, pred_boxes = {}, {}

        for image_id in data["ground_truth"]:
            for _ant in data["ground_truth"][image_id]["annotation"]:
                points = _ant["points"]
                box_points = [
                    points[0]["x"],
                    points[0]["y"],
                    points[1]["x"],
                    points[1]["y"],
                ]
                if image_id in gt_boxes:
                    gt_boxes[image_id].append(box_points)
                else:
                    gt_boxes[image_id] = [box_points]

            for pred in data["prediction"][image_id]["objects"]:
                points = pred["box"]
                box_points = [points["x1"], points["y1"], points["x2"], points["y2"]]
                if image_id in pred_boxes:
                    pred_boxes[image_id].append(box_points)
                else:
                    pred_boxes[image_id] = [box_points]

        return (gt_boxes, pred_boxes)

    def _calc_precision_recall(self, gt_boxes, pred_boxes, threshold: float) -> tuple:
        """
        A function to calculate the precision and recall

        :param gt_boxes:
        :type gt_boxes:
        :param pred_boxes:
        :type pred_boxes:
        :param threshold:
        :type threshold:

        :return: calculated precision and recall
        :rtype: tuple
        """
        num_gt_boxes, num_pred_boxes = len(gt_boxes), len(pred_boxes)
        true_positives, false_positives = 0, 0

        for pred_box in pred_boxes:
            max_iou = 0
            for gt_box in gt_boxes:
                iou = self.iou.calculate(pred_box, gt_box)
                max_iou = max(max_iou, iou)

            if max_iou >= threshold:
                true_positives += 1
            else:
                false_positives += 1

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = true_positives / num_gt_boxes

        return (precision, recall)

    def _calc_conf_distR(self, gt_boxes_dict, pred_boxes_dict, threshold):
        # TODO: Could be wrong that function check again.
        image_id_scores = {}
        for image_id in pred_boxes_dict:
            # TODO:
            # get gt&pred boxes

            gt_boxes = (gt_boxes_dict.get(image_id, []),)
            pred_boxes = pred_boxes_dict[image_id]

            for cls_id in self.class_mapping:
                cls_id = int(cls_id)
                gt_boxes_cls = [
                    box for box in gt_boxes if box.get("category_id") == cls_id
                ]
                pred_boxes_cls = [
                    box for box in pred_boxes if box.get("category_id") == cls_id
                ]

                if not gt_boxes_cls or not pred_boxes_cls:
                    continue

                confidences = []
                for pred_box in pred_boxes_cls:
                    pred_bbox = pred_box["bbox"]
                    max_iou = 0
                    for gt_box in gt_boxes_cls:
                        gt_bbox = gt_box["bbox"]
                        iou = self.iou.calculate(np.array(pred_bbox), np.array(gt_bbox))
                        if iou > max_iou:
                            max_iou = iou
                    if max_iou >= threshold:
                        confidences.append(pred_box["score"])
                if confidences:
                    image_id_scores[image_id] = np.mean(confidences)
                    break
        return image_id_scores

    def __calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        results = {}
        gt_boxes, pred_boxes = self.__preprocess(data)

        thresholds = data["metric_args"]["threshold"]

        if not isinstance(thresholds, list) and thresholds is not None:
            thresholds = [thresholds]

        results = self._calc_conf_distR(gt_boxes, pred_boxes, thresholds)
        return results

    def calculate(self, data: dict) -> dict:
        result = {}
        self._validate_data(data)
        result = self.__calculate_metrics(data, data.get("class_mapping"))
        return result


class PlotConfidenceDistribution:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
        # TODO: After check fun -> need to update
        required_keys = ["class_mapping", "ground_truth", "prediction"]
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Data must contain '{key}'.")

    def save(self, fig: Figure, filename: str) -> str:
        dir_path = "plots"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if self.cohort_id:
            filepath = f"{dir_path}/{self.cohort_id}_{filename}"
        else:
            filepath = f"{dir_path}/{filename}"

        fig.savefig(filepath, format="png")
        return filepath

    def plot(self) -> Figure:
        """
        A function to plot the confidence distribution
        """
        sns.set_theme(style="whitegrid")
        self._validate_data()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            # TODO: Check the data after update the function
            data=self.data["result"],
            x="x",
            y="y",
            hue="labels",
            palette="rocket",
            s=100,
            alpha=0.7,
        )
        ax.set_title("Scatter Plot of Points", fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel("X-axis", fontdict={"fontsize": 14, "fontweight": "medium"})
        ax.set_ylabel("Y-axis", fontdict={"fontsize": 14, "fontweight": "medium"})
        ax.legend(loc="lower right", fontsize=12)
        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("object_detection.confidence_distribution")
def calculate_confidence_distribution(data: dict, problem_type: str):
    """
    A wrapper function to calculate the confidence_distribution metrics.
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result


@plot_manager.register("object_detection.confidence_distribution")
def plot_confidence_distribution_od(
    results: dict,
    save_plot: bool,
    file_name: str = "confidence_distribution.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to plot the confidence distribution metrics.
    """
    plotter = PlotConfidenceDistribution(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
