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
from gesund.core._utils import ValidationUtils
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
                    pred_boxes[image_id].append(
                        {"box_points": box_points, "category_id": pred["category_id"]}
                    )
                else:
                    pred_boxes[image_id] = [
                        {"box_points": box_points, "category_id": pred["category_id"]}
                    ]

        return (gt_boxes, pred_boxes)

    def __calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        gt_boxes, pred_boxes = self.__preprocess(data)

        pred_class_occurrence_dict = {}
        for image_id in pred_boxes:
            for pred in pred_boxes[image_id]:
                class_id = pred["category_id"]
                if class_id in pred_class_occurrence_dict:
                    pred_class_occurrence_dict[class_id] += 1
                else:
                    pred_class_occurrence_dict[class_id] = 1

        if class_mapping is not None:
            pred_class_occurrence_dict = {
                class_mapping[str(k)]: v for k, v in pred_class_occurrence_dict.items()
            }

        payload_dict = {
            "type": "pie",
            "data": pred_class_occurrence_dict,
        }

        return payload_dict

    def calculate(self, data: dict) -> dict:
        self._validate_data(data)
        result = self.__calculate_metrics(data, data.get("class_mapping"))
        return result


class PlotPredictedDistribution:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
        required_keys = ["type", "data"]
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
        sns.set_theme(style="whitegrid")
        self._validate_data()

        pred_class_occurrence_dict = self.data["data"]

        fig, ax = plt.subplots(figsize=(10, 7))
        labels = list(pred_class_occurrence_dict.keys())
        values = list(pred_class_occurrence_dict.values())
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        ax.set_title("Predicted Distribution")
        plt.tight_layout()

        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("object_detection.predicted_distribution")
def calculate_predicted_distribution(data: dict, problem_type: str):
    """
    A wrapper function to calculate the predicted_distribution metrics.
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result


@plot_manager.register("object_detection.predicted_distribution")
def plot_predicted_distribution_od(
    results: dict,
    save_plot: bool,
    file_name: str = "predicted_distribution.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to plot the predicted distribution metrics.
    """
    plotter = PlotPredictedDistribution(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
