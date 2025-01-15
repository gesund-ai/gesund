from typing import Union, Optional
import os

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt

from gesund.core import metric_manager, plot_manager
from .iou import IoUCalc

class Classification:
    pass

class SemanticSegmentation:
    pass

class AveragePrecision:
    pass

class ObjectDetection:
    def __init__(self):
        self.iou = IoUCalc()

    def _validate_data(self, data: dict) -> bool:
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
        

    @staticmethod
    def _preprocess(data: dict, get_label=False, get_pred_scores=False) -> tuple:
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

                if get_label:
                    box_points.append(_ant["label"])

                if image_id in gt_boxes:
                    gt_boxes[image_id].append(box_points)
                else:
                    gt_boxes[image_id] = [box_points]

            for pred in data["prediction"][image_id]["objects"]:
                points = pred["box"]
                box_points = [points["x1"], points["y1"], points["x2"], points["y2"]]

                if get_label:
                    box_points.append(pred["prediction_class"])

                if get_pred_scores:
                    box_points.append(pred["confidence"])

                if image_id in pred_boxes:
                    pred_boxes[image_id].append(box_points)
                else:
                    pred_boxes[image_id] = [box_points]

        return (gt_boxes, pred_boxes)


    def _calc_precision_recall(self, gt_boxes, pred_boxes, threshold: float) -> tuple:
        pass

    def _calc_mAP_mAR(
        self, gt_boxes_dict: dict, pred_boxes_dict: dict, thresholds: list
    ) -> dict:
        pass

    def __calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        pass

    def calculate(self, data: dict) -> dict:
        result = {}

        self._validate_data(data)
        result = self.__calculate_metrics(data, data.get("class_mapping"))
        return {"result": result}
    

class PlotModelStats:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        pass

    def _validate_data(self):
        pass

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
        pass

problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}

@metric_manager.register("object_detection.model_stats")
def calculate_model_stats(data: dict, problem_type: str):
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result

@plot_manager.register("object_detection.model_stats")
def plot_model_stats(
    results: dict,
    save_plot: bool,
    file_name: str = "model_stats.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    
    plotter = PlotModelStats(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()