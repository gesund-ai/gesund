from typing import Union, Optional
import os

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt

from gesund.core import metric_manager, plot_manager
from .iou import IoUCalc


class AveragePrecision:
    pass


class Classification:
    pass


class SemanticSegmentation:
    pass


class ObjectDetection:
    def __init__(self):
        self.iou = IoUCalc()

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

    @staticmethod
    def _preprocess(data: dict, get_label=False, get_pred_scores=False) -> tuple:
        """
        A function to preprocess

        :param data: dictionary data
        :type data: dict
        :param get_label: boolean value to get the label
        :type get_label: bool
        :param get_pred_scores: boolean value to get the prediction scores
        :type get_pred_scores: bool

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

    def _calc_mAP_mAR(
        self, gt_boxes_dict: dict, pred_boxes_dict: dict, thresholds: list
    ) -> dict:
        """
        A function to calculate average precision and average recall

        :param gt_boxes_dict: image id and box points for ground truth
        :type gt_boxes_dict: dict
        :param pred_boxes_dict: image id and box points for prediction
        :type pred_boxes_dict: dict
        :param thresholds: list of threshold
        :type thresholds: list

        :return: dicitonary of results
        :rtype: dict
        """
        results = {"metric": [], "threshold": [], "value": []}
        for threshold in thresholds:
            image_precisions, image_recalls = [], []

            for image_id in gt_boxes_dict.keys():
                gt_boxes = gt_boxes_dict[image_id]
                pred_boxes = pred_boxes_dict[image_id]
                precision, recall = self._calc_precision_recall(
                    gt_boxes, pred_boxes, threshold
                )
                image_precisions.append(precision)
                image_recalls.append(recall)

            average_precision = np.mean(image_precisions)
            average_recall = np.mean(image_recalls)
            results["metric"].append("mAP")
            results["value"].append(average_precision)
            results["metric"].append("mAR")
            results["value"].append(average_recall)
            results["threshold"].append(threshold)
            results["threshold"].append(threshold)

        return pd.DataFrame(results)

    def __calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        """
        A function to  calculate the metrics

        :param data: data dicationry containing the data
        :type data: dict
        :param class_mapping: a dictionary with class mapping labels
        :type class_mapping: dict

        :return: results calculated
        :rtype: dict
        """
        results = {}

        # preprocess the data to convert the dictionasry into gt box
        # pred box
        gt_boxes, pred_boxes = self._preprocess(data)

        # get the threshold
        thresholds = data["metric_args"]["threshold"]

        if not isinstance(thresholds, list) and thresholds is not None:
            thresholds = [thresholds]

        # calculate the mAR
        results = self._calc_mAP_mAR(gt_boxes, pred_boxes, thresholds)

        return results

    def calculate(self, data: dict) -> dict:
        """
        Calculates the average precision score for the given dataset

        :param data: The input data required for calculation and plotting
                    {"prediction":, "ground_truth": , "class_mapping":}
        :type data: dict

        :return: calculated metric results
        :rtype: dict
        """
        result = {}

        # validate the data
        self._validate_data(data)

        # calculate results
        result = self.__calculate_metrics(data, data.get("class_mapping"))

        return {"result": result}


class PlotAvgPrecision:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        validates the data required for plotting the bar plot
        """
        if not isinstance(self.data["result"], pd.DataFrame):
            raise ValueError(f"Data must be a data frame.")

    def save(self, fig: Figure, filename: str) -> str:
        """
        Saves the plot to a file

        :param fig: the matplotlib figure object where the plot image is stored
        :type fig: str

        :return: Path where the plot image is saved
        :rtype: str
        """
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
        Plots the AUC curves.
        """
        sns.set_style("whitegrid")

        # validate the data
        self._validate_data()

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(
            x="metric",
            y="value",
            hue="threshold",
            data=self.data["result"],
            ax=ax,
            palette="pastel",
        )
        ax.set_xlabel("Metric", fontdict={"fontsize": 14, "fontweight": "medium"})
        ax.set_ylabel("Value", fontdict={"fontsize": 14, "fontweight": "medium"})

        if self.cohort_id:
            title_str = f"mAP and mAR at Thresholds: cohort - {self.cohort_id}"
        else:
            title_str = "mAP and mAR at Thresholds"

        ax.set_title(
            title_str,
            fontdict={"fontsize": 16, "fontweight": "medium"},
        )
        ax.legend(loc="lower right")
        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("object_detection.average_precision")
def calculate_avg_precision(data: dict, problem_type: str):
    """
    A wrapper function to calculate the average precision
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result


@plot_manager.register("object_detection.average_precision")
def plot_avg_precision(
    results: dict,
    save_plot: bool,
    file_name: str = "average_precision.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to plot average precision

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: boolean value to save plot
    :type save_plot: bool

    :return: None or path to the saved plot
    :rtype: Union[str, None]
    """
    plotter = PlotAvgPrecision(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
