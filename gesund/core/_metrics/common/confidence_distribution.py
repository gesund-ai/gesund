from typing import Union, Optional
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager
from .iou import IoUCalc


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
        check_keys = ("ground_truth", "prediction")
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

    def _preprocess(self, data: dict) -> tuple:
        """
        A function to preprocess

        :param data: dictionary data
        :type data: dict

        :return: gt, pred
        :rtype: tuple(dict, dict)
        """
        from .average_precision import ObjectDetection

        return ObjectDetection._preprocess(data, get_label=False, get_pred_scores=True)

    def _calculate_conf_dist(
        self, gt_boxes_dict, pred_boxes_dict: dict
    ) -> pd.DataFrame:
        """
        A function to organize the data for plotting the confidence distribution

        :param gt_boxes_dict: a dictionary containing the ground truth boxes
        :type gt_boxes_dict: dict
        :param pred_boxes_dict: a dictionary containing the prediction boxes
        :type pred_boxes_dict: dict

        :return: a pandas dataframe containing the organized data
        :rtype: pd.DataFrame

        """
        results = []
        for image_id, pred_boxes in pred_boxes_dict.items():
            gt_boxes = gt_boxes_dict[image_id]
            matched_gt = [False] * len(gt_boxes)

            for pred_box in pred_boxes:
                score = pred_box[-1]
                pred_box = pred_box[:-1]
                best_iou = 0
                best_gt_idx = -1

                for idx, gt_box in enumerate(gt_boxes):
                    iou = self.iou.calculate(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > 0.5:
                    if not matched_gt[best_gt_idx]:
                        results.append(
                            {"confidence": score, "label": "TP", "best_iou": best_iou}
                        )
                        matched_gt[best_gt_idx] = True
                    else:
                        results.append(
                            {"confidence": score, "label": "FP", "best_iou": best_iou}
                        )
                else:
                    results.append(
                        {"confidence": score, "label": "FP", "best_iou": best_iou}
                    )

        data = pd.DataFrame(results)
        return data

    def _calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        """
        A function to calculate the metrics for object detection

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict
        :param class_mapping: a dictionary containing the class mapping
        :type class_mapping: dict

        :return: a dictionary containing the calculated metrics
        :rtype: dict
        """
        results = {}

        # preprocess the data
        gt_boxes_dict, pred_boxes_dict = self._preprocess(data)

        # re-organize the confidence distribution
        result = self._calculate_conf_dist(gt_boxes_dict, pred_boxes_dict)
        results["confidence_distribution"] = result

        return results

    def calculate(self, data: dict) -> dict:
        """
        A function to calculate the metrics for object detection

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict

        :return: a dictionary containing the calculated metrics
        :rtype: dict
        """
        result = {}

        # validate the data
        self._validate_data(data)

        # calculate the metrics
        result = self._calculate_metrics(data, data.get("class_mapping"))
        return result


class PlotConfidenceDistribution:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        validates the data required for plotting the bar plot
        """
        if "confidence_distribution" not in self.data:
            raise ValueError("confidence_distribution data is missing.")

        if not isinstance(self.data["confidence_distribution"], pd.DataFrame):
            raise ValueError(f"Data must be a data frame.")

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

        plot_data = self.data["confidence_distribution"]

        g = sns.JointGrid(
            data=plot_data,
            x="confidence",
            y="best_iou",
            hue="label",
            space=0,
            height=8,
            ratio=7,
        )

        g.plot_joint(sns.scatterplot, palette="pastel")
        g.plot_marginals(sns.histplot, kde=True, color=".5")

        g.set_axis_labels("Confidence Score", "Best IoU", fontsize=14)
        g.figure.suptitle(
            "Scatterplot with confidence distribution histograms", fontsize=16
        )
        plt.subplots_adjust(top=0.95)

        return g.figure


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("object_detection.confidence_distribution")
def calculate_confidence_distribution(data: dict, problem_type: str):
    """
    A wrapper function to calculate the confidence_distribution metrics.

    :param data: a dictionary containing the ground truth and prediction data
    :type data: dict
    :param problem_type: the type of problem
    :type problem_type: str

    :return: a dictionary containing the calculated metrics
    :rtype: dict
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

    :param results: a dictionary containing the calculated metrics
    :type results: dict
    :param save_plot: a flag to save the plot
    :type save_plot: bool
    :param file_name: the name of the file to save the plot
    :type file_name: str
    :param cohort_id: the cohort id
    :type cohort_id: int

    :return: the file path of the saved plot
    :rtype: str
    """
    plotter = PlotConfidenceDistribution(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
