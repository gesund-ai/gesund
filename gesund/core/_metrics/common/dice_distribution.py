from typing import Optional, Union
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager
from .dice import DiceCalc
from .iou import IoUCalc


class Classification:
    pass


class ObjectDetection:
    pass


class SemanticSegmentation:
    def __init__(self):
        self.iou = IoUCalc()
        self.dice = DiceCalc()

    def _validate_data(self, data: dict) -> bool:
        """
        A function to validate the given data used for calculating metrics for object detection validation

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict

        :return: a boolean value
        :rtype: bool
        """
        from .iou_distribution import SemanticSegmentation

        return SemanticSegmentation._validate_data(SemanticSegmentation, data)

    def _preprocess(self, data: dict) -> dict:
        """
        A function to preprocess the data

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict

        :return: a dictionary containing the preprocessed data
        :rtype: dict
        """
        from .iou_distribution import SemanticSegmentation

        return SemanticSegmentation._preprocess(SemanticSegmentation, data)

    def _calculate_metrics(self, data: dict) -> dict:
        """
        A function to calculate the metrics

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict

        :return: a dictionary containing the calculated metrics
        :rtype: dict
        """
        # preprocess the data
        gt_data, pred_data = self._preprocess(data)

        # calculate the dice distribution
        result = {"image_id": [], "dice": [], "iou": []}

        for image_id in gt_data:
            gt_masks = gt_data[image_id]
            if not gt_masks:
                continue

            for gt_mask in gt_masks:
                for pred_mask in pred_data[image_id]:
                    dice = self.dice.calculate(gt_mask, pred_mask)
                    iou = self.iou.calculate_iou_mask(gt_mask, pred_mask)
                    result["image_id"].append(image_id)
                    result["dice"].append(dice)
                    result["iou"].append(iou)

            result = pd.DataFrame(result)
            result = result.dropna()
            return result

    def calculate(self, data: dict) -> dict:
        """
        A function to calculate the dice distribution for semantic segmentation

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict

        :return: a dictionary containing the dice distribution
        :rtype: dict
        """
        result = {}

        # validate the data
        self._validate_data(data)

        # calculate the metrics
        result["dice_distribution"] = self._calculate_metrics(data)

        return result


class PlotDiceDistribution:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        A function to validate the data
        """
        if "dice_distribution" not in self.data:
            raise ValueError("Missing dice_distribution in the data dictionary")

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
        A function to plot the predicted distribution

        :return: the figure object
        :rtype: Figure
        """
        # validate the data
        self._validate_data()

        sns.set_theme(style="whitegrid")
        plot_data = self.data["dice_distribution"]

        g = sns.JointGrid(data=plot_data, x="dice", y="iou", space=0, height=9, ratio=7)

        g.plot_joint(sns.scatterplot, palette="pastel")
        g.plot_marginals(sns.histplot, kde=True, color=".5")

        g.set_axis_labels("DICE", "IoU", fontsize=14)
        g.figure.suptitle("Scatterplot DICE vs IoU", fontsize=16)
        plt.subplots_adjust(top=0.95)
        return g.figure


problem_type_map = {
    "classification": Classification,
    "object_detection": ObjectDetection,
    "semantic_segmentation": SemanticSegmentation,
}


@metric_manager.register("semantic_segmentation.dice_distribution")
def calculate_dice_distribution(data: dict, problem_type: str) -> dict:
    """
    A function to calculate the dice distribution for semantic segmentation

    :parm data: a dictionary containing the ground truth and prediction data
    :type data: dict
    :param problem_type: the type of problem
    :type problem_type: str

    :return: a dictionary containing the dice distribution
    :rtype: dict
    """
    metric_manager = problem_type_map[problem_type]()
    result = metric_manager.calculate(data)
    return result


@plot_manager.register("semantic_segmentation.dice_distribution")
def plot_dice_distribution(
    results: dict,
    save_plot: bool,
    file_name: Optional[str] = "dice_distribution.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to plot the threshold

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: Boolean value to save the plot
    :type save_plot: bool
    :param file_name: the name of the file
    :type file_name: str
    :param cohort_id: the cohort id
    :type cohort_id: int

    :return: the file path
    :rtype: str
    """
    plotter = PlotDiceDistribution(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()