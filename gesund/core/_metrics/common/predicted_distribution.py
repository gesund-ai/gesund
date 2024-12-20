from typing import Union, Optional
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager
from gesund.core._utils import ValidationUtils
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
        A function to preprocess the data

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict

        :return: a tuple containing the ground truth and prediction boxes
        :rtype: tuple
        """
        from .average_precision import ObjectDetection

        return ObjectDetection._preprocess(data, get_label=True)

    def _organize_data(
        self, gt_boxes: dict, pred_boxes: dict, class_mapping: dict
    ) -> dict:
        """
        A function to organize the data

        :param gt_boxes: a dictionary containing the ground truth boxes
        :type gt_boxes: dict
        :param pred_boxes: a dictionary containing the prediction boxes
        :type pred_boxes: dict
        :param class_mapping: a dictionary containing the class mapping
        :type class_mapping: dict

        :return: a dictionary containing the organized data
        :rtype: dict
        """
        data = {"pred_class_label": [], "gt_class_label": []}

        for image_id in gt_boxes:
            data["gt_class_label"].extend([box[-1] for box in gt_boxes[image_id]])
            data["pred_class_label"].extend([box[-1] for box in pred_boxes[image_id]])

        return data

    def _calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        """
        A function to calculate the predicted_distribution metrics

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict
        :param class_mapping: a dictionary containing the class mapping
        :type class_mapping: dict

        :return: a dictionary containing the calculated metrics
        :rtype: dict
        """
        results = {}
        # preprocess the data
        gt_boxes, pred_boxes = self._preprocess(data)

        label_dist = self._organize_data(gt_boxes, pred_boxes, class_mapping)

        results["label_distribution"] = label_dist

        return results

    def calculate(self, data: dict) -> dict:
        """
        A function to calculate the predicted_distribution metrics

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


class PlotPredictedDistribution:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        Validate the data
        """
        if "label_distribution" not in self.data:
            raise ValueError("Missing label_distribution in the data dictionary")

    def save(self, fig: Figure, filename: str) -> str:
        """
        A function to save the plot

        :param fig: the figure object
        :type fig: Figure
        :param filename: the name of the file
        :type filename: str

        :return: the path to the saved file
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
        A function to plot the predicted distribution

        :return: the figure object
        :rtype: Figure
        """
        # validate the data
        self._validate_data()

        sns.set_theme(style="whitegrid")
        figx, ax = plt.subplots(1, 2, figsize=(10, 6))

        plot_data = self.data["label_distribution"]
        plot_data = pd.DataFrame(plot_data)

        for i, column in enumerate(plot_data.columns):
            value_counts = plot_data[column].value_counts(normalize=True)
            wedges, texts, autotexts = ax[i].pie(
                value_counts,
                labels=value_counts.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=sns.color_palette("pastel"),
                wedgesprops=dict(width=0.3),
            )
            for wedge in wedges:
                wedge.set_edgecolor("white")
            ax[i].set_title(f"{column} Distribution")

        title_txt = f"Class Distribution: {self.data['overall_loss']} : Top 20"
        if self.cohort_id:
            title_str = f"{title_txt}: cohort - {self.cohort_id}"
        else:
            title_str = f"{title_txt}"

        ax.set_title(title_str)
        plt.tight_layout()

        return figx


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("object_detection.predicted_distribution")
def calculate_predicted_distribution(data: dict, problem_type: str):
    """
    A wrapper function to calculate the predicted_distribution metrics.

    :param data: a dictionary containing the ground truth and prediction data
    :type data: dict
    :param problem_type: the type of problem
    :type problem_type: str

    :return: calculated metric results
    :rtype: dict
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

    :param results: a dictionary containing the results of the predicted distribution
    :type results: dict
    :param save_plot: a flag to save the plot
    :type save_plot: bool
    :param file_name: the name of the file
    :type file_name: str
    :param cohort_id: the cohort id
    :type cohort_id: Optional[int]

    :return: the path to the saved file
    :rtype: Union[str, None]
    """
    plotter = PlotPredictedDistribution(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
