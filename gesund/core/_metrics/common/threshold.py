import os
from typing import Union, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from sklearn.metrics import roc_curve

from gesund.core import metric_manager, plot_manager


class Classification:
    def _validate_data(self, data: dict) -> bool:
        """
        A function to validate the data that is required for metric calculation and plotting.

        :param data: The input data required for calculation, {"prediction":, "ground_truth": , "metadata":}
        :type data: dict

        :return: Status if the data is valid
        :rtype: bool
        """
        # Basic validation checks
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        required_keys = ["prediction", "ground_truth"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Data must contain '{key}'.")

        if len(data["prediction"]) != len(data["ground_truth"]):
            raise ValueError("Prediction and ground_truth must have the same length.")

        if (
            len(
                set(list(data["prediction"].keys())).difference(
                    set(list(data["ground_truth"].keys()))
                )
            )
            > 0
        ):
            raise ValueError("Prediction and ground_truth must have the same keys.")

        return True

    def __preprocess(self, data: dict, get_logits=False):
        """
        Preprocesses the data

        :param data: dictionary containing the data prediction, ground truth, metadata
        :type data: dict
        :param get_logits: in case of multi class classification set to True
        :type get_logits: boolean

        :return: data tuple
        :rtype: tuple
        """
        prediction, ground_truth = [], []
        for image_id in data["ground_truth"]:
            sample_gt = data["ground_truth"][image_id]
            sample_pred = data["prediction"][image_id]
            ground_truth.append(sample_gt["annotation"][0]["label"])

            if get_logits:
                prediction.append(sample_pred["logits"])
            else:
                prediction.append(sample_pred["prediction_class"])
        return (np.asarray(prediction), np.asarray(ground_truth))

    def __calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        """
        A function to calculate the metrics

        :param data: data dictionary containing data
        :type data: dict
        :param class_mapping: a dictionary with class mapping labels
        :type class_mapping: dict

        :return: results calculated
        :rtype: dict
        """
        class_order = [int(i) for i in class_mapping.keys()]

        if len(class_order) > 2:
            prediction, ground_truth = self.__preprocess(data, get_logits=True)
        else:
            prediction, ground_truth = self.__preprocess(data)

        fpr, tpr, thresholds = roc_curve(ground_truth, prediction)

        # Calculate optimal threshold using Youden's J statistic
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]

        result = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "optimal_threshold": optimal_threshold,
            "optimal_fpr": fpr[optimal_idx],
            "optimal_tpr": tpr[optimal_idx],
        }

        return result

    def calculate(self, data: dict) -> dict:
        """
        Calculates the top losses for the given dataset.

        :param data: The input data required for calculation and plotting
                {"prediction":, "ground_truth": , "metadata":, "class_mappings":}
        :type data: dict

        :return: Calculated metric results
        :rtype: dict
        """
        result = {}

        # Validate the data
        self._validate_data(data)

        # calculate the metrics
        result = self.__calculate_metrics(data, data.get("class_mapping"))

        return result


class PlotThreshold:
    def __init__(self, data: dict):
        self.data = data
        self.fpr = data["fpr"]
        self.tpr = data["tpr"]
        self.thresholds = data["thresholds"]
        self.optimal_threshold = data["optimal_threshold"]
        self.optimal_fpr = data["optimal_fpr"]
        self.optimal_tpr = data["optimal_tpr"]

    def _validate_data(self):
        """
        Validates the data required for plotting the threshold.
        """
        required_keys = [
            "fpr",
            "tpr",
            "thresholds",
            "optimal_threshold",
            "optimal_fpr",
            "optimal_tpr",
        ]
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Data must contain '{key}'.")

    def save(self, fig: Figure, filename: str) -> str:
        """
        Saves the plot to a file.

        :param filename: Path where the plot image will be saved
        :type filename: str

        :return: Path where the plot image is saved
        :rtype: str
        """
        dir_path = "plots"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filepath = f"{dir_path}/{filename}"
        fig.savefig(filepath, format="png")
        return filepath


def plot(self) -> Figure:
    """
    Plots the ROC Curve with the optimal threshold highlighted.
    """
    self._validate_data()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(self.fpr, self.tpr, label="ROC Curve")
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance")
    # Highlight the optimal threshold point
    ax.scatter(
        self.optimal_fpr,
        self.optimal_tpr,
        s=100,
        c="red",
        label=f"Optimal Threshold: {self.optimal_threshold:.2f}",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve with Optimal Threshold")
    ax.legend(loc="lower right")
    ax.grid(True)
    return fig


class SemanticSegmentation(Classification):
    pass


class ObjectDetection:
    pass


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("classification.threshold")
def calculate_threshold_metric(data: dict, problem_type: str):
    """
    A wrapper function to calculate the threshold metric.

    :param data: Dictionary of data: {"prediction": , "ground_truth": }
    :type data: dict
    :param problem_type: Type of the problem
    :type problem_type: str

    :return: Calculated results
    :rtype: dict
    """
    metric_calculator = problem_type_map[problem_type]()
    result = metric_calculator.calculate(data)
    return result


@plot_manager.register("classification.threshold")
def plot_threshold(
    results: dict, save_plot: bool, file_name: str = "threshold.png"
) -> Union[str, None]:
    """
    A wrapper function to plot the threshold.

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: Boolean value to save plot
    :type save_plot: bool

    :return: None or path to the saved plot
    :rtype: Union[str, None]
    """
    plotter = PlotThreshold(data=results)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
