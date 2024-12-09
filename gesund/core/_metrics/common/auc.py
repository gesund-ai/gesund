from typing import Union, Callable, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    auc as calculate_auc,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gesund.core import metric_manager, plot_manager


class Classification:
    def _validate_data(self, data: dict) -> bool:
        """
        Validates the data required for metric calculation and plotting.

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

        return True

    def apply_metadata(self, data: dict, metadata: dict) -> dict:
        """
        Applies metadata to the data for metric calculation and plotting.

        :param data: The input data required for calculation, {"prediction":, "ground_truth": , "metadata":}
        :type data: dict
        :param metadata: The metadata to apply
        :type metadata: dict

        :return: Filtered dataset
        :rtype: dict
        """
        return data

    def calculate(self, data: dict) -> dict:
        """
        Calculates the AUC metric for the given dataset.

        :param data: The input data required for calculation and plotting {"prediction":, "ground_truth": , "metadata":}
        :type data: dict

        :return: Calculated metric results
        :rtype: dict
        """
        # Validate the data
        self._validate_data(data)

        # Extract predictions and ground truth
        true = np.array(data["ground_truth"])
        pred_logits = np.array(data["prediction"])
        metadata = data.get("metadata", None)

        # Apply metadata if given
        if metadata is not None:
            data = self.apply_metadata(data, metadata)
            true = np.array(data["ground_truth"])
            pred_logits = np.array(data["prediction"])

        # Get class mappings if provided, else infer from data
        class_mappings = data.get("class_mappings", None)
        if class_mappings is not None:
            class_order = [int(i) for i in list(class_mappings.keys())]
        else:
            # Infer class order from data
            classes = np.unique(true)
            class_order = classes.tolist()
            class_mappings = {int(i): str(i) for i in class_order}

        # Binarize the output for multiclass
        y_true = label_binarize(true, classes=class_order)
        if len(class_order) == 2:
            y_true = np.hstack((1 - y_true, y_true))

        # Calculate ROC and AUC
        aucs = {}
        fpr = {}
        tpr = {}
        for idx, class_idx in enumerate(class_order):
            fpr[class_idx], tpr[class_idx], _ = roc_curve(
                y_true[:, idx], pred_logits[:, idx]
            )
            aucs[class_idx] = calculate_auc(fpr[class_idx], tpr[class_idx])

        result = {
            "fpr": fpr,
            "tpr": tpr,
            "aucs": aucs,
            "class_mappings": class_mappings,
            "class_order": class_order,
        }

        return result


class SemanticSegmentation(Classification):
    pass


class ObjectDetection:
    pass


class PlotAuc:
    def __init__(self, data: dict):
        self.data = data
        self.fpr = data["fpr"]
        self.tpr = data["tpr"]
        self.aucs = data["aucs"]
        self.class_mappings = data["class_mappings"]
        self.class_order = data["class_order"]

    def _validate_data(self):
        """
        Validates the data required for plotting the AUC.
        """
        required_keys = ["fpr", "tpr", "aucs"]
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Data must contain '{key}'.")

    def save(self, fig: Figure, filepath: str = "auc_plot.png") -> str:
        """
        Saves the plot to a file.

        :param fig: Matplotlib figure object to save
        :type fig: Figure
        :param filepath: Path where the plot image will be saved
        :type filepath: str

        :return: Path where the plot image is saved
        :rtype: str
        """
        fig.savefig(filepath)
        return filepath

    def plot(self) -> Figure:
        """
        Plots the AUC curves.

        :return: Matplotlib Figure object
        :rtype: Figure
        """
        self._validate_data()

        fig = plt.figure(figsize=(10, 7))
        for class_idx in self.class_order:
            plt.plot(
                self.fpr[class_idx],
                self.tpr[class_idx],
                label=f"Class {self.class_mappings[class_idx]} (AUC = {self.aucs[class_idx]:.2f})",
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curves")
        plt.legend(loc="lower right")

        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("classification.auc")
def calculate_auc_metric(data: dict, problem_type: str):
    """
    A wrapper function to calculate the AUC metric.

    :param data: Dictionary of data: {"prediction": , "ground_truth": }
    :type data: dict
    :param problem_type: Type of the problem
    :type problem_type: str

    :return: Dict of calculated results
    :rtype: dict
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result


@plot_manager.register("classification.auc")
def plot_auc(results: dict, save_plot: bool) -> Union[str, Figure]:
    """
    A wrapper function to plot the AUC curves.

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: Boolean value to save plot
    :type save_plot: bool

    :return: Figure object or path to the saved plot
    :rtype: Union[str, Figure]
    """
    plotter = PlotAuc(data=results)
    fig = plotter.plot()

    if save_plot:
        return plotter.save(fig)

    return fig
