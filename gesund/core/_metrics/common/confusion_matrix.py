from typing import Union, Callable, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

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

        return True

    def apply_metadata(self, data: dict, metadata: dict) -> dict:
        """
        A function to apply the metadata on the data for metric calculation and plotting.

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
        A function to calculate the confusion matrix for the given dataset.

        :param data: The input data required for calculation and plotting {"prediction":, "ground_truth": , "metadata":}
        :type data: dict

        :return: Calculated metric
        :rtype: dict
        """
        # Validate the data
        self._validate_data(data)

        # Extract predictions and ground truth
        true = np.array(data["ground_truth"])
        pred_categorical = np.array(data["prediction"])
        metadata = data.get("metadata", None)

        # Apply metadata if given
        if metadata is not None:
            data = self.apply_metadata(data, metadata)
            true = np.array(data["ground_truth"])
            pred_categorical = np.array(data["prediction"])

        # Get class mappings if provided, else infer from data
        class_mappings = data.get("class_mappings", None)
        if class_mappings is not None:
            class_order = [int(i) for i in list(class_mappings.keys())]
        else:
            # Infer class order from data
            classes = np.unique(np.concatenate((true, pred_categorical)))
            class_order = classes.tolist()
            class_mappings = {int(i): str(i) for i in class_order}

        # Run the calculation logic
        cm = sklearn_confusion_matrix(true, pred_categorical, labels=class_order)

        result = {
            "confusion_matrix": cm,
            "class_mappings": class_mappings,
            "class_order": class_order,
        }

        return result


class SemanticSegmentation(Classification):
    pass


class ObjectDetection:
    pass


class PlotConfusionMatrix:
    def __init__(self, data: dict):
        self.data = data
        self.confusion_matrix = data["confusion_matrix"]
        self.class_mappings = data["class_mappings"]
        self.class_order = data["class_order"]

    def _validate_data(self):
        """
        A function to validate the data required for plotting the confusion matrix.
        """
        if "confusion_matrix" not in self.data:
            raise ValueError("Data must contain 'confusion_matrix'.")

    def save(self, fig: Figure, filepath: str = "confusion_matrix.png") -> str:
        """
        A function to save the plot.

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
        Logic to plot the confusion matrix.

        :return: Matplotlib Figure object
        :rtype: Figure
        """
        # Validate the data
        self._validate_data()

        fig = plt.figure(figsize=(10, 7))
        df_cm = pd.DataFrame(
            self.confusion_matrix,
            index=[self.class_mappings[i] for i in self.class_order],
            columns=[self.class_mappings[i] for i in self.class_order],
        )
        sns.heatmap(df_cm, annot=True, fmt="g", cmap="Blues")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title("Confusion Matrix")

        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("classification.confusion_matrix")
def calculate_confusion_matrix(data: dict, problem_type: str):
    """
    A wrapper function.

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


@plot_manager.register("classification.confusion_matrix")
def plot_confusion_matrix(results: dict, save_plot: bool) -> Union[str, Figure]:
    """
    A wrapper function.

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: Boolean value to save plot
    :type save_plot: bool

    :return: Figure object or path to the saved plot
    :rtype: Union[str, Figure]
    """
    plotter = PlotConfusionMatrix(data=results)
    fig = plotter.plot()

    if save_plot:
        return plotter.save(fig)

    return fig
