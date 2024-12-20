import os
from typing import Union, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from matplotlib.figure import Figure

from gesund.core import metric_manager, plot_manager

COHORT_SIZE_LIMIT = 2
DEBUG = True


def categorize_age(age):
    if age < 18:
        return "Child"
    elif 18 <= age < 30:
        return "Young Adult"
    elif 30 <= age < 60:
        return "Adult"
    else:
        return "Senior"


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

    def _preprocess(self, data: dict, get_logits=False):
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
            prediction, ground_truth = self._preprocess(data, get_logits=True)
        else:
            prediction, ground_truth = self._preprocess(data)

        cm = sklearn_confusion_matrix(ground_truth, prediction, labels=class_order)

        result = {
            "confusion_matrix": cm,
            "class_mapping": class_mapping,
            "class_order": class_order,
        }

        return result

    def calculate(self, data: dict) -> dict:
        """
        Calculates the confusion matrix for the given data.

        :param data: The input data required for calculation and plotting
                     {"prediction":, "ground_truth": , "metadata":, "class_mappings":}
        :type data: dict

        :return: Calculated metric results
        :rtype: dict
        """
        result = {}

        # Validate the data
        self._validate_data(data)

        # calculaee the metrics
        result = self.__calculate_metrics(data, data.get("class_mapping"))

        return result


class SemanticSegmentation(Classification):
    pass


class ObjectDetection:
    pass


class PlotConfusionMatrix:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.confusion_matrix = data["confusion_matrix"]
        self.class_mapping = data["class_mapping"]
        self.class_order = data["class_order"]
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        A function to validate the data required for plotting the confusion matrix.
        """
        if "confusion_matrix" not in self.data:
            raise ValueError("Data must contain 'confusion_matrix'.")

        # TODO: require more through validationg for the data structure that is required for plotting
        # the confusion matrix

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

        if self.cohort_id:
            filepath = f"{dir_path}/{self.cohort_id}_{filename}"
        else:
            filepath = f"{dir_path}/{filename}"

        fig.savefig(filepath, format="png")
        return filepath

    def plot(self) -> Figure:
        """
        Logic to plot the confusion matrix.

        :return: Matplotlib Figure object
        :rtype: Figure
        """
        # Validate the data
        self._validate_data()
        fig, ax = plt.subplots(figsize=(10, 7))
        df_cm = pd.DataFrame(
            self.confusion_matrix,
            index=[self.class_mapping[str(i)] for i in self.class_order],
            columns=[self.class_mapping[str(i)] for i in self.class_order],
        )
        sns.heatmap(
            df_cm,
            annot=True,
            fmt="g",
            cmap=sns.dark_palette(color="#79C", as_cmap=True),
            ax=ax,
        )
        ax.set_ylabel("True label", fontdict={"fontsize": 14, "fontweight": "medium"})
        ax.set_xlabel(
            "Predicted label", fontdict={"fontsize": 14, "fontweight": "medium"}
        )

        if self.cohort_id:
            title_str = f"Confusion matrix : cohort - {self.cohort_id}"
        else:
            title_str = "Confusion matrix"

        ax.set_title(title_str, fontdict={"fontsize": 16, "fontweight": "medium"})
        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("classification.confusion_matrix")
def calculate_confusion_matrix(data: dict, problem_type: str):
    """
    A wrapper function to calculate the confusion matrix.

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
def plot_confusion_matrix(
    results: dict,
    save_plot: bool,
    file_name: str = "confusion_matrix.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to plot the confusion matrix.

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: Boolean value to save plot
    :type save_plot: bool
    :param file_name: Name of the file to save the plot
    :type file_name: str
    :param cohort_id: id of the cohort
    :type cohort_id: int

    :return: None or path to the saved plot
    :rtype: Union[str, None]
    """
    plotter = PlotConfusionMatrix(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
