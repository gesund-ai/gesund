import os
from typing import Union, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
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

        cm = confusion_matrix(ground_truth, prediction, labels=class_order)
        confused_pairs = []
        for i, actual_class in enumerate(class_order):
            for j, pred_class in enumerate(class_order):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append(
                        {
                            "true": class_mapping[str(actual_class)],
                            "predicted": class_mapping[str(pred_class)],
                            "count": cm[i, j],
                        }
                    )
        confused_pairs = sorted(confused_pairs, key=lambda x: x["count"], reverse=True)

        result = {
            "confused_pairs": confused_pairs,
            "class_mapping": class_mapping,
            "class_order": class_order,
        }
        return result

    def calculate(self, data: dict) -> dict:
        """
        Calculates the lift chart for the given data.

        :param data: The input data required for calculation and plotting
                     {"prediction":, "ground_truth": , "metadata":, "class_mapping":}
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


class SemanticSegmentation(Classification):
    pass


class ObjectDetection:
    pass


class PlotMostConfused:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.confused_pairs = data["confused_pairs"]
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        Validates the data required for plotting.
        """
        if "confused_pairs" not in self.data:
            raise ValueError("Data must contain 'confused_pairs'.")

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

    def plot(self, top_k: int = 5) -> Figure:
        """
        Plots the most confused classes.

        :param top_k: Number of top confused pairs to display
        :type top_k: int

        :return: Matplotlib Figure object
        :rtype: Figure
        """
        sns.set_style("whitegrid")

        # validate the data
        self._validate_data()

        df = pd.DataFrame(self.confused_pairs[:top_k])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="count", y=df.index, data=df, orient="h")
        ax.set_yticks(df.index)
        ax.set_yticklabels(
            [
                f"True: {row['true']} | Predicted: {row['predicted']}"
                for _, row in df.iterrows()
            ]
        )
        ax.set_xlabel("Count", fontdict={"fontsize": 14, "fontweight": "medium"})
        ax.set_ylabel(
            "Confusion Pairs", fontdict={"fontsize": 14, "fontweight": "medium"}
        )

        if self.cohort_id:
            title_str = f"Most Confused Classes : cohort - {self.cohort_id}"
        else:
            title_str = "Most Confused Classes"

        ax.set_title(title_str, fontdict={"fontsize": 16, "fontweight": "medium"})
        fig.tight_layout()
        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("classification.most_confused")
def calculate_most_confused_metric(data: dict, problem_type: str):
    """
    Calculates the most confused classes metric.

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


@plot_manager.register("classification.most_confused")
def plot_most_confused(
    results: dict,
    save_plot: bool,
    file_name: str = "most_confused.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to plot the lift chart.

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
    plotter = PlotMostConfused(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
