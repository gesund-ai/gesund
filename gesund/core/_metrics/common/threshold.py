import os
from typing import Union, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
)
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

    @staticmethod
    def _apply_threshold(
        threshold: float, y_true: np.ndarray, y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        A function to apply threshold

        :param threshold: The threshold value to be applied
        :type threshold: float
        :param y_true: the ground truth values in numpy array
        :type y_true: np.ndarray
        :param y_pred: the probabilites in numpy array
        :type y_pred: np.ndarray

        :return: results calculated and composed in dataframe
        :rtype: pd.DataFrame
        """
        predicted = (y_pred >= threshold).astype(int)

        # calculate the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()

        # calculate the metrics
        metrics = {}
        metrics["f1_score"] = f1_score(y_true, predicted)
        metrics["precision"] = precision_score(y_true, predicted)
        metrics["sensitivity"] = recall_score(y_true, predicted)
        metrics["specificity"] = round(tn / (tn + fp), 4)
        metrics["mcc"] = matthews_corrcoef(y_true, predicted)
        metrics["fpr"] = round(fp / (fp + tn), 4)
        metrics["fnr"] = round(fn / (fn + tp), 4)

        metrics = {i: [metrics[i]] for i in metrics}
        metrics["threshold"] = [threshold]
        metrics = pd.DataFrame(metrics)
        return metrics

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
        from gesund.core._metrics.common.top_losses import Classification

        prediction, ground_truth = self._preprocess(data, get_logits=True)
        pred_gt_df = pd.DataFrame(prediction)

        # softmax is applied to make the logits aka the raw outputs of the model
        # comparable and to sum upto 1
        probabilities = Classification._softmax(prediction)
        pred_gt_df["probabilities"] = probabilities[:, 1]
        pred_gt_df["ground_truth"] = ground_truth

        thresholds = data["metric_args"]["threshold"]
        th_result = pd.DataFrame()
        if isinstance(thresholds, list):
            for _threshold in thresholds:
                _result = self._apply_threshold(
                    threshold=_threshold,
                    y_true=pred_gt_df["ground_truth"],
                    y_pred=pred_gt_df["probabilities"],
                )

                th_result = pd.concat([th_result, _result], axis=0, ignore_index=True)
        else:
            _result = self._apply_threshold(
                threshold=thresholds,
                y_true=pred_gt_df["ground_truth"],
                y_pred=pred_gt_df["probabilities"],
            )
            th_result = pd.concat([th_result, _result], axis=0, ignore_index=True)

        result = {"threshold_result": th_result, "pred_gt_data": pred_gt_df}
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
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        Validates the data required for plotting the threshold.
        """
        required_keys = ["threshold_result"]
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

        if self.cohort_id:
            filepath = f"{dir_path}/{self.cohort_id}_{filename}"
        else:
            filepath = f"{dir_path}/{filename}"

        fig.savefig(filepath, format="png")
        return filepath

    def plot(self) -> Figure:
        """
        Plots the ROC Curve with the optimal threshold highlighted.
        """
        sns.set_style("whitegrid")

        self._validate_data()
        fig, ax = plt.subplots(figsize=(10, 7))
        plot_data = pd.melt(
            self.data["threshold_result"],
            id_vars=["threshold"],
            var_name="metric",
            value_name="value",
        )

        sns.barplot(
            x="metric",
            y="value",
            hue="threshold",
            data=plot_data,
            ax=ax,
            palette="pastel",
        )
        ax.set_xlabel("Metric", fontdict={"fontsize": 14, "fontweight": "medium"})
        ax.set_ylabel("Value", fontdict={"fontsize": 14, "fontweight": "medium"})

        if self.cohort_id:
            title_str = f"Threshold adjusted metrics : cohort - {self.cohort_id}"
        else:
            title_str = "Threshold adjusted metrics"

        ax.set_title(
            title_str,
            fontdict={"fontsize": 16, "fontweight": "medium"},
        )
        ax.legend(loc="lower right")
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
    :param file_name: Name of the file
    :type file_name: str

    :return: None or path to the saved plot
    :rtype: Union[str, None]
    """
    plotter = PlotThreshold(data=results)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
