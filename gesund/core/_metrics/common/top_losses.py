from typing import Union, Optional
import os

import numpy as np
import pandas as pd

# TODO 1: check if the methods has specific imports

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

    def __preprocess(
        self, data: dict, get_logits: bool = False, keep_imageid: bool = False
    ):
        """
        Preprocesses the data

        :param data: dictionary containing the data prediction, ground truth, metadata
        :type data: dict
        :param get_logits: in case of multi class classification set to True
        :type get_logits: boolean
        :param keep_imageid: to keep the image id in the response set to True
        :type keep_imageid: bool

        :return: data tuple
        :rtype: tuple
        """
        prediction, ground_truth, image_id_list = [], [], []
        for image_id in data["ground_truth"]:
            sample_gt = data["ground_truth"][image_id]
            sample_pred = data["prediction"][image_id]
            ground_truth.append(sample_gt["annotation"][0]["label"])
            image_id_list.append(image_id)

            if get_logits:
                prediction.append(sample_pred["logits"])
            else:
                prediction.append(sample_pred["prediction_class"])

        return (np.asarray(prediction), np.asarray(ground_truth), image_id_list)

    @staticmethod
    def _softmax(logits) -> np.ndarray:
        """
        A function to calculate the softmax

        :param logits: logits
        :type logits: np.ndarray

        :return: computed probabilities from logits
        :rtype: np.ndarray
        """
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities

    def _calculate_loss(
        self, y_true: np.ndarray, y_pred: np.ndarray, overall: bool = False
    ) -> float:
        # Convert logits to probabilities
        probabilities = self._softmax(y_pred)
        # Clip probabilities to prevent log(0)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
        # Create one-hot encoded labels
        y_true_one_hot = np.eye(np.max(y_true) + 1)[y_true]
        # Calculate cross-entropy loss

        if overall:
            # to calculate if the loss is calculated over all the examples
            loss = -np.sum(y_true_one_hot * np.log(probabilities)) / y_true.shape[0]
        else:
            loss = -np.log(probabilities[np.arange(len(y_true)), y_true])

        return np.round(loss, 4)

    def __calculate_metrics(self, data: dict) -> dict:
        """
        A function to calculate the metrics

        :param data: data dictionary containing data
        :type data: dict
        :param class_mapping: a dictionary with class mapping labels
        :type class_mapping: dict

        :return: results calculated
        :rtype: dict
        """
        prediction, ground_truth, image_id = self.__preprocess(
            data, get_logits=True, keep_imageid=True
        )
        pred_gt_df = pd.DataFrame(prediction)
        pred_gt_df["ground_truth"] = ground_truth.tolist()
        pred_gt_df["image_id"] = image_id

        # calculate the loss first
        # the loss considered is cross entropy loss as its the most general one used
        # for classification. Inorder to use a different loss function new class
        # method is supposed to defined and following line is replaced
        pred_gt_df["loss"] = self._calculate_loss(
            pred_gt_df["ground_truth"].to_numpy(), pred_gt_df[[0, 1]].to_numpy()
        )
        overall_loss = self._calculate_loss(
            pred_gt_df["ground_truth"].to_numpy(),
            pred_gt_df[[0, 1]].to_numpy(),
            overall=True,
        )

        # only the first 10 values with loss are picked up from the results
        # where the loss are sorted from highest to lowest order
        pred_gt_df = pred_gt_df.sort_values(by="loss", ascending=False)

        result = {"loss_data": pred_gt_df, "overall_loss": overall_loss}
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
        result = self.__calculate_metrics(data)
        return result


class PlotTopLosses:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        Validates the data required for plotting the top losses.
        """
        if "loss_data" not in self.data:
            raise ValueError("Data must contain 'loss_data'.")

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

    def plot(self, top_k: int = 9) -> Figure:
        """
        Plots the top losses.

        :param top_k: Number of top losses to display
        :type top_k: int
        :return: Matplotlib Figure object
        :rtype: Figure
        """
        # Validate the data
        self._validate_data()
        fig, ax = plt.subplots(figsize=(10, 7))
        plot_data = self.data["loss_data"].head(20)
        sns.barplot(
            data=plot_data,
            x="image_id",
            y="loss",
            ax=ax,
            palette="pastel",
            hue="ground_truth",
        )

        ax.set_xlabel("Image id", fontdict={"fontsize": 14, "fontweight": "medium"})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("Loss Value", fontdict={"fontsize": 14, "fontweight": "medium"})

        title_txt = f"Top losses : Overall loss: {self.data['overall_loss']} : Top 20"
        if self.cohort_id:
            title_str = f"{title_txt}: cohort - {self.cohort_id}"
        else:
            title_str = f"{title_txt}"

        ax.set_title(title_str, fontdict={"fontsize": 16, "fontweight": "medium"})
        ax.legend(loc="lower right")
        return fig



class SemanticSegmentation(Classification):
    pass


class ObjectDetection(Classification):
    pass


class PlotAveragePrecision:
    pass


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("object_detection.top_losses")
def calculate_top_losses(data: dict, problem_type: str):
    pass


@plot_manager.register("object_detection.top_losses")
def plot_top_losses(
<<<<<<< HEAD
    results: dict,
    save_plot: bool,
    file_name: str = "top_losses.png",
    cohort_id: Optional[int] = None,
=======
    results: dict, save_plot: bool, file_name: str = "top_losses.png"
>>>>>>> init:add top_losses module for object detection metrics and plotting
) -> Union[str, None]:
    """
    A wrapper function to plot the top losses chart.

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
    plotter = PlotTopLosses(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
