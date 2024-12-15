from typing import Union, Optional, List, Dict
import os

import numpy as np
import pandas as pd
import itertools
from datetime import datetime
import pickle
import sklearn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core._utils import ValidationUtils
from gesund.core import metric_manager, plot_manager
from gesund.core._metrics.common.average_precision import AveragePrecision

COHORT_SIZE_LIMIT = 2
DEBUG = True


class Classification:
    pass


class SemanticSegmentation(Classification):
    pass


class ObjectDetection:
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

    def __preprocess(self, data: dict, get_logits=False) -> tuple:
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
        class_order = [int(i) for i in list(class_mapping.keys())]
        # TODO: class wise auc and roc

        if len(class_order) > 2:

            prediction, ground_truth = self.__preprocess(data, get_logits=True)

        else:
            prediction, ground_truth = self.__preprocess(data)

            # TODO: check if the methods for that script are available

        return data

    def calculate(self, data: dict) -> dict:
        """
        Calculates the Top Losses metric for the given dataset.

        :param data: The input data required for calculation and plotting
                     {"prediction":, "ground_truth": , "class_mappings":}
        :type data: dict

        :return: Calculated metric results
        :rtype: dict
        """
        result = {}

        # Validate the data
        self._validate_data(data)

        # calculate results
        result = self.__calculate_metrics(data, data.get("class_mapping"))

        return result


class TopLosses:
    def __init__(self, coco_, class_mappings, loss):
        self.coco_ = coco_
        self.class_mappings = class_mappings
        self.loss = loss

    def calculate_top_losses(self, top_k=100):
        """
        Calculates and returns the top_k samples with the highest losses.
        """
        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )

        losses_list = average_precision.plot_top_losses()

        # Sort and get top_k losses
        sorted_losses = sorted(losses_list, key=lambda x: x["mIoU"])
        top_losses = sorted_losses[:top_k]

        return top_losses


class PlotTopLosses:
    def __init__(
        self, coco_, class_mappings, loss=None, meta_dict=None, cohort_id=None
    ):
        self.loss = loss
        self.meta_dict = meta_dict
        self.coco_ = coco_
        self.class_mappings = class_mappings
        self.is_meta_exists = False
        self.cohort_id = cohort_id
        if self.meta_dict is not None:
            self.is_meta_exists = True
        self.top_losses_instance = TopLosses(
            loss=self.loss, coco_=self.coco_, class_mappings=self.class_mappings
        )

    def _validate_data(self):
        """
        Validates the data required for plotting.
        """
        if not self.coco_ or not self.class_mappings:
            raise ValueError("COCO data and class mappings must be provided.")

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
        Plots the Top Losses.
        """
        self._validate_data()
        top_losses = self.top_losses_instance.calculate_top_losses()

        # Extract data for plotting
        image_ids = [item["image_id"] for item in top_losses]
        mIoUs = [item["mIoU"] for item in top_losses]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=image_ids, y=mIoUs, ax=ax)
        ax.set_title("Top Losses")
        ax.set_xlabel("Image ID")
        ax.set_ylabel("Mean IoU")
        plt.xticks(rotation=90)
        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("object_detection.top_losses")
def calculate_top_losses(data: dict, problem_type: str):
    """
    A wrapper function to calculate the Top Losses metric.

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


@plot_manager.register("object_detection.top_losses")
def plot_top_losses(
    results: dict,
    save_plot: bool,
    file_name: str = "top_losses.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to plot the Top Losses curves.
    """
    plotter = PlotTopLosses(
        coco_=results.get("coco_"),
        class_mappings=results.get("class_mappings"),
        loss=results.get("loss"),
        meta_dict=results.get("meta_dict"),
        cohort_id=cohort_id,
    )
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
