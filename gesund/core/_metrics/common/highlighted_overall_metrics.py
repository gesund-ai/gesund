from typing import Union, Optional, List, Dict
import os
import itertools
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import sklearn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager
from gesund.core._metrics.common.average_precision import AveragePrecision


class Classification:
    pass


class SemanticSegmentation:
    pass


class ObjectDetection:
    def _validate_data(self):
        pass

    def __preprocess(self):
        pass

    def __calculate_metrics(self):
        pass

    def calculate(self, data: dict) -> dict:
        self._validate_data(data)
        result = self.__calculate_metrics(data, data.get("class_mappings"))
        return result


class PlotHighlightedOverallMetrics:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        # TODO: Continue from here
        self.cohort_id = cohort_id

    def _validate_data(self):
        # TODO: Continue init parameters in here.
        pass

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
        Plot the highlighted overall metrics.
        """
        sns.set_style("whitegrid")

        self._validate_data()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(
            x="Value",
            y="Metric",
            hue="Metric",
            data=self.df,  # Check, check, check df
            palette="pastel",
            edgecolor="black",
            legend=False,
        )
        ax.set_title("Overall Metrics", fontsize=20, fontweight="bold", pad=20)
        ax.set_xlabel("Metric Value", fontsize=14, labelpad=15)
        ax.set_ylabel("Metric", fontsize=14, labelpad=15)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

        fig.tight_layout()
        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("object_detection.highlighted_overall_metrics")
def calculate_highlighted_overall_metrics(data: dict, problem_type: str):
    """
    A wrapper function to calculate the highlighted overall metrics.
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result


@plot_manager.register("object_detection.highlighted_overall_metrics")
def plot_highlighted_overall_metrics_od(
    results: dict, save_plot: bool, file_name: str = "highlighted_overall_metrics.png"
) -> Union[str, None]:
    """
    A wrapper function to plot the highlighted overall metrics.
    """
    plotter = PlotHighlightedOverallMetrics(data=results)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()

    # TODO->1 : Calculate
    """
    def calculate_highlighted_overall_metrics(self, threshold):

        pred_coco = self.coco_[0]
        gt_coco = self.coco_[1]

        return self.calculate_coco_metrics(pred_coco, gt_coco)
    """

    # TODO-> 2 : Plot
    """
    def _plot_highlighted_overall_metrics(self, threshold: float) -> Dict[str, Any]:


        Plot highlighted overall metrics at specified threshold.

        :param threshold: IoU threshold value
        :type threshold: float
        :return: Dictionary containing overall metrics data
        :rtype: Dict[str, Any]

        rename_dict = {
            f"map{int(threshold*100)}": f"mAP@{int(threshold*100)}",
            f"map{int(threshold*100)}_11": f"mAP11@{int(threshold*100)}",
            "map50": "mAP@50",
            "map75": "mAP@75",
            "map5095": "mAP@[.50,.95]",
            "mar1": "mAR@max=1",
            "mar10": "mAR@max=10",
            "mar100": "mAR@max=100",
        }
        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )
        overall_metrics = average_precision.calculate_highlighted_overall_metrics(
            threshold
        )
        for metric in list(overall_metrics.keys()):
            overall_metrics[rename_dict[metric]] = overall_metrics.pop(metric)

        val_train_dict = {}
        for value in rename_dict.values():
            val_train_dict[value] = {"Validation": overall_metrics[value]}

        payload_dict = {"type": "overall", "data": val_train_dict}
        return payload_dict
    """


# TODO->3 : PLot steps
"""
    def _plot_overall_metrics(self, overall_args, save_path=None):

        Create a bar plot visualization for overall validation metrics.

        :param overall_args: (dict, optional) Dictionary containing:
            - 'metrics': (list) List of specific metrics to include in the plot
            - 'threshold': (float) Minimum threshold value for filtering metrics
        :param save_path: (str, optional) Path where the plot should be saved.
                        If None, saves as 'overall_metrics.png'.

        :return: None
        :raises AttributeError: If no valid overall data is loaded.


        if (
            not self.overall_json_data
            or self.overall_json_data.get("type") != "overall"
        ):
            print("No valid 'overall' data found in the JSON.")
            return

        data = self.overall_json_data.get("data", {})
        df = pd.DataFrame(
            [(k, v["Validation"]) for k, v in data.items()], columns=["Metric", "Value"]
        )

        if overall_args:
            if "metrics" in overall_args:
                df = df[df["Metric"].isin(overall_args["metrics"])]
            if "threshold" in overall_args:
                df = df[df["Value"] > overall_args["threshold"]]

        df = df.sort_values("Value", ascending=False)

        plt.figure(figsize=(14, 8))
        sns.barplot(
            x="Value",
            y="Metric",
            hue="Metric",
            data=df,
            palette="pastel",
            edgecolor="black",
            legend=False,
        )

        plt.title("Overall Metrics", fontsize=20, fontweight="bold", pad=20)
        plt.xlabel("Metric Value", fontsize=14, labelpad=15)
        plt.ylabel("Metric", fontsize=14, labelpad=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.grid(True, axis="x", linestyle="--", alpha=0.7)

        for i, value in enumerate(df["Value"]):
            plt.text(
                value + 0.01,
                i,
                f"{value:.4f}",
                va="center",
                ha="left",
                fontsize=12,
                color="black",
                fontweight="bold",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig("overall_metrics.png", bbox_inches="tight", dpi=300)

        plt.show()
        plt.close()
"""

# TODO->4 : If need overall metric arguments.
"""
        'overall_metrics': {'overall_metrics_metrics': ['map', 'mar'], 'threshold': 0.5},

"""
