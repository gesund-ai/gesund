from typing import Union, Optional
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager
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
        A function to preprocess

        :param data: dictionary data
        :type data: dict

        :return: gt, pred
        :rtype: tuple(dict, dict)
        """
        from .average_precision import ObjectDetection
        
        return ObjectDetection._preprocess(data)


    def _calc_conf_distR(
        self,
        gt_boxes_dict: dict,
        pred_boxes_dict: dict,
        class_mapping: dict,
        threshold: float = None,
        predicted_class: Optional[str] = None
    ) -> pd.DataFrame:
        """
        A function to calculate the loss function

        :param gt_boxes_dict: a dictionary of gt boxes, with image id as the key
        :type gt_boxes_dict: dict
        :param pred_boxes_dict: a dictionary of pred boxes, with image id as the key
        :type pred_boxes_dict: dict
        :param class_mapping: dict
        :type class_mapping: dict

        :return: pandas data frame
        """
        avg_conf = {"image_id": [], "avg_confidence": []}
        for image_id, p_boxes in pred_boxes_dict.items():
            if image_id not in gt_boxes_dict:
                continue
            confidences = []
            for box in p_boxes:
                # [x1, y1, x2, y2, label, confidence]
                if len(box) < 6:
                    continue
                label = box[4]
                conf = box[5]

                if threshold is not None and conf < threshold:
                    continue
                if predicted_class is not None and label != predicted_class:
                    continue

                confidences.append(conf)
            if confidences:
                avg_conf["image_id"].append(image_id)
                avg_conf["avg_confidence"].append(np.mean(confidences))

        return pd.DataFrame(avg_conf)


    def _calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        gt_boxes, pred_boxes = self._preprocess(data)

        thresholds = data["metric_args"]["threshold"]
        if thresholds is not None and not isinstance(thresholds, list):
            thresholds = [thresholds]

        all_dfs = []
        if not thresholds:  # 'thresholds' as [None] if not given
            thresholds = [None]

        for t in thresholds:
            df = self._calc_conf_distR(gt_boxes, pred_boxes, class_mapping, t)
            df["threshold"] = t
            all_dfs.append(df)

        final_df = pd.concat(all_dfs, ignore_index=True)
        return {"result": final_df}

    def calculate(self, data: dict) -> dict:
        result = {}
        self._validate_data(data)
        result = self._calculate_metrics(data, data.get("class_mapping"))
        return result


class PlotConfidenceDistribution:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        validates the data required for plotting the bar plot
        """
        if not isinstance(self.data["result"], pd.DataFrame):
            raise ValueError(f"Data must be a data frame.")

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
        A function to plot the confidence distribution
        """
        sns.set_theme(style="whitegrid")
        self._validate_data()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=self.data["result"],
            x="x",
            y="y",
            hue="labels",
            palette="rocket",
            s=100,
            alpha=0.7,
        )
        ax.set_title("Scatter Plot of Points", fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel("X-axis", fontdict={"fontsize": 14, "fontweight": "medium"})
        ax.set_ylabel("Y-axis", fontdict={"fontsize": 14, "fontweight": "medium"})
        ax.legend(loc="lower right", fontsize=12)
        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("object_detection.confidence_distribution")
def calculate_confidence_distribution(data: dict, problem_type: str):
    """
    A wrapper function to calculate the confidence_distribution metrics.
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result


@plot_manager.register("object_detection.confidence_distribution")
def plot_confidence_distribution_od(
    results: dict,
    save_plot: bool,
    file_name: str = "confidence_distribution.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to plot the confidence distribution metrics.
    """
    plotter = PlotConfidenceDistribution(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
