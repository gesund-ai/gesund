from typing import Union, Optional
import os

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt

from gesund.core import metric_manager, plot_manager
from .iou import IoUCalc


class AveragePrecision:
    pass


class Classification:
    pass


class SemanticSegmentation:
    pass


class ObjectDetection:
    def __init__(self):
        """
        Initializes the ObjectDetection metric calculator.

        Initializes the IoU calculator instance.
        """
        self.iou = IoUCalc()

    def _validate_data(self, data: dict) -> bool:
        """
        Validates the data required for metric calculation and plotting.

        :param data: The input data required for calculation, {"prediction":, "ground_truth": }
        :type data: dict

        :return: Status if the data is valid
        :rtype: bool
        """
        check_keys = ("ground_truth", "prediction", "class_mapping", "metric_args")
        for _key in check_keys:
            if _key not in data:
                raise ValueError(f"Missing {_key} in the data dictionary")

        common_ids = set(list(data["prediction"].keys())).difference(
            set(list(data["ground_truth"].keys()))
        )
        if common_ids:
            raise ValueError(
                "prediction and ground truth does not have corresponding samples"
            )

    @staticmethod
    def _preprocess(data: dict, get_label=False, get_pred_scores=False) -> tuple:
        """
        Preprocesses the input data to extract ground truth and prediction boxes.

        :param data: The input data containing ground truth and predictions
        :type data: dict
        :param get_label: Flag to include labels in the boxes, defaults to False
        :type get_label: bool, optional
        :param get_pred_scores: Flag to include prediction scores, defaults to False
        :type get_pred_scores: bool, optional

        :return: A tuple of ground truth boxes and prediction boxes
        :rtype: tuple
        """
        gt_boxes, pred_boxes = {}, {}
        for image_id in data["ground_truth"]:
            for _ant in data["ground_truth"][image_id]["annotation"]:
                pts = _ant["points"]
                box_points = [pts[0]["x"], pts[0]["y"], pts[1]["x"], pts[1]["y"]]
                if get_label:
                    box_points.append(_ant["label"])
                if image_id not in gt_boxes:
                    gt_boxes[image_id] = []
                gt_boxes[image_id].append(box_points)

            if image_id in data["prediction"]:
                for pred in data["prediction"][image_id]["objects"]:
                    pts = pred["box"]
                    box_points = [pts["x1"], pts["y1"], pts["x2"], pts["y2"]]
                    if get_label:
                        box_points.append(pred["prediction_class"])
                    if get_pred_scores:
                        box_points.append(pred["confidence"])
                    if image_id not in pred_boxes:
                        pred_boxes[image_id] = []
                    pred_boxes[image_id].append(box_points)
        return gt_boxes, pred_boxes

    def _calc_precision_recall(self, gt_boxes, pred_boxes, threshold: float) -> tuple:
        """
        Calculates precision and recall based on IoU threshold.

        :param gt_boxes: Ground truth bounding boxes
        :type gt_boxes: list
        :param pred_boxes: Predicted bounding boxes
        :type pred_boxes: list
        :param threshold: IoU threshold for determining true positives
        :type threshold: float

        :return: A tuple containing precision and recall
        :rtype: tuple
        """
        num_gt_boxes = len(gt_boxes)
        true_positives, false_positives = 0, 0

        for pred_box in pred_boxes:
            max_iou = 0
            for gt_box in gt_boxes:
                iou = self.iou.calculate(pred_box, gt_box)
                max_iou = max(max_iou, iou)

            if max_iou >= threshold:
                true_positives += 1
            else:
                false_positives += 1

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = true_positives / num_gt_boxes if num_gt_boxes > 0 else 0
        return (precision, recall)

    def _calc_mAP_mAR(self, gt_boxes_dict: dict, pred_boxes_dict: dict, thresholds: list):
        """
        Calculates mean Average Precision (mAP) and mean Average Recall (mAR) for given thresholds.

        :param gt_boxes_dict: Dictionary of ground truth boxes
        :type gt_boxes_dict: dict
        :param pred_boxes_dict: Dictionary of predicted boxes
        :type pred_boxes_dict: dict
        :param thresholds: List of IoU thresholds
        :type thresholds: list

        :return: DataFrame containing mAP and mAR values for each threshold
        :rtype: pd.DataFrame
        """
        results = {"metric": [], "threshold": [], "value": []}
        for threshold in thresholds or [0.5, 0.75, 0.95]:
            image_precisions, image_recalls = [], []
            for image_id in gt_boxes_dict:
                gt_boxes = gt_boxes_dict[image_id]
                pred_boxes = pred_boxes_dict.get(image_id, [])
                precision, recall = self._calc_precision_recall(gt_boxes, pred_boxes, threshold)
                image_precisions.append(precision)
                image_recalls.append(recall)

            avg_precision = np.mean(image_precisions) if image_precisions else 0
            avg_recall = np.mean(image_recalls) if image_recalls else 0
            results["metric"].append("mAP")
            results["threshold"].append(threshold)
            results["value"].append(avg_precision)
            results["metric"].append("mAR")
            results["threshold"].append(threshold)
            results["value"].append(avg_recall)

        return pd.DataFrame(results)

    def __calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        """
        Calculates the metrics based on input data and class mapping.

        :param data: The input data containing predictions and ground truth
        :type data: dict
        :param class_mapping: Mapping of class labels
        :type class_mapping: dict

        :return: DataFrame with calculated metrics
        :rtype: dict
        """
        self._validate_data(data)
        gt_boxes, pred_boxes = self._preprocess(data)
        thresholds = data["metric_args"].get("threshold", [])
        if not isinstance(thresholds, list):
            thresholds = [thresholds]
        results_df = self._calc_mAP_mAR(gt_boxes, pred_boxes, thresholds)
        return results_df

    def calculate(self, data: dict) -> dict:
        """
        Calculates the performance metrics for object detection.

        :param data: The input data containing predictions and ground truth
        :type data: dict

        :return: Dictionary with the results DataFrame
        :rtype: dict
        """
        result_df = self.__calculate_metrics(data, data.get("class_mapping", {}))
        return {"result": result_df}

class PlotTrainAndValidationPerformance:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        """
        Initializes the plotter for training and validation performance.

        :param data: The results data to plot
        :type data: dict
        :param cohort_id: Optional cohort identifier, defaults to None
        :type cohort_id: Optional[int], optional
        """
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        Validates the data required for plotting the bar plot.
        """
        if not isinstance(self.data["result"], pd.DataFrame):
            raise ValueError(f"Data must be a data frame.")

    def save(self, fig: Figure, filename: str) -> str:
        """
        Saves the plot to a file

        :param fig: the matplotlib figure object where the plot image is stored
        :type fig: str

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
        Generates the bar plot for training and validation performance.

        :return: The matplotlib figure object containing the plot
        :rtype: Figure
        """
        self._validate_data()
        df = self.data["result"]
        df_mAP = df[df["metric"] == "mAP"].copy()
        
        df_pivot = df_mAP.pivot_table(
            index=lambda x: 0, 
            columns="threshold", 
            values="value", 
            aggfunc="mean"
        )

        rename_map = {
            0.5: "mAP@50",
            0.75: "mAP@75",
            0.95: "mAP[.50,.95]"
        }
        df_pivot = df_pivot.rename(columns=rename_map)

        melted_df = df_pivot.melt(var_name="metric", value_name="val")

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(
            x="metric",
            y="val",
            hue="metric",
            data=melted_df,
            palette="pastel",
            legend=False,
            ax=ax
        )
        ax.set_xlabel("Metric", fontdict={"fontsize": 14})
        ax.set_ylabel("Value", fontdict={"fontsize": 14})

        if self.cohort_id:
            title_str = f"Train and Validation Performance (Cohort: {self.cohort_id})"
        else:
            title_str = "Train and Validation Performance"
        ax.set_title(title_str, fontdict={"fontsize": 16})

        return fig

problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("object_detection.train_and_validation_performance")
def calculate_train_and_validation_performance(data: dict, problem_type: str):
    """
    A wrapper function to calculate the training and validation performance.

    :param data: The input data containing predictions and ground truth
    :type data: dict
    :param problem_type: The type of problem
    :type problem_type: str

    :return: Dictionary with the calculated performance results
    :rtype: dict
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result


@plot_manager.register("object_detection.train_and_validation_performance")
def plot_train_and_validation_performance(
    results: dict,
    save_plot: bool,
    file_name: str = "train_and_validation_performance.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to Train and Validation Performance

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: boolean value to save plot
    :type save_plot: bool

    :return: None or path to the saved plot
    :rtype: Union[str, None]
    """
    plotter = PlotTrainAndValidationPerformance(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()