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
        """
        Initializes the ObjectDetection metric calculator with an IoU calculator.
        """
        self.iou = IoUCalc()

    def _validate_data(self, data: dict) -> bool:
        """
        Validates the data required for metric calculation and plotting.

        :param data: The input data required for calculation, {"prediction":, "ground_truth": }
        :type data: dict

        :return: Status if the data is valid
        :rtype: bool

        :raises ValueError: If required keys are missing or keys do not match.
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
        Preprocesses the prediction and ground truth data for metric calculation.

        :param data: Dictionary containing 'prediction' and 'ground_truth' data.
        :type data: dict

        :return: Tuple of ground truth boxes and predicted boxes dictionaries.
        :rtype: tuple

        :raises ValueError: If preprocessing fails.
        """
        from .average_precision import ObjectDetection

        return ObjectDetection._preprocess(data, get_label=False, get_pred_scores=True)

    def _calculate_conf_dist(
        self, gt_boxes_dict, pred_boxes_dict: dict
    ) -> pd.DataFrame:
        """
        Calculates the confidence distribution for true positives and false positives.

        :param gt_boxes_dict: Dictionary of ground truth bounding boxes.
        :type gt_boxes_dict: dict
        :param pred_boxes_dict: Dictionary of predicted bounding boxes with scores.
        :type pred_boxes_dict: dict

        :return: DataFrame with confidence scores and labels.
        :rtype: pd.DataFrame

        :raises ValueError: If confidence distribution calculation fails.
        """
        results = []
        for image_id, pred_boxes in pred_boxes_dict.items():
            gt_boxes = gt_boxes_dict[image_id]
            matched_gt = [False] * len(gt_boxes)

            for pred_box in pred_boxes:
                score = pred_box[-1]
                pred_box = pred_box[:-1]
                best_iou = 0
                best_gt_idx = -1

                for idx, gt_box in enumerate(gt_boxes):
                    iou = self.iou.calculate(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > 0.5:
                    if not matched_gt[best_gt_idx]:
                        results.append(
                            {"confidence": score, "label": "TP", "best_iou": best_iou}
                        )
                        matched_gt[best_gt_idx] = True
                    else:
                        results.append(
                            {"confidence": score, "label": "FP", "best_iou": best_iou}
                        )
                else:
                    results.append(
                        {"confidence": score, "label": "FP", "best_iou": best_iou}
                    )

        data = pd.DataFrame(results)
        return data

    def _calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        """
        Calculates the confidence distribution histogram metrics.

        :param data: Dictionary containing 'prediction' and 'ground_truth' data.
        :type data: dict
        :param class_mapping: Dictionary mapping class IDs to class names.
        :type class_mapping: dict

        :return: Dictionary with confidence distribution histogram data.
        :rtype: dict

        :raises ValueError: If metric calculation fails.
        """
        results = {}

        # preprocess the data
        gt_boxes_dict, pred_boxes_dict = self._preprocess(data)

        # re-organize the confidence distribution
        result = self._calculate_conf_dist(gt_boxes_dict, pred_boxes_dict)
        results["confidence_distribution_histogram"] = result

        return results

    def calculate(self, data: dict) -> dict:
        """
        Calculates the confidence distribution histogram for the given data.

        :param data: The input data required for calculation and plotting, {"prediction":, "ground_truth": , "class_mapping":}
        :type data: dict

        :return: Calculated metric results
        :rtype: dict

        :raises ValueError: If data validation or calculation fails.
        """

        result = {}

        # validate the data
        self._validate_data(data)

        # calculate the metrics
        result = self._calculate_metrics(data, data.get("class_mapping"))

        return result


class PlotConfidenceDistributionHistogram:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        """
        Initializes the PlotConfidenceDistributionHistogram with data and an optional cohort identifier.

        :param data: Dictionary containing confidence distribution histogram data.
        :type data: dict
        :param cohort_id: Optional identifier for the cohort, defaults to None.
        :type cohort_id: Optional[int], optional
        """
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        Validates the data required for plotting the confidence distribution histogram.

        :raises ValueError: If required data keys are missing or data format is incorrect.
        """
        if "confidence_distribution_histogram" not in self.data:
            raise ValueError("confidence_distribution_histogram data is missing.")

        if not isinstance(self.data["confidence_distribution_histogram"], pd.DataFrame):
            raise ValueError(f"Data must be a data frame.")

    def save(self, fig: Figure, filename: str) -> str:
        """
        Saves a Matplotlib Figure object to a file.

        :param fig: A Matplotlib Figure object.
        :type fig: Figure
        :param filename: Name of the file to save the figure.
        :type filename: str

        :return: File path where the plot is saved.
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
        Plots the confidence distribution histogram.

        :return: Matplotlib Figure object with the confidence distribution histogram.
        :rtype: Figure

        :raises ValueError: If plotting fails.
        """
        self._validate_data()
        plot_data = self.data["confidence_distribution_histogram"]

        tp_data = plot_data[plot_data['label'] == 'TP']['confidence'].values
        fp_data = plot_data[plot_data['label'] == 'FP']['confidence'].values            
        bins = np.linspace(0, 1, 11)

        plt.style.use('default')
        sns.set_style("white")

        fig, ax = plt.subplots(figsize=(12, 7))

        n_tp, _, _ = ax.hist(tp_data, bins=bins, alpha=0.6, color='#aed6dc', 
                            edgecolor='#7c99b4', linewidth=1, label='True Positives')
        n_fp, _, _ = ax.hist(fp_data, bins=bins, alpha=0.6, color='#ff9a8c', 
                            edgecolor='#e88a7d', linewidth=1, label='False Positives')

        ax.set_title("Confidence Score Distribution", fontsize=16, pad=20, color='#2f4858')
        ax.set_xlabel("Confidence Score", fontsize=12, color='#2f4858')
        ax.set_ylabel("Count", fontsize=12, color='#2f4858')

        ax.grid(True, color='#e6e6e6', linestyle='-')
        ax.set_axisbelow(True)

        ax.tick_params(labelsize=10, colors='#2f4858')

        ax.set_xlim(-0.05, 1.05)

        for spine in ax.spines.values():
            spine.set_color('#e6e6e6')

        ax.legend()

        plt.tight_layout()
        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("object_detection.confidence_distribution_histogram")
def calculate_confidence_distribution_histogram(data: dict, problem_type: str):

    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result


@plot_manager.register("object_detection.confidence_distribution_histogram")
def plot_confidence_distribution_histogram_od(
    results: dict,
    save_plot: bool,
    file_name: str = "confidence_distribution_histogram.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    Plots the confidence distribution histogram for object detection.

    :param results: Dictionary containing confidence distribution histogram data.
    :type results: dict
    :param save_plot: Flag indicating whether to save the plot.
    :type save_plot: bool
    :param file_name: Name of the file to save the plot as, defaults to "confidence_distribution_histogram.png".
    :type file_name: str
    :param cohort_id: Optional identifier for the cohort, defaults to None.
    :type cohort_id: Optional[int], optional

    :return: File path of the saved plot if saved, otherwise None.
    :rtype: Union[str, None]

    :raises ValueError: If plotting fails.
    """
    plotter = PlotConfidenceDistributionHistogram(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
