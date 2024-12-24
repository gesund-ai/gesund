from typing import Optional, Union
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager
from .iou import IoUCalc


class Classification:
    pass


class ObjectDetection:
    pass


class SemanticSegmentation:
    def __init__(self):
        self.iou = IoUCalc()

    def _validate_data(self, data: dict) -> bool:
        """
        A function to validate the given data used for calculating metrics for object detection validation

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict

        :return: a boolean value
        :rtype: bool
        """
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

    @staticmethod
    def convert_rle_to_boxes(rle_str: str, width: int, height: int) -> list:
        """
        A function to convert the RLE string to boxes

        :param rle_str: the RLE string
        :type rle_str: str
        :param width: the width of the image
        :type width: int
        :param height: the height of the image
        :type height: int

        :return: a list containing the boxes
        :rtype: list
        """
        rle_nums = list(map(int, rle_str.split()))
        rle_pairs = list(zip(rle_nums[::2], rle_nums[1::2]))

        mask = np.zeros(height * width, dtype=np.uint8)
        for start, length in rle_pairs:
            mask[start : start + length] = 1

        mask = mask.reshape((height, width)).T
        bboxes = []
        top_left, bottom_right = None, None

        # find bounding boxes
        for i in range(height):
            for j in range(width):
                if mask[i, j] == 1:
                    top_left = (i, j)
                    break

        for i in range(height - 1, -1, -1):
            for j in range(width - 1, -1, -1):
                if mask[i, j] == 1:
                    bottom_right = (i, j)
                    break

        if top_left and bottom_right:
            bboxes = [top_left[1], top_left[0], bottom_right[1], bottom_right[0]]
        return bboxes

    @staticmethod
    def convert_rle_to_mask(rle_str: str, width: int, height: int) -> np.ndarray:
        """
        A function to convert the RLE string to mask

        :param rle_str: the RLE string
        :type rle_str: str
        :param width: the width of the image
        :type width: int
        :param height: the height of the image
        :type height: int

        :return: a numpy array containing the mask
        :rtype: np.ndarray
        """
        rle_nums = list(map(int, rle_str.split()))
        rle_pairs = list(zip(rle_nums[::2], rle_nums[1::2]))

        mask = np.zeros(height * width, dtype=np.uint8)
        for start, length in rle_pairs:
            mask[start : start + length] = 1

        mask = mask.reshape((height, width)).T
        return mask

    def _preprocess(self, data: dict) -> tuple:
        """
        A function to preprocess the data

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict

        :return: a tuple containing the ground truth and prediction boxes
        :rtype: tuple
        """
        pred_data, gt_data = {}, {}

        for image_id in tqdm(data["ground_truth"], desc="Preprocessing data"):
            gt_masks = []
            for annotation in data["ground_truth"][image_id]["annotation"]:
                if annotation["type"] == "mask":
                    rle = annotation["mask"]["mask"]
                    width, height = annotation["shape"]
                    mask = self.convert_rle_to_mask(rle, width, height)
                    gt_masks.append(mask)
            gt_data[image_id] = gt_masks

            pred_masks = []
            width, height = data["prediction"][image_id]["shape"]
            for rle_mask in data["prediction"][image_id]["masks"]["rles"]:
                rle = rle_mask["rle"]
                mask = self.convert_rle_to_mask(rle, width, height)
                pred_masks.append(mask)
            pred_data[image_id] = pred_masks

        return (gt_data, pred_data)

    def _calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        """
        A function to calculate the metrics for object detection

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict
        :param class_mapping: a dictionary containing the class mapping
        :type class_mapping: dict

        :return: a dictionary containing the calculated metrics
        :rtype: dict
        """
        # preprocess the data
        gt_data, pred_data = self._preprocess(data)

        # calculate the metrics
        data = {"image_id": [], "iou": []}

        for image_id in gt_data:
            gt_masks = gt_data[image_id]
            if not gt_masks:
                continue
            for gt_mask in gt_masks:
                for pred_mask in pred_data[image_id]:
                    iou = self.iou.calculate_iou_mask(gt_mask, pred_mask)
                    data["image_id"].append(image_id)
                    data["iou"].append(iou)

        data = pd.DataFrame(data)
        data = data.dropna()
        return data

    def calculate(self, data: dict) -> dict:
        """
        A function to calculate the metrics for object detection

        :param data: a dictionary containing the ground truth and prediction data
        :type data: dict

        :return: a dictionary containing the calculated metrics
        :rtype: dict
        """
        result = {}

        # validate the data
        self._validate_data(data)

        # calculate the metrics
        result["iou_distribution"] = self._calculate_metrics(
            data, class_mapping=data.get("class_mapping", {})
        )

        return result


class PlotPredictedDistribution:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.cohort_id = cohort_id

    def _validation_data(self):
        """
        Validate the data
        """
        if "iou_distribution" not in self.data:
            raise ValueError("Missing iou_distribution in the data dictionary")

    def save(self, fig: Figure, filename: str) -> str:
        """
        A function to save the plot

        :param fig: the figure object
        :type fig: Figure
        :param filename: the name of the file
        :type filename: str

        :return: the filename
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
        A function to plot the predicted distribution

        :return: the figure object
        :rtype: Figure
        """
        # validate the data
        self._validation_data()

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_data = self.data["iou_distribution"]

        # plot violin plot
        sns.violinplot(x=plot_data["iou"], ax=ax)
        ax.set_xlabel("IoU", fontdict={"fontsize": 14, "fontweight": "medium"})

        if self.cohort_id:
            title_str = f"IoU distribution: cohort - {self.cohort_id}"
        else:
            title_str = "IoU distribution"

        ax.set_title(title_str, fontdict={"fontsize": 16, "fontweight": "bold"})
        return fig


problem_type_map = {
    "classification": Classification,
    "object_detection": ObjectDetection,
    "semantic_segmentation": SemanticSegmentation,
}


@metric_manager.register("semantic_segmentation.iou_distribution")
def calculate_iou_distribution(data: dict, problem_type: str) -> dict:
    """
    A function to calculate the IoU distribution

    :param data: a dictionary containing the ground truth and prediction data
    :type data: dict
    :param problem_type: the type of the problem
    :type problem_type: str

    :return: a dictionary containing the calculated metrics
    :rtype: dict
    """
    metric_calculator = problem_type_map[problem_type]()
    result = metric_calculator.calculate(data)
    return result


@plot_manager.register("semantic_segmentation.iou_distribution")
def plot_iou_dist_sem_seg(
    results: dict,
    save_plot: bool,
    file_name: str = "iou_distribution.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to plot the threshold

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: Boolean value to save the plot
    :type save_plot: bool
    :param file_name: the name of the file
    :type file_name: str
    :param cohort_id: the cohort id
    :type cohort_id: int

    :return: the file path
    :rtype: str
    """
    plotter = PlotPredictedDistribution(data=results)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
