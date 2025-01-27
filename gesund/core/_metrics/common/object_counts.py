from typing import Union, Optional
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager

class Classification:
    pass

class SemanticSegmentation:
    """
    Performs semantic segmentation metric calculations.
    """
    def __init__(self, class_mappings: Optional[dict] = None):
        """
        Initializes the SemanticSegmentation metric calculator with optional class mappings.

        :param class_mappings: Dictionary mapping class IDs to class names, defaults to None.
        :type class_mappings: Optional[dict], optional
        """
        self.class_mappings = class_mappings or {}

    def _validate_data(self, data: dict) -> bool:
        """
        Validates the data required for semantic segmentation metric calculation.

        :param data: The input data required for calculation, {"ground_truth":, "predictions":}
        :type data: dict

        :return: Status indicating if the data is valid.
        :rtype: bool

        :raises ValueError: If data is not a dictionary or missing required keys, or if ground truth or prediction data is empty.
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        if "ground_truth" not in data:
            raise ValueError("Missing 'ground_truth' key.")
        if "predictions" not in data:
            raise ValueError("Missing 'predictions' key.")
        if not data["ground_truth"]:
            raise ValueError("Ground truth data is empty.")
        if not data["predictions"]:
            raise ValueError("Prediction data is empty.")
        return True

    def _decode_rle(self, encoded_mask: str, shape: tuple) -> np.ndarray:
        """
        Decodes a Run-Length Encoded (RLE) mask into a binary mask array.

        :param encoded_mask: The RLE encoded mask string.
        :type encoded_mask: str
        :param shape: Shape of the mask as (height, width).
        :type shape: tuple

        :return: Decoded binary mask as a NumPy array.
        :rtype: np.ndarray

        :raises ValueError: If the encoded mask cannot be decoded properly.
        """
        if not encoded_mask:
            return np.zeros(shape, dtype=np.uint8)
        
        numbers = [int(x) for x in encoded_mask.split()]
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        
        for i in range(0, len(numbers), 2):
            start = numbers[i]
            length = numbers[i + 1]
            mask[start:start + length] = 1
        
        return mask.reshape(shape)

    def _preprocess(self, data: dict) -> dict:
        """
        Processes ground truth and prediction masks to calculate pixel counts per class.

        :param data: Dictionary containing 'ground_truth' and 'predictions' data.
        :type data: dict

        :return: Dictionary with ground truth and prediction pixel counts per class.
        :rtype: dict

        :raises ValueError: If there is an error during preprocessing.
        """
        try:
            results = {"gt_counts": {}, "pred_counts": {}}
            class_mappings = {str(k): v for k, v in self.class_mappings.items()}

            # Process ground truth
            for image_id in data["ground_truth"]:
                if "annotation" not in data["ground_truth"][image_id]:
                    continue
                    
                annotations = data["ground_truth"][image_id]["annotation"]
                if not annotations:
                    continue
                    
                image_shape = annotations[0].get("shape")
                if not image_shape:
                    continue

                for annotation in annotations:
                    if annotation.get("type") == "mask":
                        label_id = str(annotation.get("label", ""))
                        label_name = class_mappings.get(label_id, "Unknown")
                        
                        mask = self._decode_rle(annotation.get("mask", {}).get("mask", ""), image_shape)
                        pixel_count = np.sum(mask)
                        
                        results["gt_counts"][label_name] = results["gt_counts"].get(label_name, 0) + pixel_count

            # Process predictions
            for image_id in data["predictions"]:
                if "masks" not in data["predictions"][image_id]:
                    continue
                    
                shape = data["predictions"][image_id].get("shape")
                if not shape:
                    continue
                    
                for mask_info in data["predictions"][image_id]["masks"].get("rles", []):
                    label_id = str(mask_info.get("class", ""))
                    label_name = class_mappings.get(label_id, "Unknown")
                    
                    mask = self._decode_rle(mask_info.get("rle", ""), shape)
                    pixel_count = np.sum(mask)
                    
                    results["pred_counts"][label_name] = results["pred_counts"].get(label_name, 0) + pixel_count

            return results
        except Exception as e:
            raise ValueError(f"Error preprocessing data: {str(e)}")

    def _calculate_metrics(self, data: dict) -> dict:
        """
        Calculates ground truth and prediction pixel counts.

        :param data: Dictionary containing preprocessed data.
        :type data: dict

        :return: Dictionary with ground truth and prediction pixel counts.
        :rtype: dict

        :raises ValueError: If there is an error during metric calculation.
        """
        try:
            counts = self._preprocess(data)
            return {
                "ground_truth_counts": counts["gt_counts"],
                "prediction_counts": counts["pred_counts"]
            }
        except Exception as e:
            raise ValueError(f"Error calculating metrics: {str(e)}")

    def calculate(self, data: dict) -> dict:
        """
        Validates data and calculates semantic segmentation metrics.

        :param data: The input data required for calculation, {"ground_truth":, "predictions":}
        :type data: dict

        :return: Calculated metric results including ground truth and prediction counts.
        :rtype: dict

        :raises ValueError: If data validation fails.
        """
        self._validate_data(data)
        return self._calculate_metrics(data)


class ObjectDetection:
    """
    Performs object detection metric calculations.
    """
    def __init__(self, class_mappings: Optional[dict] = None):
        """
        Initializes the ObjectDetection metric calculator with optional class mappings.

        :param class_mappings: Dictionary mapping class IDs to class names, defaults to None.
        :type class_mappings: Optional[dict], optional
        """
        self.class_mappings = class_mappings or {}
    def _validate_data(self, data: dict) -> bool:
        """
        Validates the data required for object detection metric calculation.

        :param data: The input data required for calculation, {"ground_truth":, "predictions":}
        :type data: dict

        :return: Status indicating if the data is valid.
        :rtype: bool

        :raises ValueError: If data is not a dictionary or missing required keys, or if ground truth or prediction data is empty.
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        if not data.get("ground_truth"):
            raise ValueError("Ground truth data is empty.")
        if not data.get("predictions"):
            raise ValueError("Prediction data is empty.")
        return True

    def _preprocess(self, data: dict) -> pd.DataFrame:
        """
        Preprocesses ground truth and prediction data into a DataFrame containing class labels.

        :param data: Dictionary containing 'ground_truth' and 'predictions' data.
        :type data: dict

        :return: DataFrame with ground truth and prediction class labels.
        :rtype: pd.DataFrame

        :raises ValueError: If there is an error during preprocessing.
        """
        results = {"gt_class_label": [], "pred_class_label": []}
        class_mappings = {str(k): v for k, v in self.class_mappings.items()}

        # Process ground truth
        if "ground_truth" in data:
            for image_id in data["ground_truth"]:
                for annotation in data["ground_truth"][image_id].get("annotation", []):
                    if annotation.get("type") == "rect" and "label" in annotation:
                        label_id = str(annotation.get("label"))
                        label_name = class_mappings.get(label_id, "Unknown")
                        results["gt_class_label"].append(label_name)

        # Process predictions
        if "predictions" in data:
            for image_id in data["predictions"]:
                for prediction in data["predictions"][image_id].get("objects", []):
                    if "prediction_class" in prediction:
                        label_id = str(prediction["prediction_class"])
                        label_name = class_mappings.get(label_id, "Unknown")
                        results["pred_class_label"].append(label_name)

        return pd.DataFrame(results)

    def _calculate_metrics(self, data: dict) -> dict:
        """
        Calculates ground truth and prediction counts for object detection.

        :param data: Dictionary containing preprocessed data.
        :type data: dict

        :return: Dictionary with ground truth and prediction counts.
        :rtype: dict

        :raises ValueError: If there is an error during metric calculation.
        """
        try:
            df = self._preprocess(data)
            return {
                "ground_truth_counts": df["gt_class_label"].value_counts().to_dict() if not df["gt_class_label"].empty else {},
                "prediction_counts": df["pred_class_label"].value_counts().to_dict() if not df["pred_class_label"].empty else {}
            }
        except Exception as e:
            raise ValueError(f"Error calculating object detection metrics: {str(e)}")

    def calculate(self, data: dict) -> dict:
        """
        Validates data and calculates object detection metrics.

        :param data: The input data required for calculation, {"ground_truth":, "predictions":}
        :type data: dict

        :return: Calculated metric results including ground truth and prediction counts.
        :rtype: dict
        """
        try:
            self._validate_data(data)
            return self._calculate_metrics(data)
        except Exception as e:
            print(f"Debug: Error in calculate: {str(e)}")
            return {
                "ground_truth_counts": {},
                "prediction_counts": {}
            }

class PlotObjectCounts:
    """
    Class to generate and save plots for object counts in ground truth and predictions.
    """
    def __init__(self, cohort_id: Optional[int] = None):
        """
        Initializes the PlotObjectCounts with an optional cohort identifier.

        :param cohort_id: Optional identifier for the cohort, defaults to None.
        :type cohort_id: Optional[int], optional
        """
        self.cohort_id = cohort_id

    def _setup_plot(self, figsize: tuple = (12, 8)) -> tuple:
        """
        Sets up the matplotlib plot with predefined styles.

        :param figsize: Size of the figure, defaults to (12, 8).
        :type figsize: tuple, optional

        :return: Tuple containing the figure and axes objects.
        :rtype: tuple
        """
        plt.close('all')  
        plt.style.use('default')  
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')  
        ax.set_facecolor('white')
        return fig, ax

    def plot_object_counts(self, gt_data: dict, pred_data: dict) -> Figure:
        """
        Plots ground truth and prediction object counts in descending order.

        :param gt_data: Dictionary containing ground truth counts per class.
        :type gt_data: dict
        :param pred_data: Dictionary containing prediction counts per class.
        :type pred_data: dict

        :return: Matplotlib Figure object with the object counts plot.
        :rtype: Figure

        :raises ValueError: If there is an error during plotting.
        """

        labels = sorted(set(gt_data.keys()).union(pred_data.keys()))
        total_counts = {
            label: gt_data.get(label, 0) + pred_data.get(label, 0)
            for label in labels
        }
        sorted_labels = sorted(labels, key=lambda x: total_counts[x], reverse=True)

        gt_counts = [gt_data.get(label, 0) for label in sorted_labels]
        pred_counts = [pred_data.get(label, 0) for label in sorted_labels]

        x = np.arange(len(sorted_labels))
        width = 0.35

        fig, ax = self._setup_plot()
        ax.bar(x - width / 2, gt_counts, width, label='Ground Truth', color="skyblue", edgecolor="white")
        ax.bar(x + width / 2, pred_counts, width, label='Prediction', color="salmon", edgecolor="white")

        ax.set_title("Object Counts")
        ax.set_xlabel("Class Labels")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_labels, rotation=45, ha='right')
        ax.legend()

        fig.tight_layout()
        return fig

    def save(self, fig: Figure, filename: str) -> str:
        """
        Saves the matplotlib figure to a local directory.

        :param fig: A Matplotlib Figure object.
        :type fig: Figure
        :param filename: Name of the file to save the figure.
        :type filename: str

        :return: File path where the plot is saved.
        :rtype: str
        """
        dir_path = "plots"
        os.makedirs(dir_path, exist_ok=True)
        filepath = f"{dir_path}/{self.cohort_id}_{filename}" if self.cohort_id else f"{dir_path}/{filename}"
        fig.savefig(filepath, format="png")
        plt.close(fig)  
        return filepath


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}

@metric_manager.register("semantic_segmentation.object_counts")
@metric_manager.register("object_detection.object_counts") 
def calculate_object_count_metric(data: dict, problem_type: str, class_mappings: Optional[dict] = None) -> dict:
    """
    Calculates object count metrics based on the problem type.

    :param data: Dictionary containing 'ground_truth' and 'predictions' data.
    :type data: dict
    :param problem_type: Type of the problem (e.g., classification, semantic_segmentation, object_detection).
    :type problem_type: str
    :param class_mappings: Optional dictionary mapping class IDs to class names, defaults to None.
    :type class_mappings: Optional[dict], optional

    :return: Dictionary with ground truth and prediction object counts.
    :rtype: dict
    """
    try:
        if "prediction" in data and "predictions" not in data:
            data["predictions"] = data["prediction"]

        mappings = class_mappings or data.get("class_mapping", {})
        processed_data = {
            "ground_truth": data.get("ground_truth", {}),
            "predictions": data.get("predictions", {}),
            "class_mapping": mappings
        }
                
        calculator_class = problem_type_map[problem_type]
        metric_calculator = calculator_class(class_mappings=mappings)
        result = metric_calculator.calculate(processed_data)
        return result

    except Exception as e:
        print(f"Debug: Error in calculate_object_count_metric: {str(e)}")
        return {
            "ground_truth_counts": {},
            "prediction_counts": {}
        }

@plot_manager.register("semantic_segmentation.object_counts")
@plot_manager.register("object_detection.object_counts")
def plot_object_count(
    results: dict,
    save_plot: bool,
    file_name: str = "object_counts.png",
    cohort_id: Optional[int] = None,
    problem_type: str = 'object_detection'
) -> Union[dict, None]:
    """
    Plots and optionally saves object counts for ground truth and predictions.

    :param results: Dictionary containing ground truth and prediction counts.
    :type results: dict
    :param save_plot: Flag indicating whether to save the plot.
    :type save_plot: bool
    :param file_name: Name of the file to save the plot as, defaults to "object_counts.png".
    :type file_name: str
    :param cohort_id: Optional identifier for the cohort, defaults to None.
    :type cohort_id: Optional[int], optional
    :param problem_type: Type of the problem, defaults to 'object_detection'.
    :type problem_type: str, optional

    :return: File path of the saved plot if saved, otherwise None.
    :rtype: Union[dict, None]

    :raises ValueError: If there is an error during plotting.
    """

    gt_counts = results.get("ground_truth_counts", {})
    pred_counts = results.get("prediction_counts", {})
    
    plotter = PlotObjectCounts(cohort_id=cohort_id)
    fig = plotter.plot_object_counts(gt_counts, pred_counts)
    
    if save_plot:
        return plotter.save(fig, file_name)
    else:
        plt.show()
        plt.close(fig)
        return None