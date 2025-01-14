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
    def __init__(self, class_mappings: Optional[dict] = None):
        self.class_mappings = class_mappings or {}

    def _validate_data(self, data: dict) -> bool:
        """Validate input data structure."""
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
        """Decode RLE format mask."""
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
        """Process masks and return pixel counts per class."""
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
        """Calculate ground truth and prediction pixel counts."""
        try:
            counts = self._preprocess(data)
            return {
                "ground_truth_counts": counts["gt_counts"],
                "prediction_counts": counts["pred_counts"]
            }
        except Exception as e:
            raise ValueError(f"Error calculating metrics: {str(e)}")

    def calculate(self, data: dict) -> dict:
        """Validate data and calculate metrics."""
        self._validate_data(data)
        metric_manager.record_usage("semantic_segmentation.object_counts")
        return self._calculate_metrics(data)


class ObjectDetection:
    def __init__(self, class_mappings: Optional[dict] = None):
        self.class_mappings = class_mappings or {}
    def _validate_data(self, data: dict) -> bool:
        """Validate input data structure."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        if not data.get("ground_truth"):
            raise ValueError("Ground truth data is empty.")
        if not data.get("predictions"):
            raise ValueError("Prediction data is empty.")
        return True

    def _preprocess(self, data: dict) -> pd.DataFrame:
        """Preprocess ground truth and prediction data into a DataFrame."""
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
        """Calculate ground truth and prediction counts."""
        try:
            df = self._preprocess(data)
            return {
                "ground_truth_counts": df["gt_class_label"].value_counts().to_dict() if not df["gt_class_label"].empty else {},
                "prediction_counts": df["pred_class_label"].value_counts().to_dict() if not df["pred_class_label"].empty else {}
            }
        except Exception as e:
            raise ValueError(f"Error calculating object detection metrics: {str(e)}")

    def calculate(self, data: dict) -> dict:
        """Validate data and calculate metrics."""
        try:
            self._validate_data(data)
            metric_manager.record_usage("object_detection.object_counts")
            return self._calculate_metrics(data)
        except Exception as e:
            print(f"Debug: Error in calculate: {str(e)}")
            return {
                "ground_truth_counts": {},
                "prediction_counts": {}
            }

class PlotObjectCounts:
    def __init__(self, cohort_id: Optional[int] = None):
        self.cohort_id = cohort_id

    def _setup_plot(self, figsize: tuple = (12, 8)) -> tuple:
        """Set up the plot with a white background and grid."""
        plt.close('all')  
        plt.style.use('default')  
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')  
        ax.set_facecolor('white')
        return fig, ax

    def plot_object_counts(self, gt_data: dict, pred_data: dict) -> Figure:
        """Plot ground truth and prediction counts in descending order."""

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
        plot_manager.record_usage("object_detection.object_counts")
        return fig

    def save(self, fig: Figure, filename: str) -> str:
        """Save the plot to a file and clean up."""
        dir_path = "plots"
        os.makedirs(dir_path, exist_ok=True)
        filepath = f"{dir_path}/{self.cohort_id}_{filename}" if self.cohort_id else f"{dir_path}/{filename}"
        fig.savefig(filepath, format="png")
        plt.close(fig)  
        plot_manager.record_usage("object_detection.object_counts")
        return filepath


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}

@metric_manager.register("semantic_segmentation.object_counts")
@metric_manager.register("object_detection.object_counts") 
def calculate_object_count_metric(data: dict, problem_type: str, class_mappings: Optional[dict] = None) -> dict:
    """Calculate object count metrics for given problem type."""
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
    """Plot and optionally save object counts."""
    
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