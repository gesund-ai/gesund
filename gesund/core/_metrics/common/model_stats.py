from typing import Union, Optional, Dict, List, Tuple
import json
import os

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt

from gesund.core import metric_manager, plot_manager

class Classification:
    pass

#1.40 minute with from iou.py ->> normally with this class  20 second it takes.
class IoUCalc:
    def calculate(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.
        
        :param box1: List of coordinates [x1, y1, x2, y2] for the first box.
        :type box1: List[float]
        :param box2: List of coordinates [x1, y1, x2, y2] for the second box.
        :type box2: List[float]
        
        :return: IoU value in the range [0, 1].
        :rtype: float
        """
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])

        inter_width = xi2 - xi1
        inter_height = yi2 - yi1
        if inter_width <= 0 or inter_height <= 0:
            return 0.0
        inter_area = inter_width * inter_height

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


class ObjectDetection:
    def __init__(self):
        """
        Initializes the ObjectDetection metric calculator and loads class mappings.
        """
        self.iou = IoUCalc()
        self.label_to_class_name = self._load_class_mappings()

    def _load_class_mappings(self) -> Dict[int, str]:
        """
        Loads a JSON file mapping label IDs to class names.
        
        :return: Dictionary mapping label IDs to class names.
        :rtype: Dict[int, str]
        """
        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(__file__)
                    )
                )
            )
        )
        json_file = os.path.join(
            project_root,
            'tests',
            '_data',
            'object_detection' if isinstance(self, ObjectDetection) else 'semantic_segmentation',
            'test_class_mappings.json'
        )
        if not os.path.exists(json_file):
            fallback_file = os.path.join(project_root, 'tests', '_data', 'test_class_mappings_default.json')
            if not os.path.exists(fallback_file):
                raise FileNotFoundError(f"Neither {json_file} nor {fallback_file} could be found.")
            json_file = fallback_file

        with open(json_file, 'r') as f:
            class_mapping_original = json.load(f)
        return {
            int(k): v.lower().replace(" ", "_")
            for k, v in class_mapping_original.items()
        }

    def _validate_data(self, data: Dict) -> bool:
        """
        Validates the data required for metric calculation.
        
        :param data: Dictionary containing 'ground_truth', 'prediction', 'class_mapping', 'metric_args'.
        :type data: Dict
        
        :return: True if data is valid.
        :rtype: bool
        """
        required_keys = {"ground_truth", "prediction", "class_mapping", "metric_args"}
        missing_keys = required_keys - data.keys()
        if missing_keys:
            raise ValueError(f"Missing keys in data: {missing_keys}")

        gt_ids = set(data["ground_truth"].keys())
        pred_ids = set(data["prediction"].keys())
        if gt_ids != pred_ids:
            raise ValueError("Mismatch between ground truth and prediction sample IDs.")
        return True

    def _preprocess(self, data: Dict, get_label: bool = True, get_pred_scores: bool = True) -> Tuple[Dict, Dict]:
        """
        Preprocesses the input data to extract ground truth and prediction boxes.
        
        :param data: Dictionary containing 'ground_truth' and 'prediction' data.
        :type data: Dict
        :param get_label: Flag to include labels in the boxes, defaults to True.
        :type get_label: bool
        :param get_pred_scores: Flag to include prediction scores, defaults to True.
        :type get_pred_scores: bool
        
        :return: Tuple of dictionaries containing ground truth boxes and prediction boxes.
        :rtype: Tuple[Dict, Dict]
        """
        gt_boxes, pred_boxes = {}, {}
        for image_id in data["ground_truth"]:
            for annotation in data["ground_truth"][image_id].get("annotation", []):
                points = annotation["points"]
                box = [points[0]["x"], points[0]["y"], points[1]["x"], points[1]["y"]]
                if get_label:
                    label = self.label_to_class_name.get(annotation["label"], "unknown")
                    box.append(label)
                gt_boxes.setdefault(image_id, []).append(box)

            for pred in data["prediction"][image_id].get("objects", []):
                box = [pred["box"]["x1"], pred["box"]["y1"], pred["box"]["x2"], pred["box"]["y2"]]
                if get_label:
                    label = self.label_to_class_name.get(pred.get("prediction_class", -1), "unknown")
                    box.append(label)
                if get_pred_scores:
                    box.append(pred.get("confidence", 0.0))
                pred_boxes.setdefault(image_id, []).append(box)

        return gt_boxes, pred_boxes

    def _preprocess_by_class(self, data: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        Organizes ground truth and predictions by class for metric calculation.
        
        :param data: Dictionary containing preprocessed ground truth and prediction boxes.
        :type data: Dict
        
        :return: Tuple of dictionaries containing GT boxes, predicted boxes, and confidences by class.
        :rtype: Tuple[Dict, Dict, Dict]
        """
        gt_boxes, pred_boxes = self._preprocess(data, get_label=True, get_pred_scores=True)
        classes = list(self.label_to_class_name.values()) + ["unknown"]
        gt_by_class = {cls: [] for cls in classes}
        pred_by_class = {cls: [] for cls in classes}
        confidences_by_class = {cls: [] for cls in classes}

        for boxes in gt_boxes.values():
            for box in boxes:
                gt_by_class[box[-1]].append(box[:-1])

        for boxes in pred_boxes.values():
            for box in boxes:
                cls = box[-2]
                confidence = box[-1]
                pred_by_class[cls].append(box[:-2])
                confidences_by_class[cls].append(confidence)

        return gt_by_class, pred_by_class, confidences_by_class

    def _mean_ap_ar(
        self,
        gt_boxes: List[List[float]],
        pred_boxes: List[List[float]],
        confidences: List[float],
        threshold: float,
        max_detections: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Compute average precision and average recall for the given class at specified IoU threshold.
        
        :param gt_boxes: List of ground truth bounding boxes.
        :param pred_boxes: List of predicted bounding boxes.
        :param confidences: Prediction confidence scores.
        :param threshold: IoU threshold used for matching predictions to ground truth.
        :param max_detections: Max number of predictions to keep (for AR).
        :return: (AP, AR) tuple.
        """
        if not gt_boxes:
            return 0.0, 0.0

        sorted_indices = np.argsort(confidences)[::-1]
        pred_boxes = [pred_boxes[i] for i in sorted_indices]
        if max_detections:
            pred_boxes = pred_boxes[:max_detections]

        matches = [max(self.iou.calculate(pred, gt) for gt in gt_boxes) >= threshold for pred in pred_boxes]
        tp = np.cumsum(matches)
        fp = np.cumsum([not m for m in matches])

        recall = tp / len(gt_boxes)
        precision = tp / (tp + fp)

        recall_levels = np.linspace(0, 1, 11)
        ap_list = []
        for r in recall_levels:
            valid = recall >= r
            ap_list.append(np.max(precision[valid]) if any(valid) else 0)
        ap = sum(ap_list) / len(ap_list)
        mean_recall = recall.mean() if len(recall) > 0 else 0.0

        return ap, mean_recall

    def _calc_metrics_per_class(self, data: Dict) -> Dict:
        """
        Calculates metrics (AP/AR) per class.
        
        :param data: Dictionary containing ground truth and prediction data.
        :type data: Dict
        
        :return: Dictionary of metrics per class.
        :rtype: Dict
        """
        gt_dict, pred_dict, conf_dict = self._preprocess_by_class(data)
        metrics = {}
        for cls, gt_boxes in gt_dict.items():
            pred_boxes = pred_dict.get(cls, [])
            confidences = conf_dict.get(cls, [])

            metrics[cls] = {
                "AP@10": self._mean_ap_ar(gt_boxes, pred_boxes, confidences, 0.5, 10)[0],
                "AP@50": self._mean_ap_ar(gt_boxes, pred_boxes, confidences, 0.5)[0],
                "AP@75": self._mean_ap_ar(gt_boxes, pred_boxes, confidences, 0.75)[0],
                "AP@[.50,.95]": np.mean([
                    self._mean_ap_ar(gt_boxes, pred_boxes, confidences, thr)[0]
                    for thr in np.arange(0.5, 1.0, 0.05)
                ]),
                "AR@max=100": self._mean_ap_ar(gt_boxes, pred_boxes, confidences, 0.5, 100)[1],
                "AR@max=10": self._mean_ap_ar(gt_boxes, pred_boxes, confidences, 0.5, 10)[1],
                "AR@max=1": self._mean_ap_ar(gt_boxes, pred_boxes, confidences, 0.5, 1)[1],
            }
        return metrics

    def plot_metrics(self, metrics: Dict) -> None:
        """
        Generates and displays a bar plot of metrics by class.
        
        :param metrics: Dictionary containing per-class metrics.
        :type metrics: Dict
        """
        metrics_order = ["AP@10", "AP@50", "AP@75", "AP@[.50,.95]", "AR@max=100", "AR@max=10", "AR@max=1"]
        data = [
            {"Class": cls, **{metric: metrics[cls].get(metric, 0) for metric in metrics_order}}
            for cls in metrics
        ]
        df = pd.DataFrame(data)
        df_melted = df.melt(id_vars="Class", var_name="Metric", value_name="Value")

        plt.figure(figsize=(12, 8))
        sns.barplot(x="Metric", y="Value", hue="Class", data=df_melted, dodge=False)
        plt.title("Object Detection Metrics by Class")
        plt.legend(title="Class", loc="upper right")
        plt.savefig('metrics_plot.png')
        plt.show()

    def calculate(self, data: Dict) -> Dict:
        """
        Validates data and calculates object detection metrics.
        
        :param data: Dictionary containing 'prediction', 'ground_truth', 'class_mapping', 'metric_args'.
        :type data: Dict
        
        :return: Dictionary with calculated metrics.
        :rtype: Dict
        """
        self._validate_data(data)
        metrics = self._calc_metrics_per_class(data)
        return {"highlighted": metrics}


class SemanticSegmentation:
    """Performs semantic segmentation metric calculations."""
    def __init__(self) -> None:
        """
        Initializes the SemanticSegmentation metric calculator and loads class mappings.
        """
        self.label_to_class_name = self._load_class_mappings()

    def _load_class_mappings(self) -> Dict[int, str]:
        """
        Loads a JSON file mapping label IDs to class names.
        
        :return: Dictionary mapping label IDs to class names.
        :rtype: Dict[int, str]
        """        
        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(__file__)
                    )
                )
            )
        )
        json_file = os.path.join(
            project_root,
            'tests',
            '_data',
            'semantic_segmentation' if isinstance(self, SemanticSegmentation) else 'object_detection',
            'test_class_mappings.json'
        )
        if not os.path.exists(json_file):
            fallback_file = os.path.join(project_root, 'tests', '_data', 'test_class_mappings_default.json')
            if not os.path.exists(fallback_file):
                raise FileNotFoundError(f"Neither {json_file} nor {fallback_file} could be found.")
            json_file = fallback_file

        with open(json_file, 'r') as f:
            class_mapping_original = json.load(f)
        return {
            int(k): v.lower().replace(" ", "_")
            for k, v in class_mapping_original.items()
        }

    def _validate_data(self, data: dict) -> None:
        """
        Validates the data required for semantic segmentation metric calculation.
        
        :param data: Dictionary containing 'ground_truth', 'prediction', and 'metric_args'.
        :type data: dict
        
        :raises ValueError: If required keys are missing or data is malformed.
        """
        for key in ["ground_truth", "prediction", "metric_args"]:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        gt_ids = set(data["ground_truth"].keys())
        pred_ids = set(data["prediction"].keys())
        if gt_ids != pred_ids:
            raise ValueError(f"Image ID mismatch. GT: {gt_ids - pred_ids}, Pred: {pred_ids - gt_ids}")

        for img_id, gt_data in data["ground_truth"].items():
            if "annotation" not in gt_data:
                raise ValueError(f"Missing 'annotation' in GT for image {img_id}")
            for ann in gt_data["annotation"]:
                if "shape" not in ann:
                    raise ValueError(f"Missing shape in GT annotation for image {img_id}")
                if "mask" not in ann.get("mask", {}):
                    raise ValueError(f"Missing mask in GT annotation for image {img_id}")

        for img_id, pred_data in data["prediction"].items():
            if "shape" not in pred_data:
                raise ValueError(f"Missing shape in predictions for image {img_id}")
            if "masks" not in pred_data or "rles" not in pred_data["masks"]:
                raise ValueError(f"Missing masks/rles in predictions for image {img_id}")

    def _decode_rle(self, rle_str: str, height: int, width: int) -> np.ndarray:
        """
        Decode a run-length encoded string into a binary mask.
        
        :param rle_str: The run-length encoded mask.
        :param height: The height of the target image.
        :param width: The width of the target image.
        :return: 2D mask array (uint8).
        """
        if not rle_str:
            return np.zeros((height, width), dtype=np.uint8)

        rle_numbers = list(map(int, rle_str.split()))
        if len(rle_numbers) % 2 != 0:
            raise ValueError("Invalid RLE format: odd number of elements")

        mask = np.zeros(height * width, dtype=np.uint8)
        current_pos = 0
        for start, length in zip(rle_numbers[::2], rle_numbers[1::2]):
            start_idx = start - 1
            end_idx = start_idx + length
            if end_idx > height * width:
                raise ValueError(f"RLE exceeds image dimensions: {end_idx} > {height * width}")
            mask[start_idx:end_idx] = 1
            current_pos = end_idx

        return mask.reshape((height, width))

    def _preprocess(self, data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts annotations and predictions into numeric masks.
        
        :param data: Dictionary containing 'ground_truth' and 'prediction' data.
        :type data: dict
        
        :return: Tuple of numpy arrays for ground truth masks and predicted masks.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        gt_masks, pred_masks = [], []
        for img_id in data["ground_truth"]:
            gt_entry = data["ground_truth"][img_id]
            first_ann = gt_entry["annotation"][0]
            height = int(first_ann["shape"][0])
            width = int(first_ann["shape"][1])

            combined_gt_mask = np.full((height, width), 255, dtype=np.uint8) 
            for ann in gt_entry["annotation"]:
                ann_height = int(ann["shape"][0])
                ann_width = int(ann["shape"][1])
                if ann_height != height or ann_width != width:
                    raise ValueError(
                        f"Annotation shape mismatch in GT for image {img_id}"
                    )
                ann_mask = self._decode_rle(ann["mask"]["mask"], height, width)
                combined_gt_mask = np.where(
                    ann_mask, ann["label"], combined_gt_mask
                )

            pred_entry = data["prediction"][img_id]
            p_height = int(pred_entry["shape"][0])
            p_width = int(pred_entry["shape"][1])
            if p_height != height or p_width != width:
                raise ValueError(f"Prediction shape mismatch for image {img_id}")

            combined_pred_mask = np.full((p_height, p_width), 255, dtype=np.uint8)
            for rle_info in pred_entry["masks"]["rles"]:
                pred_mask = self._decode_rle(rle_info["rle"], p_height, p_width)
                combined_pred_mask = np.where(pred_mask, rle_info["class"], combined_pred_mask)

            gt_masks.append(combined_gt_mask)
            pred_masks.append(combined_pred_mask)

        return np.array(gt_masks), np.array(pred_masks)

    def calculate(self, data: Dict) -> Dict:
        """
        Validates data and calculates semantic segmentation metrics.
        
        :param data: Dictionary containing 'ground_truth', 'prediction', and 'metric_args'.
        :type data: Dict
        
        :return: Dictionary with calculated metrics.
        :rtype: Dict
        """
        self._validate_data(data)
        gt_masks, pred_masks = self._preprocess(data)
        metrics = {"per_class": {}}

        for class_id, class_name in self.label_to_class_name.items():
            if class_name == "background":
                continue
            tp = np.sum((gt_masks == class_id) & (pred_masks == class_id))
            fp = np.sum((gt_masks != class_id) & (pred_masks == class_id))
            fn = np.sum((gt_masks == class_id) & (pred_masks != class_id))
            union = tp + fp + fn
            iou = tp / union if union > 0 else 0.0

            metrics["per_class"][class_name] = {
                "AP@10": iou if iou >= 0.1 else 0.0,
                "AP@50": iou if iou >= 0.5 else 0.0,
                "AP@75": iou if iou >= 0.75 else 0.0,
                "AP@[.50,.95]": np.mean([
                    1.0 if iou >= thr else 0.0
                    for thr in np.arange(0.5, 1.0, 0.05)
                ]),
                "AR@max=100": iou,
                "AR@max=10": iou,
                "AR@max=1": iou,
            }
        return {"highlighted": metrics}

    def plot_metrics(self, metrics: Dict) -> Figure:
        """
        Generates a bar plot for semantic segmentation metrics per class.
        
        :param metrics: Dictionary containing per-class metrics.
        :type metrics: Dict
        
        :return: Matplotlib Figure object with the bar plot.
        :rtype: Figure
        """
        per_class = {k: v for k, v in metrics.items() if k != "overall"}
        rows = []
        for cls, cls_metrics in per_class.items():
            rows.append({"Class": cls, **cls_metrics})

        df = pd.DataFrame(rows)
        plt.figure(figsize=(14, 8), facecolor='white')
        df_melted = df.melt(id_vars="Class", var_name="Metric", value_name="Value")
        sns.barplot(x="Metric", y="Value", hue="Class", data=df_melted, palette="viridis")
        plt.ylim(0, 1)
        plt.title("Segmentation Metrics (AP/AR Style)")
        plt.tight_layout()
        return plt.gcf()


class PlotModelStats:
    """
    Class to generate and save plots for model metrics results.
    Supports separate table-like views for semantic segmentation.
    """
    def __init__(self, data: dict, cohort_id: Optional[int] = None) -> None:
        """
        Initializes the PlotModelStats with data and optional cohort identifier.
        
        :param data: Dictionary containing metric results.
        :type data: dict
        :param cohort_id: Optional identifier for the cohort, defaults to None.
        :type cohort_id: Optional[int], optional
        """
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self) -> None:
        """
        Validates that the data contains a DataFrame in 'result'.
        
        :raises ValueError: If 'result' is not a DataFrame.
        """
        if not isinstance(self.data["result"], pd.DataFrame):
            raise ValueError("Data must be a DataFrame.")

    def save(self, fig: Figure, filename: str) -> str:
        """
        Save the matplotlib figure to a local directory.
        
        :param fig: A Matplotlib Figure object.
        :param filename: Name of the file to save.
        :return: The file path of the saved plot.
        """
        dir_path = "plots"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if self.cohort_id:
            filepath = os.path.join(dir_path, f"{self.cohort_id}_{filename}")
        else:
            filepath = os.path.join(dir_path, filename)

        fig.savefig(filepath, bbox_inches='tight', format="png")
        return filepath

    def plot(self) -> Figure:
        """
        Determines the appropriate plotting method based on the data structure.
        
        :return: Matplotlib Figure object with the generated plot.
        :rtype: Figure
        
        :raises ValueError: If no highlighted metrics are found.
        """
        highlighted = self.data.get("highlighted", {})
        if not highlighted:
            raise ValueError("No highlighted metrics found.")
        if "overall" in highlighted:
            return self._plot_segmentation_table(highlighted)
        return self._plot_generic_table(highlighted)

    def _plot_segmentation_table(self, metrics: Dict) -> Figure:
        """
        Creates a table showing semantic segmentation metrics.
        
        :param metrics: Dictionary containing the metric results.
        :type metrics: Dict
        
        :return: Matplotlib Figure with a table of segmentation metrics.
        :rtype: Figure
        """
        metrics_order = ["AP@10", "AP@50", "AP@75", "AP@[.50,.95]", "AR@max=100", "AR@max=10", "AR@max=1"]
        data = []
        for cls_name, values in metrics["per_class"].items():
            if cls_name == "background":
                continue
            for m in metrics_order:
                data.append({"Metric": m, "Class": cls_name, "Value": values.get(m, 0)})

        df = pd.DataFrame(data)
        df_pivot = df.pivot(index="Metric", columns="Class", values="Value").fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        ax.axis('off')
        table = ax.table(
            cellText=df_pivot.round(2).values,
            rowLabels=df_pivot.index,
            colLabels=df_pivot.columns,
            cellLoc='center',
            loc='center'
        )
        for cell in table.get_celld().values():
            cell.set_facecolor("white")
            cell.set_edgecolor("black")

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        plt.title("Semantic Segmentation Metrics by Class (Table View)", pad=20)
        plt.tight_layout()
        return fig

    def _plot_generic_table(self, metrics: Dict) -> Figure:
        """
        Creates a generic table for displaying metrics, used for object detection or as a fallback.
        
        :param metrics: Dictionary containing the metric information.
        :type metrics: Dict
        
        :return: Matplotlib Figure object with a table of metrics.
        :rtype: Figure
        """
        if "per_class" in metrics:
            per_class_data = metrics["per_class"]
            rows = []
            for cls_name, cls_metrics in per_class_data.items():
                if cls_name == "background":
                    continue
                for metric_name, val in cls_metrics.items():
                    rows.append({"Class": cls_name, "Metric": metric_name, "Value": val})

            df = pd.DataFrame(rows)
            df_pivot = df.pivot(index="Metric", columns="Class", values="Value").fillna(0)

        else:
            rows = []
            for cls_name, metric_dict in metrics.items():
                for metric, val in metric_dict.items():
                    rows.append({"Class": cls_name, "Metric": metric, "Value": val})
            df = pd.DataFrame(rows)
            df_pivot = df.pivot(index="Class", columns="Metric", values="Value").fillna(0)

        df_rounded = df_pivot.round(2)
        fig, ax = plt.subplots(figsize=(15, 10), facecolor='white')
        ax.axis('off')
        table = ax.table(
            cellText=df_rounded.values,
            rowLabels=df_rounded.index,
            colLabels=df_rounded.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        plt.title("Model Metrics")
        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}

@metric_manager.register("semantic_segmentation.model_stats")
@metric_manager.register("object_detection.model_stats")
def calculate_model_stats(data: dict, problem_type: str):
    """
    Calculates model statistics based on the problem type.
    
    :param data: Dictionary containing prediction and ground truth data.
    :type data: dict
    :param problem_type: Type of the problem 
    :type problem_type: str
    
    :return: Calculated metric results.
    :rtype: dict
    """
    metric_calculator = problem_type_map[problem_type]()
    result = metric_calculator.calculate(data)
    return result

@plot_manager.register("object_detection.model_stats")
def plot_model_stats(
    results: dict,
    save_plot: bool,
    file_name: str = "model_stats.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    Plots object detection metrics, optionally saving the plot to disk.
    
    :param results: Dictionary containing metric results.
    :type results: dict
    :param save_plot: Flag indicating whether to save the plot.
    :type save_plot: bool
    :param file_name: Name of the file to save the plot as, defaults to "model_stats.png".
    :type file_name: str
    :param cohort_id: Optional identifier for the cohort, defaults to None.
    :type cohort_id: Optional[int], optional
    
    :return: File path of the saved plot if saved, otherwise None.
    :rtype: Union[str, None]
    """
    plotter = PlotModelStats(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
        
@plot_manager.register("semantic_segmentation.model_stats")
def plot_model_stats_semantic_segmentation(
    results: dict,
    save_plot: bool,
    file_name: str = "model_stats.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    Plots semantic segmentation metrics, optionally saving the plot to disk.
    
    :param results: Dictionary containing metric results.
    :type results: dict
    :param save_plot: Flag indicating whether to save the plot.
    :type save_plot: bool
    :param file_name: Name of the file to save the plot as, defaults to "model_stats.png".
    :type file_name: str
    :param cohort_id: Optional identifier for the cohort, defaults to None.
    :type cohort_id: Optional[int], optional
    
    :return: File path of the saved plot if saved, otherwise None.
    :rtype: Union[str, None]
    """
    plotter = PlotModelStats(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()