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

class SemanticSegmentation:
    pass

class IoUCalc:
    def calculate(self, box1: List[float], box2: List[float]) -> float:
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
        self.iou = IoUCalc()
        self.label_to_class_name = self._load_class_mappings()

    def _load_class_mappings(self) -> Dict[int, str]:
        json_file = os.path.expanduser('~/gesund/tests/_data/object_detection/test_class_mappings.json')
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Class mappings file not found at {json_file}")
        
        with open(json_file, 'r') as f:
            class_mapping_original = json.load(f)
        
        return {int(k): v.lower().replace(" ", "_") for k, v in class_mapping_original.items()}

    def _validate_data(self, data: Dict) -> bool:
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
        if not gt_boxes:
            return 0.0, 0.0

        sorted_indices = np.argsort(confidences)[::-1]
        pred_boxes = [pred_boxes[i] for i in sorted_indices]
        if max_detections:
            pred_boxes = pred_boxes[:max_detections]

        matches = [
            max(self.iou.calculate(pred, gt) for gt in gt_boxes) >= threshold
            for pred in pred_boxes
        ]

        tp = np.cumsum(matches)
        fp = np.cumsum([not m for m in matches])
        recall = tp / len(gt_boxes)
        precision = tp / (tp + fp)

        recall_levels = np.linspace(0, 1, 11)
        ap = sum(np.max(precision[recall >= r]) if any(recall >= r) else 0 for r in recall_levels) / len(recall_levels)
        mean_recall = recall.mean() if len(recall) > 0 else 0.0
        return ap, mean_recall

    def _calc_metrics_per_class(self, data: Dict) -> Dict:
        gt_dict, pred_dict, conf_dict = self._preprocess_by_class(data)
        metrics = {}
        for cls, gt_boxes in gt_dict.items():
            pred_boxes = pred_dict.get(cls, [])
            confidences = conf_dict.get(cls, [])

            metrics[cls] = {
                "AP@10": self._mean_ap_ar(gt_boxes, pred_boxes, confidences, 0.5, 10)[0],
                "AP@50": self._mean_ap_ar(gt_boxes, pred_boxes, confidences, 0.5)[0],
                "AP@75": self._mean_ap_ar(gt_boxes, pred_boxes, confidences, 0.75)[0],
                "AP@[.50,.95]": np.mean([self._mean_ap_ar(gt_boxes, pred_boxes, confidences, thr)[0] for thr in np.arange(0.5, 1.0, 0.05)]),
                "AR@max=100": self._mean_ap_ar(gt_boxes, pred_boxes, confidences, 0.5, 100)[1],
                "AR@max=10": self._mean_ap_ar(gt_boxes, pred_boxes, confidences, 0.5, 10)[1],
                "AR@max=1": self._mean_ap_ar(gt_boxes, pred_boxes, confidences, 0.5, 1)[1],
            }
        return metrics

    def plot_metrics(self, metrics: Dict):
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
        self._validate_data(data)
        metrics = self._calc_metrics_per_class(data)
        return {"highlighted": metrics}


class PlotModelStats:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
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
            highlighted = self.data.get("highlighted", {})
            if not highlighted:
                raise ValueError("No highlighted metrics found.")

            rows = []
            for cls_name, metric_dict in highlighted.items():
                for metric, val in metric_dict.items():
                    rows.append({"Class": cls_name, "Metric": metric, "Value": val})

            df = pd.DataFrame(rows)
            df = df[df["Class"] != "unknown"]
            df_pivot = df.pivot(index="Class", columns="Metric", values="Value").fillna(0)

            fig, ax = plt.subplots(figsize=(15, 10))
            ax.axis('off')  

            table = ax.table(
                cellText=np.round(df_pivot.values, 2),
                rowLabels=df_pivot.index,
                colLabels=df_pivot.columns,
                cellLoc='center',
                loc='center'
            )
            
            for cell in table._cells.values():
                cell.set_facecolor('#404040')  
                cell.set_text_props(color='white')  
                
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)

            plt.tight_layout()
            return fig

problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}

@metric_manager.register("object_detection.model_stats")
def calculate_model_stats(data: dict, problem_type: str):
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result

@plot_manager.register("object_detection.model_stats")
def plot_model_stats(
    results: dict,
    save_plot: bool,
    file_name: str = "model_stats.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    
    plotter = PlotModelStats(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()