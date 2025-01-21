from typing import Union, Optional
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gesund.core import metric_manager, plot_manager
from .iou import IoUCalc


class Classification:
    pass

class ObjectDetection:
    pass

class IoUCalc:
    def calculate(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        intersection = np.logical_and(pred_mask, gt_mask)
        union = np.logical_or(pred_mask, gt_mask)
        iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
        return iou_score


class SemanticSegmentation:
    def __init__(self):
        self.iou = IoUCalc()

    def _validate_data(self, data: dict) -> None:
        check_keys = ("ground_truth", "prediction", "metric_args")
        for _key in check_keys:
            if _key not in data:
                raise ValueError(f"Missing {_key} in the data dictionary")
        common_ids = set(data["prediction"].keys()).difference(set(data["ground_truth"].keys()))
        if common_ids:
            raise ValueError(f"Prediction and ground truth mismatch for image IDs: {common_ids}")
        for image_id in data["ground_truth"]:
            if ("shape" not in data["ground_truth"][image_id] 
                and "shape" not in data["prediction"][image_id]):
                raise ValueError(f"Missing 'shape' key for image {image_id}")

    def _decode_rle(self, rle_str: str, height: int, width: int) -> np.ndarray:
        if rle_str == "":
            return np.zeros((height, width), dtype=np.uint8)
        rle_numbers = list(map(int, rle_str.split()))
        mask = np.zeros(height * width, dtype=np.uint8)
        for i in range(0, len(rle_numbers), 2):
            start = rle_numbers[i] - 1
            length = rle_numbers[i + 1]
            mask[start:start + length] = 1
        return mask.reshape((height, width))

    def _preprocess(self, data: dict) -> pd.DataFrame:
        results = {
            "image_id": [],
            "height": [],
            "width": [],
            "rle": [],
            "gt_rle": [],
            "confidence": []
        }

        for image_id in data["ground_truth"]:
            try:
                shape = data["ground_truth"][image_id].get("shape") or data["prediction"][image_id]["shape"]
                height, width = shape[0], shape[1]
            except (KeyError, TypeError) as e:
                raise ValueError(f"Invalid shape data for image {image_id}: {e}")

            gt_rles = []
            for ann in data["ground_truth"][image_id]["annotation"]:
                if ann["type"] == "mask":
                    rle = ann["mask"]["mask"]
                    gt_rles.append(rle)

            # Process pred's
            preds = data["prediction"][image_id].get("masks", {}).get("rles", [])
            if not preds:
                preds = []

            for pred in preds:
                rle = pred.get("rle", "")
                confidence = pred.get("confidence", 1.0)
                if rle == "":
                    continue
                results["image_id"].append(image_id)
                results["height"].append(height)
                results["width"].append(width)
                results["rle"].append(rle)
                results["gt_rle"].append(np.nan)  
                results["confidence"].append(confidence)

            for gt_rle in gt_rles:
                results["image_id"].append(image_id)
                results["height"].append(height)
                results["width"].append(width)
                results["rle"].append(np.nan)  # No  mask
                results["gt_rle"].append(gt_rle)
                results["confidence"].append(0.0)  

        return pd.DataFrame(results)

    def _calc_overall_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall mean Average Precision and mean Average Recall metrics."""
        results = {"metric": [], "value": []}

        thresholds = [0.10, 0.50, 0.75, 0.95]
        max_dets = [1, 10, 100]

        metrics = {
            "mAP@10": [],
            "mAP@50": [],
            "mAP@75": [],
            "mAP@[50,95]": [],
            "mAR@max=1": [],
            "mAR@max=10": [],
            "mAR@max=100": []
        }

        image_ids = data["image_id"].unique()

        for image_id in image_ids:
            image_data = data[data["image_id"] == image_id]
            height = image_data["height"].iloc[0]
            width = image_data["width"].iloc[0]

            preds = image_data[image_data["rle"].notna()]
            pred_rles = preds["rle"].values
            confidences = preds["confidence"].values

            gts = image_data[image_data["gt_rle"].notna()]
            gt_rles = gts["gt_rle"].values

            pred_masks = [self._decode_rle(rle, height, width) for rle in pred_rles]
            gt_masks = [self._decode_rle(rle, height, width) for rle in gt_rles]

            sorted_indices = np.argsort(confidences)[::-1]
            pred_masks_sorted = [pred_masks[i] for i in sorted_indices]

            matched_gts = set()
            iou_scores = []
            for pred_mask in pred_masks_sorted:
                max_iou = 0.0
                for gt_idx, gt_mask in enumerate(gt_masks):
                    if gt_idx in matched_gts:
                        continue
                    iou = self.iou.calculate(pred_mask, gt_mask)
                    if iou > max_iou:
                        max_iou = iou
                        best_gt = gt_idx
                if max_iou >= 0.5:
                    matched_gts.add(best_gt)
                iou_scores.append(max_iou)

            for thresh in thresholds:
                tp = sum(iou >= thresh for iou in iou_scores)
                fp = len(iou_scores) - tp
                fn = len(gt_masks) - tp

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                if thresh == 0.10:
                    metrics["mAP@10"].append(precision)
                elif thresh == 0.50:
                    metrics["mAP@50"].append(precision)
                elif thresh == 0.75:
                    metrics["mAP@75"].append(precision)
                elif thresh == 0.95:
                    metrics["mAP@[50,95]"].append(precision)

            for n in max_dets:
                top_iou = iou_scores[:n]
                tp = sum(iou >= 0.50 for iou in top_iou)
                fn = len(gt_masks) - tp

                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics[f"mAR@max={n}"].append(recall)

        for metric, values in metrics.items():
            avg_value = np.round(np.mean(values) if values else 0, decimals=3)
            results["metric"].append(metric)
            results["value"].append(avg_value)

        return pd.DataFrame(results)

    def __calculate_metrics(self, data: dict) -> pd.DataFrame:
        self._validate_data(data)
        preprocessed_data = self._preprocess(data)
        return self._calc_overall_metrics(preprocessed_data)

    def calculate(self, data: dict) -> dict:
        try:
            result = self.__calculate_metrics(data)
            return {"result": result}
        except Exception as e:
            raise RuntimeError(f"Error in calculating metrics: {e}")


class PlotOverallMetric:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self):
        if not isinstance(self.data["result"], pd.DataFrame):
            raise ValueError("Data must be a DataFrame.")

    def save(self, fig: plt.Figure, filename: str) -> str:
        dir_path = "plots"
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"{self.cohort_id}_{filename}" if self.cohort_id else filename)
        fig.savefig(filepath, format="png")
        return filepath

    def plot(self) -> plt.Figure:
            plt.style.use('dark_background')
            sns.set_style("darkgrid")
            self._validate_data()

            metrics = self.data["result"]
            metrics_filtered = metrics[metrics['metric'].str.contains('mAP|mAR')]
            metrics_sorted = metrics_filtered.sort_values(by='metric', 
                key=lambda x: pd.Categorical(x, 
                    ['mAP@10', 'mAP@50', 'mAP@75', 'mAP@[50,95]',
                    'mAR@max=1', 'mAR@max=10', 'mAR@max=100']))
            
            values_only = metrics_sorted['value'].round(3).values
            metrics_only = metrics_sorted['metric'].values
            display_data = np.column_stack([metrics_only, values_only])


            fig, ax = plt.subplots(figsize=(8, len(metrics_sorted) * 0.6 + 2))
            fig.patch.set_facecolor('#2E2E2E')
            ax.set_facecolor('#2E2E2E')
            ax.axis('off')


            table = ax.table(cellText=display_data,
                            cellLoc='center',
                            loc='center')


            cell_color = '#363636'
            text_color = 'white'

            for _, cell in table._cells.items():
                cell.set_facecolor(cell_color)
                cell.set_text_props(color=text_color)
                cell.set_edgecolor('gray')

            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)

            title = f"Overall Metrics: Cohort - {self.cohort_id}" if self.cohort_id else "Overall Metrics"
            ax.set_title(title, fontsize=16, pad=20, color='white')

            return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}

@metric_manager.register("semantic_segmentation.overall_metric")
def calculate_overall_metric(data: dict, problem_type: str):
    return problem_type_map[problem_type]().calculate(data)

@plot_manager.register("semantic_segmentation.overall_metric")
def plot_overall_metric(results: dict, save_plot: bool, file_name: str = "overall_metric.png", cohort_id: Optional[int] = None) -> Union[str, None]:
    plotter = PlotOverallMetric(results, cohort_id)
    fig = plotter.plot()
    return plotter.save(fig, file_name) if save_plot else plt.show()