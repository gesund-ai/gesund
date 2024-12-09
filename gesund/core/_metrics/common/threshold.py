from typing import Union, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

from gesund.core import metric_manager, plot_manager


class Classification:
    def _validate_data(self, data: dict) -> bool:
        """
        Validates the data required for metric calculation and plotting.
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        required_keys = ["prediction", "ground_truth"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Data must contain '{key}'.")
        if len(data["prediction"]) != len(data["ground_truth"]):
            raise ValueError(
                "Prediction and ground_truth must have the same number of samples."
            )
        return True

    def apply_metadata(self, data: dict, metadata: dict) -> dict:
        """
        Applies metadata to filter the data for calculation and plotting.
        """
        return data

    def calculate(self, data: dict) -> dict:
        """
        Calculates statistical tables for the given dataset.
        """
        # Validate the data
        self._validate_data(data)

        # Extract data
        true = np.array(data["ground_truth"])
        pred_categorical = np.array(data["prediction"])
        pred_logits = data.get("pred_logits", None)
        metadata = data.get("metadata", None)
        class_mappings = data.get("class_mappings", None)

        # Apply metadata if provided
        if metadata is not None:
            data = self.apply_metadata(data, metadata)
            true = np.array(data["ground_truth"])
            pred_categorical = np.array(data["prediction"])

        # Handle class mappings
        if class_mappings:
            classes = [int(k) for k in class_mappings.keys()]
            class_labels = [class_mappings[str(k)] for k in classes]
        else:
            classes = np.unique(true)
            class_labels = [str(k) for k in classes]
            class_mappings = {k: str(k) for k in classes}

        # Calculate confusion matrix
        cm = confusion_matrix(true, pred_categorical, labels=classes)

        # Calculate per-class statistics
        per_class_metrics = {}
        for idx, cls in enumerate(classes):
            tp = cm[idx, idx]
            fn = cm[idx, :].sum() - tp
            fp = cm[:, idx].sum() - tp
            tn = cm.sum() - (tp + fp + fn)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = (
                2 * precision * sensitivity / (precision + sensitivity)
                if (precision + sensitivity) > 0
                else 0
            )
            mcc = matthews_corrcoef(
                (true == cls).astype(int), (pred_categorical == cls).astype(int)
            )

            per_class_metrics[class_mappings[str(cls)]] = {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Precision": precision,
                "F1 Score": f1,
                "Matthews CC": mcc,
            }

        # Calculate overall metrics
        overall_accuracy = accuracy_score(true, pred_categorical)
        macro_f1 = f1_score(true, pred_categorical, average="macro")
        macro_precision = precision_score(true, pred_categorical, average="macro")
        macro_recall = recall_score(true, pred_categorical, average="macro")
        macro_specificity = np.mean(
            [metrics["Specificity"] for metrics in per_class_metrics.values()]
        )
        macro_mcc = matthews_corrcoef(true, pred_categorical)

        if pred_logits is not None:
            try:
                macro_auc = roc_auc_score(
                    true, pred_logits, multi_class="ovo", average="macro"
                )
            except ValueError:
                macro_auc = "Undefined"
        else:
            macro_auc = "Undefined"

        overall_metrics = {
            "Accuracy": overall_accuracy,
            "Macro F1 Score": macro_f1,
            "Macro Precision": macro_precision,
            "Macro Recall": macro_recall,
            "Macro Specificity": macro_specificity,
            "Matthews CC": macro_mcc,
            "Macro AUC": macro_auc,
        }

        result = {
            "per_class_metrics": per_class_metrics,
            "overall_metrics": overall_metrics,
            "confusion_matrix": cm,
            "classes": class_labels,
        }

        return result


class PlotStatsTables:
    def __init__(self, data: dict):
        self.data = data
        self.per_class_metrics = data["per_class_metrics"]
        self.overall_metrics = data["overall_metrics"]
        self.confusion_matrix = data["confusion_matrix"]
        self.classes = data["classes"]

    def _validate_data(self):
        """
        Validates the data required for plotting the stats tables.
        """
        required_keys = [
            "per_class_metrics",
            "overall_metrics",
            "confusion_matrix",
            "classes",
        ]
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Data must contain '{key}'.")

    def save(self, figure: Figure, filepath: str = "stats_tables.png") -> str:
        """
        Saves the specified figure to a file.

        Args:
            figure: The matplotlib Figure object to save
            filepath: The path where to save the figure

        Returns:
            str: The filepath where the figure was saved
        """
        figure.savefig(filepath)
        return filepath

    def plot(self) -> List[Figure]:
        """
        Creates the stats tables plots.

        Returns:
            List[Figure]: List containing the confusion matrix and metrics figures
        """
        self._validate_data()
        figures = []

        # Plot confusion matrix
        cm_fig = plt.figure(figsize=(8, 6))
        figures.append(cm_fig)

        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.classes,
            yticklabels=self.classes,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.close(cm_fig)

        # Plot per-class metrics
        metrics_df = pd.DataFrame(self.per_class_metrics).T
        metrics_to_plot = ["Sensitivity", "Specificity", "Precision", "F1 Score"]
        metrics_df = metrics_df[metrics_to_plot].reset_index()
        metrics_df = metrics_df.melt(
            id_vars="index", var_name="Metric", value_name="Value"
        )

        metrics_fig = plt.figure(figsize=(10, 6))
        figures.append(metrics_fig)

        sns.barplot(data=metrics_df, x="index", y="Value", hue="Metric", palette="Set2")
        plt.xlabel("Classes")
        plt.ylabel("Metric Value")
        plt.title("Per-Class Metrics")
        plt.legend(title="Metric")
        plt.close(metrics_fig)

        print("Overall Metrics:")
        for metric, value in self.overall_metrics.items():
            print(
                f"{metric}: {value:.4f}"
                if isinstance(value, float)
                else f"{metric}: {value}"
            )

        return figures


class SemanticSegmentation(Classification):
    pass


class ObjectDetection(Classification):
    pass


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("classification.stats_tables")
def calculate_stats_tables(data: dict, problem_type: str):
    """
    Calculates the stats tables metric.
    """
    metric_calculator = problem_type_map[problem_type]()
    result = metric_calculator.calculate(data)
    return result


@plot_manager.register("classification.stats_tables")
def plot_stats_tables(results: dict, save_plot: bool) -> Union[str, None]:
    """
    Plots the stats tables.
    """
    plotter = PlotStatsTables(data=results)
    figures = plotter.plot()
    if save_plot:
        for fig in figures:
            plotter.save(fig)

    return None
