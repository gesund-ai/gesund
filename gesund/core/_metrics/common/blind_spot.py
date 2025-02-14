import os
from typing import Union, Dict, List, Optional

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

        :param data: The input data required for calculation, {"prediction":, "ground_truth": }
        :type data: dict

        :return: Status if the data is valid
        :rtype: bool
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        required_keys = ["prediction", "ground_truth"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Data must contain '{key}'.")

        if len(data["prediction"]) != len(data["ground_truth"]):
            raise ValueError("Prediction and ground_truth must have the same length.")

        if (
            len(
                set(list(data["prediction"].keys())).difference(
                    set(list(data["ground_truth"].keys()))
                )
            )
            > 0
        ):
            raise ValueError("Prediction and ground_truth must have the same keys.")

        return True

    def _preprocess(self, data: dict, get_logits=False):
        """
        Preprocesses the data

        :param data: dictionary containing the data prediction, ground truth
        :type data: dict
        :param get_logits: in case of multi class classification set to True
        :type get_logits: boolean

        :return: data tuple
        :rtype: tuple
        """
        prediction, ground_truth = [], []
        for image_id in data["ground_truth"]:
            sample_gt = data["ground_truth"][image_id]
            sample_pred = data["prediction"][image_id]
            ground_truth.append(sample_gt["annotation"][0]["label"])

            if get_logits:
                prediction.append(sample_pred["logits"])
            else:
                prediction.append(sample_pred["prediction_class"])
        return (np.asarray(prediction), np.asarray(ground_truth))

    def __calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        """
        A function to calculate the metrics

        :param data: data dictionary containing data
        :type data: dict
        :param class_mapping: a dictionary with class mapping labels
        :type class_mapping: dict

        :return: results calculated
        :rtype: dict
        """
        class_order = [int(i) for i in class_mapping.keys()]
        class_labels = [class_mapping[str(k)] for k in class_order]

        true = np.array(data["ground_truth"])
        pred_categorical = np.array(data["prediction"])
        pred_logits = data.get("pred_logits", None)

        if len(class_order) > 2:
            prediction, ground_truth = self._preprocess(data, get_logits=True)
            pred_logits = prediction
            pred_categorical = np.argmax(prediction, axis=1)
        else:
            prediction, ground_truth = self._preprocess(data)
            pred_categorical = prediction
            pred_logits = None

        true = ground_truth
        cm = confusion_matrix(true, pred_categorical, labels=class_order)
        per_class_metrics = {}
        TP_sum = FP_sum = FN_sum = TN_sum = 0
        for idx, _cls in enumerate(class_order):
            tp = cm[idx, idx]
            fn = cm[idx, :].sum() - tp
            fp = cm[:, idx].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            TP_sum += tp
            FP_sum += fp
            FN_sum += fn
            TN_sum += tn

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = (
                2 * precision * sensitivity / (precision + sensitivity)
                if (precision + sensitivity) > 0 else 0
            )
            mcc = matthews_corrcoef(
                (true == _cls).astype(int), (pred_categorical == _cls).astype(int)
            )

            per_class_metrics[class_mapping[str(_cls)]] = {
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

        # Calculate micro metrics
        if TP_sum + FP_sum > 0:
            micro_precision = TP_sum / (TP_sum + FP_sum)
        else:
            micro_precision = 0.0

        if TP_sum + FN_sum > 0:
            micro_recall = TP_sum / (TP_sum + FN_sum)
        else:
            micro_recall = 0.0

        if micro_precision + micro_recall > 0:
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        else:
            micro_f1 = 0.0

        if TN_sum + FP_sum > 0:
            micro_specificity = TN_sum / (TN_sum + FP_sum)
        else:
            micro_specificity = 0.0

        if pred_logits is not None:
            try:
                micro_auc = roc_auc_score(true, pred_logits, multi_class="ovo", average="micro")
            except ValueError:
                micro_auc = np.nan
        else:
            micro_auc = np.nan

        # Calculate overall metrics
        overall_accuracy = accuracy_score(true, pred_categorical)
        macro_f1 = f1_score(true, pred_categorical, average="macro")
        macro_precision = precision_score(true, pred_categorical, average="macro")
        macro_recall = recall_score(true, pred_categorical, average="macro")
        macro_specificity = np.mean(
            [metrics["Specificity"] for metrics in per_class_metrics.values()]
        )
        macro_mcc = matthews_corrcoef(true, pred_categorical)

        overall_metrics = {
            "Accuracy": overall_accuracy,
            "Macro F1 Score": macro_f1,
            "Macro Precision": macro_precision,
            "Macro Recall": macro_recall,
            "Macro Specificity": macro_specificity,
            "Matthews CC": macro_mcc,
            "Macro AUC": roc_auc_score(true, pred_logits, multi_class="ovo", average="macro") if pred_logits is not None else np.nan,
            "Micro Precision": micro_precision,
            "Micro Recall": micro_recall,
            "Micro F1 Score": micro_f1,
            "Micro Specificity": micro_specificity,
            "Micro AUC": micro_auc,
        }

        result = {
            "per_class_metrics": per_class_metrics,
            "overall_metrics": overall_metrics,
            "confusion_matrix": cm,
            "classes": class_labels,
        }

        return result

    def calculate(self, data: dict) -> dict:
        """
        Calculates the blind spot chart for the given data.

        :param data: The input data required for calculation and plotting
                     {"prediction":, "ground_truth": , "metadata":, "class_mappings":}
        :type data: dict

        :return: Calculated metric results
        :rtype: dict
        """
        result = {}

        self._validate_data(data)

        result = self.__calculate_metrics(data, data.get("class_mapping"))

        return result

class ObjectDetection:
    pass

class SemanticSegmentation:
    pass

class PlotBlindSpot:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        """
        Initializes the PlotBlindSpot with data and an optional cohort identifier.

        :param data: Dictionary containing metric results.
        :type data: dict
        :param cohort_id: Optional identifier for the cohort, defaults to None.
        :type cohort_id: Optional[int], optional
        """
        self.data = data
        self.per_class_metrics = data["per_class_metrics"]
        self.overall_metrics = data["overall_metrics"]
        self.confusion_matrix = data["confusion_matrix"]
        self.classes = data["classes"]
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        Validates the data required for plotting the stats tables.

        :raises ValueError: If required data keys are missing.
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

    def save(self, fig: Figure, filename: str) -> str:
        """
        Saves multiple Matplotlib Figure objects to files.

        :param figs: List of Matplotlib Figure objects to save
        :type figs: List[Figure]
        :param filenames: List of filenames where the plot images will be saved
        :type filenames: List[str]

        :return: List of file paths where the plot images are saved
        :rtype: List[str]
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
        Plots the blind spot chart.

        :return: Matplotlib Figure object with the blind spot plot.
        :rtype: Figure

        :raises ValueError: If plotting fails.
        """
        self._validate_data()
        sns.set_style("whitegrid")

        metrics_df = pd.DataFrame.from_dict(self.per_class_metrics, orient='index').reset_index()
        metrics_df.rename(columns={'index': 'Class'}, inplace=True)

        per_class_cols = ["Sensitivity", "Specificity", "Precision", "F1 Score"]
        for col in per_class_cols:
            if col not in metrics_df.columns:
                metrics_df[col] = np.nan

        metrics_df = metrics_df[["Class"] + per_class_cols]
        class_df = metrics_df.melt(id_vars="Class", var_name="Metric", value_name="Value")

        macro_cols = [
            "Accuracy",
            "Macro F1 Score",
            "Macro Precision",
            "Macro Recall",
            "Macro Specificity",
        ]
        micro_cols = [
            "Micro Precision",
            "Micro Recall",
            "Micro F1 Score",
            "Micro Specificity",
        ]

        # Macro
        macro_values = {c: self.overall_metrics.get(c, np.nan) for c in macro_cols}
        macro_df = pd.DataFrame([macro_values]).melt(var_name="Metric", value_name="Value")
        macro_df["Class"] = "Overall (Macro)"

        # Micro
        micro_values = {c: self.overall_metrics.get(c, np.nan) for c in micro_cols}
        micro_df = pd.DataFrame([micro_values]).melt(var_name="Metric", value_name="Value")
        micro_df["Class"] = "Overall (Micro)"

        # Combine
        plot_df = pd.concat([class_df, macro_df, micro_df], ignore_index=True)

        unique_classes = plot_df["Class"].unique()
        base_colors = sns.color_palette("Set2", len(unique_classes))
        custom_palette = {}
        for idx, cls_name in enumerate(unique_classes):
            if cls_name == "Overall (Macro)":
                custom_palette[cls_name] = "blue"
            elif cls_name == "Overall (Micro)":
                custom_palette[cls_name] = "red"
            else:
                custom_palette[cls_name] = base_colors[idx]

        fig_metrics, ax_metrics = plt.subplots(figsize=(20, 15))
        sns.barplot(
            data=plot_df,
            x="Metric",
            y="Value",
            hue="Class",
            palette=custom_palette,
            ax=ax_metrics
        )

        for p in ax_metrics.patches:
            height = p.get_height()
            ax_metrics.annotate(
                format(height, ".2f"),
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                xytext=(0, 6),
                textcoords="offset points"
            )

        ax_metrics.set_xlabel("Metrics", fontdict={"fontsize": 14, "fontweight": "medium"})
        ax_metrics.set_ylabel("Metric Value", fontdict={"fontsize": 14, "fontweight": "medium"})
        if self.cohort_id:
            ax_metrics.set_title(
                f"Comparison of Performance Metrics Across Classes : cohort - {self.cohort_id}",
                fontdict={"fontsize": 16, "fontweight": "medium"}
            )
        else:
            ax_metrics.set_title(
                "Comparison of Performance Metrics Across Classes",
                fontdict={"fontsize": 16, "fontweight": "medium"}
            )
        ax_metrics.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        return fig_metrics


problem_type_map = {
    "classification": Classification,
    "object_detection": ObjectDetection,
    "semantic_segmentation": SemanticSegmentation,
}

@metric_manager.register("classification.blind_spot")
def calculate_blind_spot(data: dict, problem_type: str) -> dict:
    """
    A wrapper function to calculate the blind spot.

    :param data: Dictionary of data: {"prediction": , "ground_truth": }
    :type data: dict
    :param problem_type: Type of the problem
    :type problem_type: str

    :return: Dict of calculated results
    :rtype: dict
    """
    metric_manager = problem_type_map[problem_type]()
    result = metric_manager.calculate(data)
    return result

@plot_manager.register("classification.blind_spot")
def plot_blind_spot(
    results: dict,
    save_plot: bool,
    file_name: Optional[str] = "blind_spot.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to plot the blind spot.

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: Boolean value to save plot
    :type save_plot: bool
    :param file_name: name of the file
    :type file_name: str
    :param cohort_id: id of the cohort
    :type cohort_id: int

    :return: None or path to the saved plot
    :rtype: Union[str, None]
    """
    plotter = PlotBlindSpot(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, file_name)
    else:
        plt.show()