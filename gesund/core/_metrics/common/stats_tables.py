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

COHORT_SIZE_LIMIT = 2
DEBUG = True


def categorize_age(age):
    if age < 18:
        return "Child"
    elif 18 <= age < 30:
        return "Young Adult"
    elif 30 <= age < 60:
        return "Adult"
    else:
        return "Senior"


class Classification:
    def _validate_data(self, data: dict) -> bool:
        """
        A function to validate the data that is required for metric calculation and plotting.

        :param data: The input data required for calculation, {"prediction":, "ground_truth": , "metadata":}
        :type data: dict

        :return: Status if the data is valid
        :rtype: bool
        """
        # Basic validation checks
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

    def __preprocess(self, data: dict, get_logits=False):
        """
        Preprocesses the data

        :param data: dictionary containing the data prediction, ground truth, metadata
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

    def apply_metadata(self, data: dict) -> dict:
        """
        Applies metadata to the data for metric calculation and plotting.

        :param data: The input data required for calculation, {"prediction":, "ground_truth": , "metadata":}
        :type data: dict

        :return: Filtered dataset
        :rtype: dict
        """
        # TODO:: This function could be global to be applied across metrics

        df: pd.DataFrame = pd.DataFrame.from_records(data["metadata"])
        cohorts_data = {}

        metadata_columns = df.columns.tolist()
        metadata_columns.remove("image_id")
        lower_case = {i: i.lower() for i in metadata_columns}
        df = df.rename(columns=lower_case)

        if "age" in list(lower_case.values()):
            df["age"] = df["age"].apply(categorize_age)

        for grp, subset_data in df.groupby(list(lower_case.values())):
            grp_str = ",".join([str(i) for i in grp])

            if subset_data.shape[0] < COHORT_SIZE_LIMIT:
                print(
                    f"Warning - grp excluded - {grp_str} cohort size < {COHORT_SIZE_LIMIT}"
                )
            else:
                image_ids = set(subset_data["image_id"].to_list())
                filtered_data = {
                    "prediction": {
                        i: data["prediction"][i]
                        for i in data["prediction"]
                        if i in image_ids
                    },
                    "ground_truth": {
                        i: data["ground_truth"][i]
                        for i in data["ground_truth"]
                        if i in image_ids
                    },
                    "metadata": subset_data,
                    "class_mapping": data["class_mapping"],
                }
                cohorts_data[grp_str] = filtered_data

        return data

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
            prediction, ground_truth = self.__preprocess(data, get_logits=True)
            pred_logits = prediction
            pred_categorical = np.argmax(prediction, axis=1)
        else:
            prediction, ground_truth = self.__preprocess(data)
            pred_categorical = prediction
            pred_logits = None
        true = ground_truth

        cm = confusion_matrix(true, pred_categorical, labels=class_order)
        per_class_metrics = {}
        for idx, cls in enumerate(class_order):
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

            per_class_metrics[class_mapping[str(cls)]] = {
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

    def calculate(self, data: dict) -> dict:
        """
        Calculates the lift chart for the given data.

        :param data: The input data required for calculation and plotting
                     {"prediction":, "ground_truth": , "metadata":, "class_mappings":}
        :type data: dict

        :return: Calculated metric results
        :rtype: dict
        """
        result = {}

        # Validate the data
        self._validate_data(data)
        metadata = data.get("metadata")

        if DEBUG:
            metadata = None

        if metadata:
            cohort_data = self.apply_metadata(data)
            for _cohort_key in cohort_data:
                result[_cohort_key] = self.__calculate_metrics(
                    cohort_data[_cohort_key], data.get("class_mapping")
                )
        else:
            result = self.__calculate_metrics(data, data.get("class_mapping"))

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

    def save(self, figs: List[Figure], filenames: List[str]) -> List[str]:
        """
        Saves multiple Matplotlib Figure objects to files.

        :param figs: List of Matplotlib Figure objects to save
        :type figs: List[Figure]
        :param filenames: List of filenames where the plot images will be saved
        :type filenames: List[str]

        :return: List of file paths where the plot images are saved
        :rtype: List[str]
        """
        filepaths = []
        dir_path = "plots"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for fig, filename in zip(figs, filenames):
            filepath = os.path.join(dir_path, filename)
            fig.savefig(filepath, format="png")
            filepaths.append(filepath)
        return filepaths

    def plot(self) -> List[Figure]:
        """
        Plots the stats tables and returns the figure objects.

        :return: List of Matplotlib Figure objects [confusion_matrix, per_class_metrics]
        :rtype: List[Figure]
        """
        self._validate_data()
        figures = []

        # Plot confusion matrix
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        cm = self.confusion_matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.classes,
            yticklabels=self.classes,
        )
        ax_cm.set_xlabel("Predicted Labels")
        ax_cm.set_ylabel("True Labels")
        ax_cm.set_title("Confusion Matrix")
        figures.append(fig_cm)

        # Plot per-class metrics
        metrics_df = pd.DataFrame(self.per_class_metrics).T
        metrics_to_plot = ["Sensitivity", "Specificity", "Precision", "F1 Score"]
        metrics_df = metrics_df[metrics_to_plot].reset_index()
        metrics_df = metrics_df.melt(
            id_vars="index", var_name="Metric", value_name="Value"
        )

        fig_metrics, ax_metrics = plt.subplots(figsize=(10, 6))
        sns.barplot(data=metrics_df, x="index", y="Value", hue="Metric", palette="Set2")
        ax_metrics.set_xlabel("Classes")
        ax_metrics.set_ylabel("Metric Value")
        ax_metrics.set_title("Per-Class Metrics")
        ax_metrics.legend(title="Metric")
        figures.append(fig_metrics)

        # Display overall metrics
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
def plot_stats_tables(results: dict, save_plot: bool) -> Union[List[str], List[Figure]]:
    """
    Plots the stats tables.

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: Boolean value to save plots
    :type save_plot: bool

    :return: List of figure objects or paths to the saved plots
    :rtype: Union[List[str], List[Figure]]
    """
    plotter = PlotStatsTables(data=results)
    figs = plotter.plot()
    if save_plot:
        filenames = [f"plot_{i}.png" for i in range(len(figs))]
        return plotter.save(figs, filenames=filenames)
    else:
        plt.show()
