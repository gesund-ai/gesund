from locale import normalize
import numpy as np
import sklearn
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

from .accuracy import Accuracy
from .auc import AUC
from .most_confused import MostConfused
from .dataset_stats import DatasetStats
from gesund.utils.validation_data_utils import Statistics


class StatsTables:
    def __init__(self, class_mappings):
        """
        Initialize the StatsTables object with class mappings.

        :param class_mappings: A dictionary mapping class indices to class names.
        """
        self.class_mappings = class_mappings
        self.class_order = list(range(len(class_mappings.keys())))

        self.auc = AUC(class_mappings)
        self.accuracy = Accuracy(class_mappings)

    def calculate_statistics_classbased_table(self, true, pred_categorical):
        """
        Calculate statistics for class-based metrics.

        This function computes various statistics such as True Positives, False Positives,
        True Negatives, and False Negatives for each class based on the true and predicted
        categorical values. It renames the metrics for better readability.

        :param true: The ground truth values as a pandas Series or array-like.
        :param pred_categorical: The predicted values as a pandas Series or array-like.
        
        :return: A dictionary containing the renamed statistics for each class.
        """
        rename_statistics_dict = {
            "FP": "False Positive",
            "TP": "True Positive",
            "FN": "False Negative",
            "TN": "True Negative",
            "TPR": "Sensitivity (TPR)",
            "TNR": "Specificity (TNR)",
            "PPV": "Precision",
            "NPV": "Negative Predictive Value",
            "FPR": "False Positive Rate",
            "FNR": "False Negative Rate",
            "F1": "F1",
        }
        sense_spec_dict_renamed = {}
        sense_spec_dict = self.auc.calculate_sense_spec(
            true=true, pred_categorical=pred_categorical
        )

        class_accuracies = self.accuracy.calculate_accuracy(
            true=true, pred_categorical=pred_categorical, target_class="all"
        )

        for keys in rename_statistics_dict:
            sense_spec_dict_renamed[
                str(rename_statistics_dict[keys])
            ] = sense_spec_dict[keys]

        sense_spec_dict_renamed["Accuracy"] = class_accuracies
        return sense_spec_dict_renamed

    def calculate_highlighted_overall_metrics(
        self, true, pred_categorical, pred_logits, cal_conf_interval=False
    ):
        """
        Calculate overall highlighted metrics including accuracy, F1 scores, and AUC.

        This function aggregates various metrics across all classes, including accuracy,
        macro and micro F1 scores, precision, recall, and AUC, optionally calculating
        confidence intervals for the metrics.

        :param true: The ground truth values as a pandas Series or array-like.
        :param pred_categorical: The predicted categorical values as a pandas Series or array-like.
        :param pred_logits: The predicted logits from the model as an array-like.
        :param cal_conf_interval: Whether to calculate confidence intervals for metrics.
        
        :return: A tuple of calculated metrics or a list of metrics with confidence intervals if specified.
        """
        accuracy = self.accuracy.calculate_accuracy(
            true=true, pred_categorical=pred_categorical, target_class="overall"
        )

        sense_spec_metrics = self.auc.calculate_sense_spec(true, pred_categorical)

        if len(self.class_mappings) > 2:
            _pred = pred_logits.T
        else:
            _pred = pred_categorical

        # Note: multiclass ROC AUC currently only handles the ‘macro’ and ‘weighted’ averages in sklearn.metrics.roc_auc_score v1.0.2
        macro_auc = self.auc.calculate_multiclass_roc_statistics(true, pred_logits)[
            "Macro"
        ]
        micro_f1 = sense_spec_metrics["Micro F1"]
        macro_f1 = sense_spec_metrics["Macro F1"]
        macro_prec = sense_spec_metrics["Macro Precision"]
        micro_prec = sense_spec_metrics["Micro Precision"]
        macro_rec = sense_spec_metrics["Macro Recall"]
        micro_rec = sense_spec_metrics["Micro Recall"]
        macro_specificity = sense_spec_metrics["Macro Specificity"]
        micro_specificity = sense_spec_metrics["Micro Specificity"]
        matthews = sense_spec_metrics["Matthew's Correlation Coefficient"]

        metrics_list = (
            accuracy,
            micro_f1,
            macro_f1,
            macro_auc,
            micro_prec,
            macro_prec,
            micro_rec,
            macro_rec,
            macro_specificity,
            micro_specificity,
            matthews,
        )

        if cal_conf_interval:
            new_metric_list = []
            for metric in metrics_list:
                if type(metric) != str:
                    c_i = Statistics.calculate_confidence_interval(metric, len(true))
                else:
                    c_i = "nan"
                new_metric_list.append((metric, c_i))

            return new_metric_list
        return metrics_list

    def _calculate_shannon_diversity_index(self, true):
        """
        Calculate the Shannon diversity index for the true labels.

        This function computes the Shannon diversity index, which quantifies
        the diversity of classes present in the true labels.

        :param true: The ground truth values as a pandas Series or array-like.
        
        :return: A float representing the Shannon diversity index.
        """
        class_dist_dict = true.value_counts().to_dict()
        if len(class_dist_dict) > 1:
            n = true.shape[0]
            classes = [
                (class_name, float(count))
                for class_name, count in class_dist_dict.items()
            ]
            k = len(class_dist_dict)
            H = -sum(
                [(count / n) * np.log((count / n)) for class_name, count in classes]
            )  # shannon entropy
            return H / np.log(k)
        else:  # in case where there is only one class then its highly imbalanaced
            return 0

    def calculate_blind_spot_metrics(self, true, pred_categorical, pred_logits):
        """
        Calculate metrics focusing on blind spots in the predictions.

        This function computes a range of metrics, including overall accuracy, 
        class-wise accuracy, and Shannon diversity index, to identify potential
        blind spots in the model's performance.

        :param true: The ground truth values as a pandas Series or array-like.
        :param pred_categorical: The predicted categorical values as a pandas Series or array-like.
        :param pred_logits: The predicted logits from the model as an array-like.
        
        :return: A dictionary containing the metrics for blind spots across classes.
        """
        # Calculate Metrics
        accuracy = self.accuracy.calculate_accuracy(
            true=true, pred_categorical=pred_categorical, target_class="overall"
        )

        class_accuracies = self.accuracy.calculate_accuracy(
            true=true, pred_categorical=pred_categorical, target_class="all"
        )

        classwise_occurrence = true.value_counts().to_dict()
        for key in list(classwise_occurrence.keys()):
            classwise_occurrence[
                self.class_mappings[str(key)]
            ] = classwise_occurrence.pop(key)
        classwise_occurrence["total"] = true.count()

        sense_spec_metrics = self.auc.calculate_sense_spec(true, pred_categorical)

        if len(self.class_mappings) > 2:
            _pred = pred_logits.T
        else:
            _pred = pred_categorical

        # Create table dict
        filters = ["Average"] + list(self.class_mappings.keys())
        blind_spot_metrics_dict = dict()
        for filter_ in filters:
            blind_spot_metrics_dict[filter_] = dict()

        # calcuate shannon diversity index
        shannon_di = self._calculate_shannon_diversity_index(true)

        # Fill metrics in
        blind_spot_metrics_dict["Average"]["Accuracy"] = accuracy

        # Note: multiclass ROC AUC currently only handles the ‘macro’ and ‘weighted’ averages in sklearn.metrics.roc_auc_score v1.0.2
        self.auc.calculate_multiclass_roc_statistics(true, pred_logits)["Macro"]
        blind_spot_metrics_dict["Average"][
            "Macro AUC"
        ] = self.auc.calculate_multiclass_roc_statistics(true, pred_logits)["Macro"]
        blind_spot_metrics_dict["Average"]["Micro F1"] = sense_spec_metrics["Micro F1"]
        blind_spot_metrics_dict["Average"]["Macro F1"] = sense_spec_metrics["Macro F1"]
        blind_spot_metrics_dict["Average"]["Macro Precision"] = sense_spec_metrics[
            "Macro Precision"
        ]
        blind_spot_metrics_dict["Average"]["Micro Precision"] = sense_spec_metrics[
            "Micro Precision"
        ]
        blind_spot_metrics_dict["Average"]["Macro Sensitivity"] = sense_spec_metrics[
            "Macro Recall"
        ]
        blind_spot_metrics_dict["Average"]["Micro Sensitivity"] = sense_spec_metrics[
            "Micro Recall"
        ]
        blind_spot_metrics_dict["Average"]["Macro Specificity"] = sense_spec_metrics[
            "Macro Specificity"
        ]
        blind_spot_metrics_dict["Average"]["Micro Specificity"] = sense_spec_metrics[
            "Micro Specificity"
        ]
        blind_spot_metrics_dict["Average"]["Matthews C C"] = sense_spec_metrics[
            "Matthew's Correlation Coefficient"
        ]
        blind_spot_metrics_dict["Average"]["Diversity Index"] = shannon_di
        blind_spot_metrics_dict["Average"]["Sample Size"] = true.shape[0]

        for class_ in self.class_mappings:
            class_int = int(class_)
            blind_spot_metrics_dict[class_]["Accuracy"] = class_accuracies[class_int]
            blind_spot_metrics_dict[class_]["F1"] = sense_spec_metrics["F1"][class_int]
            blind_spot_metrics_dict[class_]["Precision"] = sense_spec_metrics["PPV"][
                class_int
            ]
            blind_spot_metrics_dict[class_]["Sensitivity"] = sense_spec_metrics["TPR"][
                class_int
            ]
            blind_spot_metrics_dict[class_]["Specificity"] = sense_spec_metrics["TNR"][
                class_int
            ]
            blind_spot_metrics_dict[class_]["Matthews C C"] = sense_spec_metrics[
                "Matthew's Classwise C C"
            ][class_int]
            blind_spot_metrics_dict[class_]["Class Size"] = sum(true == class_int)

        return blind_spot_metrics_dict
