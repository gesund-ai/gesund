import math

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score

from ..metrics.auc import AUC
from gesund.core._utils import ValidationUtils, Statistics


class PlotAUC:
    def __init__(self, pred_logits, meta_pred_true, class_mappings):
        self.pred_logits = pred_logits
        self.class_mappings = class_mappings
        self.meta_pred_true = meta_pred_true

        self.validation_utils = ValidationUtils(meta_pred_true)
        self.auc = AUC(class_mappings=class_mappings)

    def precision_recall_multiclass_statistics(self, target_attribute_dict=None):
        """
        Plots ROC Curve for target_class.
        References:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        https://plotly.com/python/roc-and-pr-curves/
        :param target_class: target class to produce ROC plot
        :return: payload_dict
        """
        filtered_meta_pred_true = self.validation_utils.filter_attribute_by_dict(
            target_attribute_dict
        )
        true = filtered_meta_pred_true["true"]
        filtered_pred_logits = self.pred_logits.T.loc[
            filtered_meta_pred_true["true"].index
        ].T
        (points, aucs,) = self.auc.calculate_multiclass_precision_recall_statistics(
            true, filtered_pred_logits
        )

        payload_dict = {"type": "precision", "data": {"points": points, "aucs": aucs}}
        return payload_dict

    def roc_multiclass_statistics(self, target_attribute_dict=None):
        """
        Plots ROC Curve for target_class, returns either plotly or matplotlib object.
        References:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        https://plotly.com/python/roc-and-pr-curves/
        :param target_class: target class to produce ROC plot
        :param true: True labels.
        :param pred_logits: prediction logits.
        :return: matplotlib or plotly fig object
        """
        filtered_meta_pred_true = self.validation_utils.filter_attribute_by_dict(
            target_attribute_dict
        )
        true = filtered_meta_pred_true["true"]
        filtered_pred_logits = self.pred_logits.T.loc[
            filtered_meta_pred_true["true"].index
        ].T

        points, aucs = self.auc.calculate_multiclass_roc_statistics(
            true=true, pred_logits=filtered_pred_logits, return_points=True
        )

        payload_dict = {"type": "roc", "data": {"points": points, "aucs": aucs}}
        return payload_dict

    def confidence_histogram_scatter_distribution(
        self, predicted_class="overall", n_samples=300, randomize_x=True, n_bins=25
    ):
        # Plot Scatters

        # Filtering data
        pred_true = self.meta_pred_true[["pred_categorical", "true"]].copy()

        if predicted_class != "overall":
            pred_true = pred_true[pred_true["pred_categorical"] == predicted_class]

        if n_samples > pred_true.shape[0]:
            n_samples = pred_true.shape[0]
        pred_true = pred_true.sample(n_samples, replace=True)

        # Filter and obtain probabilities
        pred_true["labels"] = pred_true["pred_categorical"] == pred_true["true"]
        pred_true["labels"] = (
            pred_true["labels"].replace(True, "TP").replace(False, "FP")
        )

        filtered_pred_logits = self.pred_logits[pred_true.index].max()
        pred_true["y"] = filtered_pred_logits

        # Renaming columns
        int_class_mappings = {int(k): v for k, v in self.class_mappings.items()}
        pred_true = pred_true.replace(
            {"pred_categorical": int_class_mappings, "true": int_class_mappings}
        )
        pred_true = pred_true.rename(
            columns={"pred_categorical": "Prediction", "true": "Ground Truth"}
        )

        # Randomize x if needed
        if randomize_x:
            pred_true["x"] = np.random.uniform(0, 1, pred_true.shape[0])
        else:
            pred_true["x"] = pred_true["y"]

        points = list(
            pred_true.reset_index()
            .rename(columns={"index": "image_id"})
            .T.to_dict()
            .values()
        )

        # Plot histogram

        histogram = Statistics.calculate_histogram(
            pred_true["y"], min_=0, max_=1, n_bins=n_bins
        )

        payload_dict = {
            "type": "mixed",
            "data": {"points": points, "histogram": histogram},
        }

        return payload_dict
