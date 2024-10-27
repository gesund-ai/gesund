import math

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score

from ..metrics.auc import AUC
from ..metrics.stats_tables import *
from ..plots.stats_tables import PlotStatsTables


class PlotBlindSpot:
    def __init__(
        self, true, pred_logits, pred_categorical, class_mappings, meta_pred_true
    ):
        """
        Initialize the PlotBlindSpot class with true labels, prediction logits, categorical predictions, 
        class mappings, and metadata of predictions.

        :param true: (pd.Series) True labels for the classification problem.
        :param pred_logits: (pd.DataFrame) Prediction logits for each class.
        :param pred_categorical: (pd.Series) Categorical predictions based on the logits.
        :param class_mappings: (dict) Mapping of class indices to class names.
        :param meta_pred_true: (pd.DataFrame) Metadata containing additional information about predictions.
        """
        self.true = true
        self.pred_logits = pred_logits
        self.pred_categorical = pred_categorical
        self.class_mappings = class_mappings
        self.class_order = [int(i) for i in list(class_mappings.keys())]

        self.plot_stats_tables = PlotStatsTables(
            true=true,
            pred_logits=pred_logits,
            pred_categorical=pred_categorical,
            meta_pred_true=meta_pred_true,
            class_mappings=class_mappings,
        )

        self.auc = AUC(class_mappings)

    def blind_spot_metrics(self, target_attribute_dict=None):
        """
        Calculate blind spot metrics for the classification problem.

        This function utilizes the PlotStatsTables class to compute metrics that help identify 
        the blind spots in predictions, specifically where certain classes may be underrepresented 
        or misclassified.

        :param target_attribute_dict: (dict, optional) A dictionary of attributes to filter the metrics. 
            If provided, only metrics for the specified attributes will be calculated.

        :return: (dict) A dictionary containing the blind spot metrics, including class-wise statistics 
            and any relevant performance metrics.
        """
        blind_spot_metrics_dict = self.plot_stats_tables.blind_spot_metrics(
            target_attribute_dict
        )

        return blind_spot_metrics_dict
