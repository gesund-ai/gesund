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
        Plots ROC Curve for target_class.
        References:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        https://plotly.com/python/roc-and-pr-curves/
        :param target_class: target class to produce ROC plot
        :return: payload_dict
        """
        blind_spot_metrics_dict = self.plot_stats_tables.blind_spot_metrics(
            target_attribute_dict
        )

        return blind_spot_metrics_dict
