import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score

from ..metrics.threshold import ThresholdMetrics


class PlotThreshold:
    def __init__(self, true, pred_logits, class_mappings):
        self.true = true
        self.pred_logits = pred_logits

        self.threshold_metrics = ThresholdMetrics(true, pred_logits, class_mappings)

    def class_performance_by_threshold(self, predicted_class, threshold):
        """
        Calculates class wise stats for the given class.
        :param target_class: Class to calculate metrics.
        :param true: True labels for samples
        :param pred_categorical: Prediction for samples
        :return: payload_dict
        """

        predicted_class_logits = self.pred_logits.loc[predicted_class]
        thresholded_pred = (predicted_class_logits > threshold) * 1

        predicted_true = (self.true == predicted_class) * 1
        payload_dict = self.threshold_metrics.calculate_statistics_per_class(
            predicted_true, thresholded_pred
        )
        return payload_dict
