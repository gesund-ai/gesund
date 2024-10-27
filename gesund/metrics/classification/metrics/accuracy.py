import numpy as np
import pandas as pd
from sklearn.metrics import auc


class Accuracy:
    def __init__(self, class_mappings=None):
        """
        Initialize the Accuracy class with optional class mappings.

        :param class_mappings: (dict, optional) A dictionary mapping class labels to integers. If provided, the order of
                               classes is based on the keys of this dictionary for calculating per-class accuracy.
        """
        self.class_mappings = class_mappings
        if self.class_mappings:
            self.class_order = [int(i) for i in list(class_mappings.keys())]

    def _calculate_accuracy(self, true, pred_categorical):
        """
        Calculate accuracy between true and predicted categorical labels.

        :param true: (array-like) Array containing the true labels.
        :param pred_categorical: (array-like) Array containing the predicted categorical labels.

        :return: (float) Accuracy score between 0 and 1. If the input arrays are empty, returns 0.
        """
        if len(true) != 0:
            return float(np.sum(true == pred_categorical) / len(true))
        else:
            return 0

    def calculate_accuracy(self, true, pred_categorical, target_class=None):
        """
        Calculate accuracy for the dataset or specific class.

        This function computes accuracy based on the provided true and predicted labels, allowing for options to calculate
        accuracy for the entire dataset, individual classes, or an overall score. If `target_class` is specified, accuracy
        is calculated for that class only.

        :param true: (array-like) Array of true labels.
        :param pred_categorical: (array-like) Array of predicted categorical labels.
        :param target_class: (str or int, optional) Specifies which class to calculate accuracy for. Options include:
            - "overall": Returns overall accuracy across all classes.
            - "all": Returns a dictionary with accuracy scores for each class.
            - (int): Specific class label to calculate accuracy for that particular class.

        :return: (float or dict) Overall accuracy (if `target_class` is "overall" or None), per-class accuracy (if 
                 `target_class` is "all"), or accuracy for a specific class (if `target_class` is an integer representing
                 a class label).
        """
        if true is None:
            true = self.true
        if pred_categorical is None:
            pred_categorical = self.pred_categorical

        if target_class == "overall" or target_class is None:
            return self._calculate_accuracy(true, pred_categorical)

        elif target_class == "all":
            class_accuracies = {}
            for target_class in self.class_order:
                true_idx = true == target_class
                class_true = true[true_idx]
                class_pred_categorical = pred_categorical[true_idx]
                class_accuracies[target_class] = self._calculate_accuracy(
                    class_true, class_pred_categorical
                )
            return class_accuracies

        elif target_class is not None:
            true_idx = true == target_class
            class_true = true[true_idx]
            class_pred_categorical = pred_categorical[true_idx]
            return self._calculate_accuracy(class_true, class_pred_categorical)
        else:
            return self._calculate_accuracy(true, pred_categorical)
