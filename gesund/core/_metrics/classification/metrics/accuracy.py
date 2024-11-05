import numpy as np
import pandas as pd
from sklearn.metrics import auc


class Accuracy:
    def __init__(self, class_mappings=None):
        self.class_mappings = class_mappings
        if self.class_mappings:
            self.class_order = [int(i) for i in list(class_mappings.keys())]

    def _calculate_accuracy(self, true, pred_categorical):
        if len(true) != 0:
            return float(np.sum(true == pred_categorical) / len(true))
        else:
            return 0

    def calculate_accuracy(self, true, pred_categorical, target_class=None):
        """
        Returns accuracy for all dataset between 0-1.
        return: accuracy between 0-1.
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
