import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn


class ConfusionMatrix:
    def __init__(self, class_mappings):
        self.class_mappings = class_mappings
        self.class_order = [int(i) for i in list(class_mappings.keys())]

    def calculate_confusion_matrix(self, true, pred_categorical, labels=None):
        """
        Identical to scikit-Learn confusion matrix
        :param true: true labels as a list = [1,0,3,4] for 4 sample dataset
        :param pred_categorical: categorical predictions as a list = [1,0,3,4] for 4 sample dataset
        :param labels: order of classes inside list
        :return: confusion matrix
        """
        if labels is not None:
            return sklearn.metrics.confusion_matrix(
                true, pred_categorical, labels=labels
            )
        else:
            return sklearn.metrics.confusion_matrix(
                true, pred_categorical, labels=self.class_order
            )
