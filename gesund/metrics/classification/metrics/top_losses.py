import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn


class TopLosses:
    def __init__(self, loss, meta_pred_true):
        self.loss = loss
        self.meta_pred_true = meta_pred_true

    def calculate_top_losses(self, predicted_class=None, top_k=10):

        if predicted_class:
            pred_categorical_target_class_index = self.meta_pred_true[
                self.meta_pred_true["pred_categorical"] == predicted_class
            ].index
            sorted_top_loss = self.loss[
                pred_categorical_target_class_index
            ].T.sort_values(by=0, ascending=False)
        else:
            sorted_top_loss = self.loss.T.sort_values(by=0, ascending=False)
        return sorted_top_loss
