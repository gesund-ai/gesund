import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn


class TopLosses:
    def __init__(self, loss, meta_pred_true):
        """
        Initialize the TopLosses class.

        This constructor initializes the class with the loss values and metadata of predicted and true labels.

        :param loss: DataFrame or array-like structure containing the loss values for each sample.
        :param meta_pred_true: DataFrame containing metadata, predicted classes, and true classes.
        """
        self.loss = loss
        self.meta_pred_true = meta_pred_true

    def calculate_top_losses(self, predicted_class=None, top_k=10):
        """
        Calculate the top K losses for the predicted class or overall losses.

        This function retrieves the indices of the samples that correspond to the specified predicted class, 
        or calculates the top losses across all samples if no class is specified. The results are sorted 
        in descending order based on the loss values.

        :param predicted_class: (str, optional) The class for which to calculate top losses. 
                               If None, calculates for all classes.
        :param top_k: (int, optional) The number of top losses to return. Default is 10.

        :return: A DataFrame containing the sorted top losses. The DataFrame is sorted in descending 
                 order of loss values, with the top K entries returned.
        """
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
