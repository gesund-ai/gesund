import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn

from ..metrics.confusion_matrix import ConfusionMatrix
from gesund.utils.validation_data_utils import ValidationUtils, Statistics


class PlotConfusionMatrix:
    def __init__(self, meta_pred_true, class_mappings):
        """
        Initialize the PlotConfusionMatrix class with metadata, true labels, and class mappings.

        :param meta_pred_true: (pd.DataFrame) A DataFrame containing the true labels and predicted labels.
        :param class_mappings: (dict) Mapping of class indices to class names.
        """
        self.class_mappings = class_mappings
        self.class_order = [int(i) for i in list(class_mappings.keys())]

        self.meta_pred_true = meta_pred_true
        self.confusion_matrix = ConfusionMatrix(class_mappings)
        self.validation_utils = ValidationUtils(meta_pred_true)

    def confusion_matrix_(self, target_attribute_dict=None):
        """
        Calculate the confusion matrix based on true and predicted labels.

        This function filters the metadata according to the specified attributes, computes the confusion matrix,
        and formats the results to include class names and their corresponding values.

        :param target_attribute_dict: (dict, optional) A dictionary of attributes used to filter the samples.
            If provided, only the samples matching these attributes will be considered for the confusion matrix.

        :return: A dictionary containing the confusion matrix and additional information, including:
            - 'type' (str): Type of the metric, which is "confusion".
            - 'data' (dict): Contains:
                - 'Validation' (list): A list of dictionaries with actual class names, predicted class names,
                  and corresponding values.
                - 'class_order' (dict): The class mappings used in the confusion matrix.
        """
        filtered_meta_pred_true = self.validation_utils.filter_attribute_by_dict(
            target_attribute_dict
        )
        true = filtered_meta_pred_true["true"]
        pred_categorical = filtered_meta_pred_true["pred_categorical"]

        #  , labels=[0,1,2,3,4] # Burak: TO DO: Class Order
        z = self.confusion_matrix.calculate_confusion_matrix(true, pred_categorical)
        x = self.class_order.copy()
        y = self.class_order.copy()

        confusion_matrix_w_class_names = []
        for x_ in x:
            for y_ in y:
                confusion_matrix_w_class_names.append(
                    {
                        "actual": self.class_mappings[str(x_)],
                        "prediction": self.class_mappings[str(y_)],
                        "value": float(z[y_, x_]),
                    }
                )

        payload_dict = {
            "type": "confusion",
            "data": {
                "Validation": confusion_matrix_w_class_names,
                "class_order": self.class_mappings,
            },
        }  #  ,  z.tolist(), "class_order": self.class_order
        return payload_dict
