import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn

from ..metrics import LiftChart
from gesund.utils.validation_data_utils import ValidationUtils, Statistics


class PlotLiftGainChart:
    def __init__(self, true, pred_logits, meta, class_mappings):
        """
        Initialize the PlotLiftGainChart class with true labels, predicted logits, metadata, and class mappings.

        :param true: (pd.Series) A Series containing the true labels for the samples.
        :param pred_logits: (pd.DataFrame) A DataFrame containing the predicted logits for the samples.
        :param meta: (pd.DataFrame) A DataFrame containing metadata associated with the samples.
        :param class_mappings: (dict) Mapping of class indices to class names.
        """
        self.class_mappings = class_mappings
        self.class_order = [int(i) for i in list(class_mappings.keys())]

        self.true = true
        self.pred_logits = pred_logits
        self.lift_chart_calculate = LiftChart(class_mappings)
        self.validation_utils = ValidationUtils(meta)

    def lift_chart(self, target_attribute_dict, predicted_class=None):
        """
        Calculate the lift curve points based on true labels and predicted logits.

        This function filters the metadata according to the specified attributes, computes the lift points,
        and formats the results to include class names.

        :param target_attribute_dict: (dict) A dictionary of attributes used to filter the samples.
            The function will only consider the samples that match these attributes when calculating the lift curve.

        :param predicted_class: (int, optional) The specific predicted class for which to calculate the lift curve.
            If not provided, the lift curve will be calculated for all classes.

        :return: A dictionary containing the lift curve data and additional information, including:
            - 'type' (str): Type of the metric, which is "lift".
            - 'data' (dict): Contains:
                - 'points' (list): A list of points representing the lift curve.
                - 'class_order' (dict): The class mappings used in the lift calculation.
        """
        filtered_meta_pred_true = self.validation_utils.filter_attribute_by_dict(
            target_attribute_dict
        )

        lift_points_dict = self.lift_chart_calculate.calculate_lift_curve_points(
            self.true.loc[filtered_meta_pred_true.index],
            self.pred_logits.T.loc[filtered_meta_pred_true.index].T,
            predicted_class=predicted_class,
        )

        payload_dict = {
            "type": "lift",
            "data": {
                "points": lift_points_dict,
                "class_order": self.class_mappings,
            },
        }  #  ,  z.tolist(), "class_order": self.class_order
        return payload_dict
