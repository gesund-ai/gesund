import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn

from ..metrics import LiftChart
from gesund.utils.validation_data_utils import ValidationUtils, Statistics


class PlotLiftGainChart:
    def __init__(self, true, pred_logits, meta, class_mappings):
        self.class_mappings = class_mappings
        self.class_order = [int(i) for i in list(class_mappings.keys())]

        self.true = true
        self.pred_logits = pred_logits
        self.lift_chart_calculate = LiftChart(class_mappings)
        self.validation_utils = ValidationUtils(meta)

    def lift_chart(self, target_attribute_dict, predicted_class=None):
        """
        Calculates confusion matrix given true/predicted labels.
        :param true: True labels for samples
        :param pred_categorical: Prediction for samples
        :return:
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
