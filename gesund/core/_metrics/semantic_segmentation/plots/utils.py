import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score


class PlotUtils:
    def filter_attribute_by_dict(self, target_attribute_dict=None):
        """
        Filters data by more than one attribute.
        """
        if bool(target_attribute_dict) != False:
            all_params = target_attribute_dict.keys()
            filtered_meta_pred_true = self.meta_pred_true.copy()
            for target_attribute in all_params:
                if self.is_list_numeric(self.meta_pred_true[target_attribute].values):
                    slider_min, slider_max = target_attribute_dict[target_attribute]
                    filtered_meta_pred_true = filtered_meta_pred_true[
                        self.meta_pred_true[target_attribute].between(
                            slider_min, slider_max
                        )
                    ]
                else:
                    target_value = target_attribute_dict[target_attribute]
                    filtered_meta_pred_true = filtered_meta_pred_true[
                        self.meta_pred_true[target_attribute] == target_value
                    ]
            return filtered_meta_pred_true
        else:
            return self.meta_pred_true

    def filter_attribute(self, target_attribute_dict):
        """
        Filters data by single attribute.
        """
        target_attribute = list(target_attribute_dict.keys())[0]
        target_value = target_attribute_dict[target_attribute]
        if self.is_list_numeric(self.meta_pred_true[target_attribute].values):
            slider_min, slider_max = target_attribute_dict[target_attribute]
            filtered_meta_pred_true = self.meta_pred_true[
                self.meta_pred_true[target_attribute].between(slider_min, slider_max)
            ]
        else:
            filtered_meta_pred_true = self.meta_pred_true[
                self.meta_pred_true[target_attribute] == target_value
            ]
        return filtered_meta_pred_true

    def multifilter_attribute(self, target_attributes_dict):
        """
        Filters data by more than one attribute.
        """
        all_params = target_attributes_dict.keys()
        filtered_meta_pred_true = self.meta_pred_true.copy()
        for target_attribute in all_params:
            if self.is_list_numeric(self.meta_pred_true[target_attribute].values):
                slider_min, slider_max = target_attributes_dict[target_attribute]
                filtered_meta_pred_true = filtered_meta_pred_true[
                    self.meta_pred_true[target_attribute].between(
                        slider_min, slider_max
                    )
                ]
            else:
                target_value = target_attributes_dict[target_attribute]
                filtered_meta_pred_true = filtered_meta_pred_true[
                    self.meta_pred_true[target_attribute] == target_value
                ]
        return filtered_meta_pred_true

    # Getters

    def get_classes(self):
        return self.class_order

    def get_predict_categorical(self, id_name):
        return self.meta_pred_true.loc[id_name]["pred_categorical"]

    def get_meta_df(self):
        return self.meta

    def get_entities(self):
        return self.meta.columns.tolist()

    def get_meta_with_results(self):
        return self.meta_pred_true.copy()

    # Typecheckers
    def is_list_numeric(self, x_list):
        return all(
            [
                isinstance(
                    i,
                    (
                        int,
                        float,
                        np.int,
                        np.int8,
                        np.int16,
                        np.int32,
                        np.float,
                        np.float16,
                        np.float32,
                        np.float64,
                    ),
                )
                for i in x_list
            ]
        )
