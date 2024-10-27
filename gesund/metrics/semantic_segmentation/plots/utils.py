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

        This function filters the dataset based on multiple attributes provided in a dictionary.
        If no filtering criteria are provided, the original dataset is returned.

        :param target_attribute_dict: A dictionary where keys are attribute names and values are either:
            - A tuple of two numeric values specifying the range for numeric attributes.
            - A single value for categorical attributes.

        :return: A filtered DataFrame containing only the rows that match the specified criteria.
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
        Filters data by a single attribute.

        This function filters the dataset based on a single attribute provided in a dictionary.

        :param target_attribute_dict: A dictionary with one key-value pair where the key is the 
            attribute name and the value is either:
            - A tuple of two numeric values specifying the range for numeric attributes.
            - A single value for categorical attributes.

        :return: A filtered DataFrame containing only the rows that match the specified criteria.
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

        This function filters the dataset based on multiple attributes provided in a dictionary.

        :param target_attributes_dict: A dictionary where keys are attribute names and values are either:
            - A tuple of two numeric values specifying the range for numeric attributes.
            - A single value for categorical attributes.

        :return: A filtered DataFrame containing only the rows that match the specified criteria.
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
        """
        Retrieve the ordered list of classes.

        :return: A list of classes in the specified order.
        """
        return self.class_order

    def get_predict_categorical(self, id_name):
        """
        Get the categorical prediction for a specified ID.

        :param id_name: A string representing the ID of the item whose prediction is requested.

        :return: The categorical prediction associated with the specified ID.
        """
        return self.meta_pred_true.loc[id_name]["pred_categorical"]

    def get_meta_df(self):
        """
        Retrieve the metadata DataFrame.

        :return: A DataFrame containing the metadata.
        """
        return self.meta

    def get_entities(self):
        """
        Retrieve the list of entities from the metadata.

        :return: A list of entity names (column names) in the metadata DataFrame.
        """
        return self.meta.columns.tolist()

    def get_meta_with_results(self):
        """
        Get the metadata combined with results.

        :return: A copy of the DataFrame containing metadata along with prediction results.
        """
        return self.meta_pred_true.copy()

    # Typecheckers
    def is_list_numeric(self, x_list):
        """
        Check if all elements in a list are numeric.

        :param x_list: A list to check for numeric values.

        :return: True if all elements are numeric; False otherwise.
        """
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
