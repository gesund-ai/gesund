import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn
from typing import Dict, List, Optional, Union, Any
from gesund.core._utils import ValidationUtils, Statistics


class PlotDatasetStats:
    def __init__(self, class_mappings, meta_data_dict=None):
        self.meta_data_dict = meta_data_dict
        if bool(meta_data_dict):
            self.is_meta_exists = True
        self.class_mappings = class_mappings
        self.class_idxs = [int(i) for i in list(class_mappings.keys())]

    def calculate_meta_distributions(
        self, 
        meta: pd.DataFrame
        ) -> Dict[str, Dict[str, Any]]:
        """
        Calculates statistics on meta data.
        :param true: true labels as a list = [1,0,3,4] for 4 sample dataset
        :param pred_categorical: categorical predictions as a list = [1,0,3,4] for 4 sample dataset
        :param labels: order of classes inside list
        :return: dict that contains class dist. for validation/train dataset.
        """
        # Histogram charts for numerical values
        numerical_columns = [
            column
            for column in meta.columns
            if ValidationUtils.is_list_numeric(meta[column].values.tolist())
        ]

        histograms = {
            numerical_column: Statistics.calculate_histogram(
                meta[numerical_column],
                min_=meta[numerical_column].min(),
                max_=meta[numerical_column].max(),
                n_bins=10,
            )
            for numerical_column in numerical_columns
        }

        # Bar charts for categorical values

        categorical_columns = list(set(meta.columns) - set(numerical_columns))
        bars = {
            categorical_column: meta[categorical_column].value_counts().to_dict()
            for categorical_column in categorical_columns
        }

        return {"bar": bars, "histogram": histograms}

    def _plot_meta_distributions(self) -> Dict[str, Any]:
        """
        Plot distributions of metadata attributes.

        :return: Dictionary containing plot configuration and data
        :rtype: Dict[str, Any]
        """
        meta = pd.DataFrame(self.meta_data_dict).T
        meta_counts = self.calculate_meta_distributions(meta)
        data_dict = {}
        data_dict["Validation"] = meta_counts
        payload_dict = {}
        payload_dict["type"] = "bar"
        payload_dict["data"] = data_dict
        return payload_dict
