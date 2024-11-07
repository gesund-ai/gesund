import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn

from gesund.core._utils import ValidationUtils, Statistics

class PlotDatasetStats:
    def __init__(self,meta):
        self.meta = meta 

    def meta_distributions(self):
        meta_counts = self._calculate_meta_distributions(self.meta)
        data_dict = {}
        data_dict["Validation"] = meta_counts
        payload_dict = {}
        payload_dict["type"] = "bar"
        payload_dict["data"] = data_dict
        return payload_dict



    def _calculate_meta_distributions(self, meta):
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
