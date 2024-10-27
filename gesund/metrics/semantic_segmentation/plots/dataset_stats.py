import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn

from gesund.utils.validation_data_utils import ValidationUtils, Statistics

class PlotDatasetStats:
    def __init__(self, meta):
        """
        Initialize the PlotDatasetStats object with metadata.

        This constructor takes in the metadata DataFrame which contains the data 
        needed for plotting and calculating statistics.

        :param meta: (pd.DataFrame) A DataFrame containing the metadata for the dataset.
        """
        self.meta = meta 

    def meta_distributions(self):
        """
        Generate metadata distributions for validation data.

        This function calculates the distributions of metadata and prepares the 
        data for visualization. It returns a dictionary structured for a bar chart.

        :return: (dict) A dictionary containing the validation metadata distributions, 
        formatted for plotting with keys:
            - 'type' (str): The type of chart ('bar').
            - 'data' (dict): A dictionary containing validation counts.
        """
        meta_counts = self._calculate_meta_distributions(self.meta)
        data_dict = {}
        data_dict["Validation"] = meta_counts
        payload_dict = {}
        payload_dict["type"] = "bar"
        payload_dict["data"] = data_dict
        return payload_dict



    def _calculate_meta_distributions(self, meta):
        """
        Calculate statistics on metadata.

        This helper function computes both histogram statistics for numerical columns 
        and value counts for categorical columns in the provided metadata DataFrame.

        :param meta: (pd.DataFrame) The metadata DataFrame to calculate distributions from.

        :return: (dict) A dictionary containing:
            - 'bar' (dict): Value counts for categorical columns.
            - 'histogram' (dict): Histograms for numerical columns, where keys are 
              column names and values are histogram data.
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
