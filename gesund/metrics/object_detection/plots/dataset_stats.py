import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn

from gesund.utils.validation_data_utils import ValidationUtils, Statistics

class PlotDatasetStats:
    def __init__(self, class_mappings, meta_data_dict=None):
        """
        Initialize the PlotDatasetStats class.

        This class is responsible for calculating and plotting statistical distributions 
        based on the provided class mappings and metadata. It initializes the internal state 
        with class indexes and metadata existence checks.

        :param class_mappings: A dictionary mapping class names or IDs to their corresponding indexes.
        :param meta_data_dict: A dictionary containing metadata for validation or training, if available.
        """
        self.meta_data_dict = meta_data_dict
        if bool(meta_data_dict):
            self.is_meta_exists = True
        self.class_mappings = class_mappings
        self.class_idxs = [int(i) for i in list(class_mappings.keys())]

    def calculate_meta_distributions(self, meta):
        """
        Calculate statistics on metadata.

        This method computes histograms for numerical columns and bar charts for categorical columns 
        from the provided metadata DataFrame. It returns a dictionary summarizing the distributions.

        :param meta: A pandas DataFrame containing metadata, where each column represents a feature.
        
        :return: A dictionary containing the distributions, with keys:
            - 'bar' (dict): A dictionary of categorical columns with their value counts.
            - 'histogram' (dict): A dictionary of numerical columns with their histogram data.
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

    def _plot_meta_distributions(self):
        """
        Plot the distributions of metadata.

        This method creates a bar chart representation of the calculated metadata distributions 
        and prepares the data for visualization. The metadata is converted to a DataFrame format 
        if a metadata dictionary exists.

        :return: A dictionary containing the type of plot and the data for validation metadata.
            - 'type' (str): The type of plot (e.g., 'bar').
            - 'data' (dict): A dictionary summarizing the validation metadata distributions.
        """
        meta = pd.DataFrame(self.meta_data_dict).T
        meta_counts = self.calculate_meta_distributions(meta)
        data_dict = {}
        data_dict["Validation"] = meta_counts
        payload_dict = {}
        payload_dict["type"] = "bar"
        payload_dict["data"] = data_dict
        return payload_dict
