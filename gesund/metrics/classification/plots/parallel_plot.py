import numpy as np
import pandas as pd


class PlotParallel:
    def __init__(self, meta_pred_true):
        """
        Initialize the PlotParallel class with metadata and predictions.

        :param meta_pred_true: (pd.DataFrame) A DataFrame containing the metadata, predicted classes, and true classes.
        """
        self.meta_pred_true = meta_pred_true

    def parallel_categorical_analysis(self, true_class):
        """
        Perform categorical analysis for the specified true class.

        This function filters the metadata to include only samples that belong to the specified true class.
        It identifies categorical variables within the metadata and returns a JSON representation of the filtered data.

        :param true_class: (any) The true class label for which to filter the samples and perform analysis.

        :return: A dictionary containing the type of analysis and the filtered data in JSON format, including:
            - 'type' (str): Type of the analysis, which is "parallel".
            - 'data' (str): A JSON string representation of the filtered metadata for the specified true class,
              including the categorical predictions and true values.
        """
        # Define categorical variables.
        metas_list = list(self.meta_pred_true.columns)
        metas_list.remove("pred_categorical")
        metas_list.remove("true")

        for clm in metas_list:
            if type(self.meta_pred_true.iloc[0][clm]) != str:
                metas_list.remove(clm)
        # Filter true class
        filtered_meta_pred_true = self.meta_pred_true[
            self.meta_pred_true["true"] == true_class
        ]
        # Anonymize ids (We may remove it for filtering.)
        filtered_meta_pred_true.index = np.arange(len(filtered_meta_pred_true.index))
        return {
            "type": "parallel",
            "data": filtered_meta_pred_true[
                [*metas_list, "pred_categorical", "true"]
            ].to_json(),
        }
