import numpy as np
import pandas as pd


class PlotParallel:
    def __init__(self, meta_pred_true):
        self.meta_pred_true = meta_pred_true

    def parallel_categorical_analysis(self, true_class):
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
