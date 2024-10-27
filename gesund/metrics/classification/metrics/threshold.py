import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score

from .auc import AUC


class ThresholdMetrics:
    def __init__(self, true, pred_logits, class_mappings):
        """
        Initialize the ThresholdMetrics class.

        This constructor initializes the class with the true labels, predicted logits, and class mappings.

        :param true: Array-like of true labels for the samples.
        :param pred_logits: Array-like of predicted logits for the samples.
        :param class_mappings: A dictionary mapping class indices to class names.
        """
        self.pred_logits = pred_logits
        self.true = true
        self.class_mappings = class_mappings

        self.auc = AUC(class_mappings=class_mappings)

    def calculate_statistics_per_class(self, true, pred_categorical):
        """
        Calculates class-wise statistics for the given class.

        This function computes various metrics such as F1 score, sensitivity, specificity, 
        precision, and others for a specified class based on true labels and predicted categorical values.

        :param true: Array-like of true labels for samples.
        :param pred_categorical: Array-like of predicted categorical labels for samples.
        
        :return: A dictionary containing:
            - 'type' (str): Type of graph to be generated, currently set to "bar".
            - 'data' (dict): Contains metrics for two graphs:
                - 'graph_1' (dict): Metrics including F1, Sensitivity, Specificity, Precision, 
                                   Matthew's Classwise C C, FPR, FNR.
                - 'graph_2' (dict): Counts of True Positives (TP), True Negatives (TN), 
                                   False Positives (FP), and False Negatives (FN).
        """
        sense_spec_dict = self.auc.calculate_sense_spec(true, pred_categorical)

        F1 = sense_spec_dict["F1"][1]
        TPR = sense_spec_dict["TPR"][1]
        TNR = sense_spec_dict["TNR"][1]
        PPV = sense_spec_dict["PPV"][1]
        NPV = sense_spec_dict["NPV"][1]
        FPR = sense_spec_dict["FPR"][1]
        FNR = sense_spec_dict["FNR"][1]
        TP = sense_spec_dict["TP"][1]
        TN = sense_spec_dict["TN"][1]
        FP = sense_spec_dict["FP"][1]
        FN = sense_spec_dict["FN"][1]
        MCC = sense_spec_dict["Matthew's Classwise C C"][1]

        payload_dict = {
            "type": "bar",
            "data": {
                "graph_1": {
                    "F1": F1,
                    "Sensitivity": TPR,
                    "Specificity": TNR,
                    "Precision": PPV,
                    "Matthew's Classwise C C": MCC,
                    "FPR": FPR,
                    "FNR": FNR,
                },
                "graph_2": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
            },
        }
        return payload_dict
