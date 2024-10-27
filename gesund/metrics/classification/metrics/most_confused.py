import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn

from .confusion_matrix import ConfusionMatrix


class MostConfused:
    def __init__(self, class_mappings):
        """
        Initialize the MostConfused class.

        This class is responsible for calculating confusion metrics among classes based on the provided mappings.

        :param class_mappings: A dictionary mapping class indices to class names.
        """
        self.class_mappings = class_mappings
        self.class_order = [int(i) for i in list(class_mappings.keys())]

        self.confusion_matrix = ConfusionMatrix(class_mappings=class_mappings)

    def calculate_most_confused(self, true, pred_categorical):
        """
        Calculate the most confused classes in the dataset.

        This function computes the confusion matrix and identifies the pairs of classes that are most frequently confused 
        with each other based on the predictions.

        :param true: A list or array of true class labels.
        :param pred_categorical: A list or array of predicted class labels.
        
        :return: A dictionary containing the most confused class pairs and their counts, with the following keys:
            - 'true' (int): The true class index of the pair.
            - 'pred_categorical' (int): The predicted class index of the pair.
            - 'count' (int): The number of times this confusion occurred.
        """
        confusion_matrix = self.confusion_matrix.calculate_confusion_matrix(
            true=true, pred_categorical=pred_categorical
        )
        confused_list_idxs = np.array([])
        confused_list_values = np.array([])
        for x in range(np.shape(confusion_matrix)[0]):
            for y in range(np.shape(confusion_matrix)[0]):
                if x != y:
                    confused_list_idxs = np.append(
                        confused_list_idxs, [self.class_order[x], self.class_order[y]]
                    )
                    confused_list_values = np.append(
                        confused_list_values, confusion_matrix[x, y]
                    )
        confused_list_idxs = confused_list_idxs.reshape(-1, 2)

        most_confused_df = pd.DataFrame(
            np.hstack((confused_list_idxs, np.reshape(confused_list_values, (-1, 1)))),
            columns=["true", "pred_categorical", "count"],
        )
        most_confused_df = most_confused_df.sort_values(by="count", ascending=False)
        most_confused_df.index = np.arange(len(most_confused_df.index))
        # Remove never confused samples
        most_confused_df = most_confused_df[most_confused_df["count"] > 0]
        return most_confused_df.to_dict()
