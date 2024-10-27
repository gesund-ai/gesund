import numpy as np
import pandas as pd

from ..metrics.auc import AUC
from ..metrics.dataset_stats import DatasetStats
from gesund.utils.validation_data_utils import ValidationUtils, Statistics


class PlotPredictionDataAnalysis:
    def __init__(
        self,
        true,
        pred_logits,
        pred_categorical,
        meta_pred_true,
        class_mappings,
        meta,
    ):
        self.true = true
        self.pred_logits = pred_logits
        self.pred_categorical = pred_categorical
        self.class_mappings = class_mappings
        self.meta_pred_true = meta_pred_true
        self.meta = meta

        self.dataset_stats = DatasetStats()
        self.auc = AUC(class_mappings)
        self.validation_utils = ValidationUtils(meta_pred_true=meta_pred_true)

    def prediction_distribution(self, target_attribute_dict=None):
        """
        Calculates the distribution of predictions for the specified target attributes.

        This function computes the occurrences of predicted classes based on 
        the provided target attributes. The results are formatted into a 
        payload dictionary for visualization.

        :param target_attribute_dict: (dict, optional) A dictionary of attributes 
            to filter the predictions by. If None, no filtering is applied.
        
        :return: A dictionary containing:
            - 'type' (str): The type of visualization (e.g., 'square').
            - 'data' (dict): A dictionary mapping class names to their occurrence counts.
        """
        filtered_meta_pred_true = self.validation_utils.filter_attribute_by_dict(
            target_attribute_dict
        )
        pred_categorical = filtered_meta_pred_true["pred_categorical"]

        occurrences = pred_categorical.value_counts().sort_index()
        occurrences.index = occurrences.index.map(str)
        occurrences = occurrences.rename(index=self.class_mappings)
        occurrences = occurrences.to_dict()

        payload_dict = {
            "type": "square",
            "data": occurrences,
        }
        return payload_dict

    def gtless_confidence_histogram_scatter_distribution(
        self, predicted_class="overall", n_samples=300, randomize_x=True, n_bins=25
    ):
        """
        Generates a scatter distribution and histogram for prediction confidence levels.

        This function filters the predictions based on the specified class 
        and visualizes the confidence levels as a scatter plot along with 
        a histogram of those confidence levels.

        :param predicted_class: (str) The class for which to filter predictions. 
            If 'overall', all predictions are included.
        :param n_samples: (int) The number of samples to include in the visualization.
        :param randomize_x: (bool) Whether to randomize the x-axis values.
        :param n_bins: (int) The number of bins for the histogram.

        :return: A dictionary containing:
            - 'type' (str): The type of visualization (e.g., 'mixed').
            - 'data' (dict): A dictionary with 'points' and 'histogram' for plotting.
        """
        # Filtering data
        filtered_pred_categorical = pd.DataFrame(self.pred_categorical.copy())
        if n_samples > filtered_pred_categorical.shape[0]:
            n_samples = filtered_pred_categorical.shape[0]
        filtered_pred_categorical = filtered_pred_categorical.sample(
            n_samples, replace=True
        )
        if predicted_class != "overall":
            filtered_pred_categorical = filtered_pred_categorical[
                filtered_pred_categorical["pred_categorical"] == predicted_class
            ]

        filtered_pred_logits = self.pred_logits[filtered_pred_categorical.index].max()
        filtered_pred_categorical["y"] = filtered_pred_logits

        # Renaming columns
        int_class_mappings = {int(k): v for k, v in self.class_mappings.items()}
        filtered_pred_categorical = filtered_pred_categorical.replace(
            {"pred_categorical": int_class_mappings}
        )
        filtered_pred_categorical = filtered_pred_categorical.rename(
            columns={"pred_categorical": "Prediction"}
        )

        # Randomize x if needed
        if randomize_x:
            filtered_pred_categorical["x"] = np.random.uniform(
                0, 1, filtered_pred_categorical.shape[0]
            )
        else:
            filtered_pred_categorical["x"] = filtered_pred_categorical["y"]

        points = list(
            filtered_pred_categorical.reset_index()
            .rename(columns={"index": "image_id"})
            .T.to_dict()
            .values()
        )

        # Plot histogram

        histogram = Statistics.calculate_histogram(
            filtered_pred_categorical["y"], min_=0, max_=1, n_bins=n_bins
        )

        payload_dict = {
            "type": "mixed",
            "data": {"points": points, "histogram": histogram},
        }

        return payload_dict

    def explore_predictions(self, predicted_class=None, top_k=3000):
        """
        Identifies the samples with the highest losses for further analysis.

        This function computes and organizes the samples with the highest 
        loss, allowing for examination of the model's predictions on these 
        challenging cases.

        :param predicted_class: (str, optional) The specific class to focus 
            on when calculating top losses. If None, evaluates overall losses.
        :param top_k: (int) The number of samples with the highest losses to return.

        :return: A dictionary containing:
            - 'type' (str): The type of visualization (e.g., 'image').
            - 'data' (list): A list of dictionaries containing information about 
              the top loss samples including image IDs, ranks, predictions, 
              confidences, and associated metadata.
        """

        # Check if overall top loss need to be observed.
        if predicted_class == "overall":
            predicted_class = None

        all_predictions = self.pred_categorical.copy(deep=True).head(top_k)
        predicted_logits = self.pred_logits.T.copy(deep=True)
        predicted_logits = predicted_logits.loc[all_predictions.index]
        meta = self.meta_pred_true[
            self.meta_pred_true.columns.difference(["pred_categorical", "true"])
        ]

        if predicted_class:
            logits_predicted_class_probability = predicted_logits[predicted_class]
        else:
            logits_predicted_class_probability = predicted_logits.max(axis=1)

        top_loss_data = []
        i = 1  # FIX.
        for idx in all_predictions.index:
            top_loss_data.append(
                {
                    "image_id": idx,
                    "rank": i,
                    "Prediction": self.class_mappings[
                        str(predicted_logits.loc[idx].idxmax())
                    ],
                    "Confidence": float(predicted_logits.loc[idx].max()),
                    "Meta Data": meta.loc[idx].to_dict(),
                }
            )
            i += 1
        payload_dict = {"type": "image", "data": top_loss_data}
        return payload_dict

    def softmax_probabilities_distribution(self, predicted_class, n_bins=25):
        """
        Computes and visualizes the distribution of softmax probabilities for a given class.

        This function calculates the histogram of softmax probabilities for the 
        specified predicted class and prepares it for visualization.

        :param predicted_class: (str) The class for which to compute the softmax probabilities.
        :param n_bins: (int) The number of bins for the histogram.

        :return: A dictionary containing:
            - 'type' (str): The type of visualization (e.g., 'scatter').
            - 'data' (dict): The histogram data for the specified class.
        """
        payload_dict = {
            "type": "scatter",
            "data": Statistics.calculate_histogram(
                array_=self.pred_logits.T[predicted_class],
                min_=0,
                max_=1,
                n_bins=n_bins,
            ),
        }
        return payload_dict

    def class_distributions(self):
        """
        Computes and returns the distributions of the predicted classes.

        This function gathers statistics on the true and predicted class labels, 
        providing counts of each class for validation.

        :return: A dictionary containing:
            - 'type' (str): The type of visualization (e.g., 'bar').
            - 'data' (dict): A dictionary with the counts of each class in 
              the validation dataset.
        """
        data = {}
        counts = self.dataset_stats.calculate_class_distributions(self.true)
        validation_counts = counts[0]
        validation_counts_renamed = {
            self.class_mappings[str(k)]: v for k, v in validation_counts.items()
        }

        payload_dict = {
            "type": "bar",
            "data": {"Validation": validation_counts_renamed},
        }
        return payload_dict

    def meta_distributions(self):
        """
        Analyzes and returns distributions of metadata associated with predictions.

        This function computes counts of various metadata attributes in the 
        validation dataset, providing insights into the characteristics of 
        the dataset used for predictions.

        :return: A dictionary containing:
            - 'type' (str): The type of visualization (e.g., 'bar').
            - 'data' (dict): A dictionary with counts of each metadata attribute.
        """
        meta_counts = self.dataset_stats.calculate_meta_distributions(self.meta)
        data_dict = {}
        data_dict["Validation"] = meta_counts
        payload_dict = {}
        payload_dict["type"] = "bar"
        payload_dict["data"] = data_dict
        return payload_dict

    def prediction_dataset_distribution(self, target_attribute_dict=None):
        """
        Computes and compares the distributions of true and predicted classes.

        This function analyzes the distributions of true labels and predicted 
        labels, providing a visual representation of the model's performance.

        :param target_attribute_dict: (dict, optional) A dictionary of attributes 
            to filter the predictions by. If None, no filtering is applied.

        :return: A dictionary containing:
            - 'type' (str): The type of visualization (e.g., 'bar').
            - 'data' (dict): A dictionary with distributions of true and predicted classes.
        """
        filtered_meta_pred_true = self.validation_utils.filter_attribute_by_dict(
            target_attribute_dict
        )

        # Dataset Distribution
        true_ = filtered_meta_pred_true["true"]
        data = {}
        dataset_class_counts = self.dataset_stats.calculate_class_distributions(true_)
        dataset_class_counts = dataset_class_counts[0]
        dataset_class_counts = {
            self.class_mappings[str(k)]: v for k, v in dataset_class_counts.items()
        }

        # Prediction Distribution
        pred_categorical = filtered_meta_pred_true["pred_categorical"]

        pred_class_counts = pred_categorical.value_counts().sort_index()
        pred_class_counts.index = pred_class_counts.index.map(str)
        pred_class_counts = pred_class_counts.rename(index=self.class_mappings)
        pred_class_counts = pred_class_counts.to_dict()

        payload_dict = {}
        payload_dict["type"] = "square"
        payload_dict["data"] = {
            "Annotation": dataset_class_counts,
            "Prediction": pred_class_counts,
        }
        return payload_dict
