import numpy as np
import pandas as pd

from ..metrics.auc import AUC
from ..metrics.dataset_stats import DatasetStats
from gesund.core._utils import ValidationUtils, Statistics


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
        Calculates true positive, true negative, false positive, false negatives for the given class.
        :param target_class: Class to calculate metrics.
        :param true: True labels for samples
        :param pred_categorical: Prediction for samples
        :return: payload_dict
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
        # Plot Scatters

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
        Calculates minimum loss samples with appropriate information.
        :param self:
        :param predicted_class: Class of interest to calculate top losses.
        :param top_k: Number of most loss samples
        :return: payload_dict
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
        Calculates statistics on classes.
        :param true: true labels as a list = [1,0,3,4] for 4 sample dataset
        :param pred_categorical: categorical predictions as a list = [1,0,3,4] for 4 sample dataset
        :param labels: order of classes inside list
        :return: confusion matrix
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
        meta_counts = self.dataset_stats.calculate_meta_distributions(self.meta)
        data_dict = {}
        data_dict["Validation"] = meta_counts
        payload_dict = {}
        payload_dict["type"] = "bar"
        payload_dict["data"] = data_dict
        return payload_dict

    def prediction_dataset_distribution(self, target_attribute_dict=None):
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