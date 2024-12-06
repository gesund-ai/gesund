import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn

from ..metrics.most_confused import MostConfused


class PlotMostConfused:
    def __init__(
        self,
        true,
        pred,
        pred_categorical,
        meta,
        meta_pred_true,
        loss,
        class_mappings,
    ):
        self.true = true
        self.pred = pred
        self.pred_categorical = pred_categorical
        self.meta = meta
        self.meta_pred_true = meta_pred_true
        self.loss = loss
        self.class_mappings = class_mappings

        self.most_confused = MostConfused(class_mappings)

    def most_confused_bar(self, top_k=5):
        """
        Calculates most confused classes on overall data.
        :param top_k: number of most confused relations to show.
        :return: payload_dict:
        """
        most_confused_dict = self.most_confused.calculate_most_confused(
            self.true, self.pred_categorical
        )
        most_confused_df = pd.DataFrame(most_confused_dict)
        most_confused_df[["true", "pred_categorical"]] = most_confused_df[
            ["true", "pred_categorical"]
        ].replace({int(k): v for k, v in self.class_mappings.items()})
        data = [
            {
                "True": row[1]["true"],
                "Prediction": row[1]["pred_categorical"],
                "count": row[1]["count"],
            }
            for row in most_confused_df.iterrows()
        ]

        payload_dict = {"type": "bar", "data": data}
        return payload_dict

    def most_confused_class_images(
        self,
        true_class,
        predicted_class,
        top_k=9,
        rank_mode="best_prediction_probability",
    ):
        """
        Filters most confused class samples, with three rank mode. It returns a dictionary of samples where true class
        is true_class, however, prediction for these samples are predicted_class.
        best_prediction_probability: Sorts with respect to the highest softmax value of predicted class.
        worst_true_probability: Sorts with respect to the lowest softmax value of true class.
        lose_by_small_margin: Sorts with respect min(true class softmax-predicted class softmax)
        where
        predicted class softmax>true class softmax.

        :param true_class: True class of samples
        :param predicted_class: Predicted class of samples
        :param top_k: Number of most confused samples.
        :param rank_mode: best_prediction_probability,worst_true_probability,lose_by_small_margin
        :return: payload_dict
        """
        image_ids_confused = self.meta_pred_true.loc[
            (self.true == true_class) & (self.pred_categorical == predicted_class)
        ].index
        best_pred_str = "best_prediction_probability"
        worst_true_str = "worst_true_probability"
        small_margin_str = "lose_by_small_margin"
        if rank_mode == best_pred_str:
            logits_ranked = self.pred.T.loc[image_ids_confused].sort_values(
                by=predicted_class, ascending=False
            )
        elif rank_mode == worst_true_str:
            logits_ranked = self.pred.T.loc[image_ids_confused].sort_values(
                by=true_class, ascending=True
            )
        elif rank_mode == small_margin_str:
            true_logits = self.pred.T.loc[image_ids_confused][true_class]
            predicted_logits = self.pred.T.loc[image_ids_confused][predicted_class]
            logits_ranked = self.pred.T.loc[
                (true_logits - predicted_logits).sort_values(ascending=False).index
            ]
        else:
            logits_ranked = self.pred.T.loc[image_ids_confused].sort_values(
                by=true_class, ascending=False
            )

        logits_ranked = logits_ranked.head(top_k)
        most_confused_data = []
        i = 1
        for idx in logits_ranked.index:
            most_confused_data.append(
                {
                    "image_id": idx,
                    "rank": i,
                    "Loss": float(self.loss[idx].item()),
                    "Ground Truth": {
                        "Class Name": self.class_mappings[str(true_class)],
                        "Confidence": float(logits_ranked.loc[idx][true_class]),
                    },
                    "Prediction": {
                        "Class Name": self.class_mappings[str(predicted_class)],
                        "Confidence": float(logits_ranked.loc[idx][predicted_class]),
                    },
                    "Meta Data": self.meta.loc[idx].to_dict(),
                }
            )
            i += 1
        payload_dict = {"type": "image", "data": most_confused_data}
        return payload_dict
