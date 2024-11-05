import numpy as np
import pandas as pd

from ..metrics.top_losses import TopLosses


class PlotTopLosses:
    def __init__(self, pred, meta, meta_pred_true, loss, class_mappings):
        self.pred = pred
        self.loss = loss
        self.meta = meta
        self.meta_pred_true = meta_pred_true
        self.class_mappings = class_mappings

        self.top_losses_ = TopLosses(loss=self.loss, meta_pred_true=meta_pred_true)

    def top_losses(self, predicted_class=None, top_k=9):
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

        top_losses_class = self.top_losses_.calculate_top_losses(
            predicted_class, top_k
        ).head(top_k)
        top_loss_true_class = self.meta_pred_true.loc[top_losses_class.index]["true"]

        predicted_logits = self.pred.T.loc[top_losses_class.index]

        if predicted_class:
            logits_top_loss_predicted_class_probability = predicted_logits[
                predicted_class
            ]
        else:
            logits_top_loss_predicted_class_probability = predicted_logits.max(axis=1)

        top_loss_data = []
        i = 1  # FIX.
        for idx in top_losses_class.index:
            top_loss_data.append(
                {
                    "image_id": idx,
                    "rank": i,
                    "Ground Truth": self.class_mappings[
                        str(top_loss_true_class.loc[idx])
                    ],
                    "Prediction": self.class_mappings[
                        str(predicted_logits.loc[idx].idxmax())
                    ],
                    "Loss": float(top_losses_class.loc[idx]),
                    "Confidence": float(
                        logits_top_loss_predicted_class_probability.loc[idx]
                    ),
                    "Meta Data": self.meta.loc[idx].to_dict(),
                }
            )
            i += 1
        payload_dict = {"type": "image", "data": top_loss_data}
        return payload_dict
