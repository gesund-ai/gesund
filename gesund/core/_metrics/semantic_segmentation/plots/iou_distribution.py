import random
import pickle

import pandas as pd
import numpy as np

from ..metrics.coco_metrics import COCOMetrics
from gesund.core._utils import ValidationUtils, Statistics


class PlotIoUDistribution:
    def __init__(
        self,
        class_mappings,
        ground_truth_dict=None,
        prediction_dict=None,
        artifacts_path=None,
        study_list=None,
    ):
        if ground_truth_dict:
            self.ground_truth_dict = ground_truth_dict
            self.prediction_dict = prediction_dict
            self.class_mappings = class_mappings
            self.coco_metrics = COCOMetrics(class_mappings)
            self.artifacts_path = artifacts_path
            self.study_list = study_list

    def iou_distribution(self, n_samples=300):
        try:
            artifacts = self._load_pickle(self.artifacts_path)
        except:
            coco_metrics = COCOMetrics(self.class_mappings)
            artifacts = coco_metrics.create_artifacts(
                self.ground_truth_dict, self.prediction_dict
            )

        iou_series = pd.Series(artifacts["iou"]["imagewise_iou"]).sort_values()
        iou_series = iou_series.sample(min(n_samples, iou_series.size)).to_dict()

        points = []

        for image_id in iou_series:
            iou_point = {
                "image_id": image_id,
                "x": random.uniform(0, 1),
                "y": float(iou_series[image_id]),
            }
            points.append(iou_point)

        # Plot Histogram
        histogram = Statistics.calculate_histogram(
            pd.Series(iou_series.values()), min_=0, max_=1, n_bins=10
        )

        payload_dict = payload_dict = {
            "type": "mixed",
            "data": {"points": points, "histogram": histogram},
        }

        return payload_dict

    def _load_pickle(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
