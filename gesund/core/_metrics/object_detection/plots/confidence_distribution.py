from itertools import chain

import numpy as np
import pandas as pd
import random

from ..metrics.average_precision import AveragePrecision
from gesund.core._utils import ValidationUtils, Statistics

class PlotConfidenceGraphs:
    def __init__(self, class_mappings, coco_, meta_data_dict=None):
        self.class_mappings = class_mappings
        self.coco_ = coco_
        if bool(meta_data_dict):
            self.is_meta_exists = True
            self.validation_utils = ValidationUtils(meta_data_dict)

    def _plot_confidence_histogram_scatter_distribution(
        self, predicted_class=None, n_samples=300
    ):

        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )

        threshold = 0.1
        conf_points = average_precision.calculate_confidence_distribution(
            threshold, predicted_class=predicted_class
        )
        conf_points_list = []
        for neuron, value in conf_points.items():
            conf_points_list.append(
                {
                    "image_id": neuron,
                    "y": value,
                    "x": random.uniform(0, 1),
                    "labels": "TP",
                }
            )

        y_values = [d["y"] for d in conf_points_list]

        # Plot histogram
        histogram = Statistics.calculate_histogram(y_values, min_=0, max_=1, n_bins=10)

        payload_dict = {
            "type": "mixed",
            "data": {"points": conf_points_list, "histogram": histogram},
        }

        return payload_dict
