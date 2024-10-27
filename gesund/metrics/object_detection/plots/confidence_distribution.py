from itertools import chain

import numpy as np
import pandas as pd
import random

from ..metrics.average_precision import AveragePrecision
from gesund.utils.validation_data_utils import ValidationUtils, Statistics

class PlotConfidenceGraphs:
    def __init__(self, class_mappings, coco_, meta_data_dict=None):
        """
        Initialize the PlotConfidenceGraphs class.

        This class is responsible for plotting confidence graphs using provided class mappings and COCO data. 
        It also initializes validation utilities if metadata is provided.

        :param class_mappings: A dictionary mapping class names to their corresponding IDs.
        :param coco_: A COCO dataset object or related structure used for validation.
        :param meta_data_dict: A dictionary containing metadata for validation, if available.
        """
        self.class_mappings = class_mappings
        self.coco_ = coco_
        if bool(meta_data_dict):
            self.is_meta_exists = True
            self.validation_utils = ValidationUtils(meta_data_dict)

    def _plot_confidence_histogram_scatter_distribution(
        self, predicted_class=None, n_samples=300
    ):
        """
        Plot a histogram and scatter distribution of confidence scores.

        This method calculates the confidence distribution for predicted classes and creates a histogram 
        of confidence scores. It randomly assigns x-coordinates to the true positives for plotting.

        :param predicted_class: (str, optional): The specific predicted class to analyze. 
            If None, all classes are considered.
        :param n_samples: (int): The number of samples to consider for the scatter distribution.
        
        :return: A dictionary containing the plotted data, which includes:
            - 'type' (str): The type of plot (e.g., 'mixed').
            - 'data' (dict): Contains:
                - 'points' (list): A list of dictionaries with 'image_id', 'y' (confidence value), 'x' (random x-coordinate), 
                  and 'labels' indicating true positives.
                - 'histogram' (dict): The histogram data of confidence values.
        """
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
