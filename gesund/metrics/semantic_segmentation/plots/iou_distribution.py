import random
import pickle 

import pandas as pd 
import numpy as np

from ..metrics.coco_metrics import COCOMetrics
from gesund.utils.validation_data_utils import ValidationUtils, Statistics

class PlotIoUDistribution:
    def __init__(self, class_mappings, ground_truth_dict=None, prediction_dict=None, artifacts_path=None, study_list=None):
        """
        Initialize the PlotIoUDistribution object.

        This constructor sets up the necessary parameters for calculating and plotting
        Intersection over Union (IoU) distributions based on provided ground truth and 
        prediction data, class mappings, and an optional artifacts path.

        :param class_mappings: (dict) A mapping of class IDs to class names.
        :param ground_truth_dict: (dict, optional) A dictionary containing ground truth data.
        :param prediction_dict: (dict, optional) A dictionary containing prediction data.
        :param artifacts_path: (str, optional) Path to the artifacts file to load IoU data.
        :param study_list: (list, optional) A list of studies associated with the dataset.
        """
        if ground_truth_dict:
            self.ground_truth_dict = ground_truth_dict
            self.prediction_dict = prediction_dict
            self.class_mappings = class_mappings
            self.coco_metrics = COCOMetrics(class_mappings)
            self.artifacts_path = artifacts_path
            self.study_list = study_list

    def iou_distribution(self, n_samples=300):
        """
        Calculate and return the IoU distribution.

        This method computes the Intersection over Union (IoU) values for the predictions
        and ground truth data. It samples a specified number of IoU values, generates 
        random points for visualization, and prepares a histogram of the IoU distribution.

        :param n_samples: (int, optional) The number of IoU samples to return; defaults to 300.

        :return: (dict) A dictionary containing:
            - 'type' (str): The type of plot ('mixed').
            - 'data' (dict): A dictionary containing:
                - 'points' (list): A list of dictionaries with 'image_id', 'x', and 'y' coordinates.
                - 'histogram' (dict): The histogram data representing the IoU distribution.
        """
        try:
            artifacts = self._load_pickle(self.artifacts_path)
        except:
            coco_metrics = COCOMetrics(self.class_mappings)
            artifacts = coco_metrics.create_artifacts(self.ground_truth_dict, self.prediction_dict)

        iou_series = pd.Series(artifacts["iou"]["imagewise_iou"]).sort_values()
        iou_series = iou_series.sample(min(n_samples,iou_series.size)).to_dict()

        points = []

        for image_id in iou_series:
            iou_point = {
                    "image_id": image_id,
                    "x": random.uniform(0,1),

                    "y": float(iou_series[image_id]),
                }
            points.append(iou_point)

        # Plot Histogram
        histogram = Statistics.calculate_histogram(pd.Series(iou_series.values()), min_=0, max_=1, n_bins=10
        )
            
        payload_dict =payload_dict = {"type": "mixed","data": {"points": points, "histogram": histogram}}

        return payload_dict

    def _load_pickle(self, file_path):
        """
        Load a pickle file.

        This helper method reads a pickle file from the given file path and returns the 
        deserialized content.

        :param file_path: (str) The path to the pickle file to be loaded.

        :return: (any) The content of the pickle file, deserialized.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)
