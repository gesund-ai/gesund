import pandas as pd
import numpy as np
import pickle
import math
import os

from ..metrics import COCOMetrics

class PlotTopMisses:
    def __init__(self, ground_truth_dict, prediction_dict, class_mappings, artifacts_path, study_list=None):
        """
        Initialize the PlotTopMisses class.

        This class is responsible for plotting top misses based on the given ground truth and prediction data.

        :param ground_truth_dict: A dictionary containing ground truth data for evaluation.
        :param prediction_dict: A dictionary containing prediction data for evaluation.
        :param class_mappings: A mapping of class names to their corresponding IDs.
        :param artifacts_path: Path to save or load artifacts.
        :param study_list: An optional list of studies to include in the analysis.
        """
        self.ground_truth_dict = ground_truth_dict
        self.prediction_dict = prediction_dict
        self.class_mappings=class_mappings
        self.artifacts_path = artifacts_path
        self.study_list = study_list

    def top_misses(self, metric, sort_by, top_k=150):
        """
        Retrieve the top misses based on a specified metric.

        This function calculates the top K images with the highest misses for a specified metric, 
        either sorting in ascending or descending order as specified by the user.

        :param metric: A string representing the metric to evaluate (e.g., 'IoU', 'Accuracy').
        :param sort_by: A string indicating the sort order ('Ascending' or 'Descending').
        :param top_k: An integer indicating the number of top misses to return (default is 150).

        :return: A dictionary containing the top misses data with the following structure:
            - 'type' (str): Type of data (should always be 'image').
            - 'data' (list): A list of dictionaries with keys 'image_id', 'rank', and the specified metric.
        """
        try:
            artifacts = self._load_pickle(self.artifacts_path)
        except:
            coco_metrics = COCOMetrics(self.class_mappings)
            artifacts = coco_metrics.create_artifacts(self.ground_truth_dict, self.prediction_dict)

        additional_metrics = artifacts.keys()
        top_misses_data = []
        i = 1  # FIX.

        if metric in additional_metrics:
            for image_id, value in pd.DataFrame(
                artifacts['{}'.format(metric)]).sort_values('imagewise_{}'.format(metric), ascending=(sort_by == "Ascending")
                                                            ).head(top_k).to_dict()['imagewise_{}'.format(metric)].items():

                top_loss_single_image = {
                        "image_id": image_id,
                        "rank": i,
                        metric: value,
                }
                top_misses_data.append(top_loss_single_image)
                i += 1
        else:
            for image_id, value in pd.DataFrame(artifacts['misevals']['imagewise_metrics']).T.sort_values(metric,ascending=(sort_by == "Ascending")).head(top_k)[metric].to_dict().items():
                top_loss_single_image = {
                        "image_id": image_id,
                        "rank": i,
                        metric: value,
                    }
                top_misses_data.append(top_loss_single_image)
                i += 1

        payload_dict = {"type": "image", "data": top_misses_data}
        return payload_dict

    def _load_pickle(self, file_path):
        """
        Load a pickle file from the specified path.

        This utility function reads a pickle file and returns its content.

        :param file_path: A string representing the path to the pickle file to be loaded.
        
        :return: The content of the pickle file, typically a dictionary or data structure saved in it.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)
