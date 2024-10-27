from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from gesund.utils.validation_data_utils import ValidationUtils, Statistics


class PlotObjectStats:
    def __init__(self, coco_, class_mappings, meta_data_dict=None):
        """
        Initialize the PlotObjectStats class.

        This class is responsible for plotting statistics related to object detection using COCO format data.
        It initializes with class mappings and optional metadata to assist in validation and visualization.

        :param coco_: A list containing COCO formatted data where the first element is predictions 
                      and the second element contains annotations.
        :param class_mappings: A dictionary mapping class names or IDs to their corresponding indexes.
        :param meta_data_dict: A dictionary containing metadata for validation or training, if available.
        """
        self.is_meta_exists = False
        self.coco_ = coco_

        if bool(meta_data_dict):
            self.is_meta_exists = True
            self.validation_utils = ValidationUtils(meta_data_dict)
        self.validation_utils = ValidationUtils(meta_data_dict)
        self.class_mappings = class_mappings
        self.class_idxs = [int(i) for i in list(class_mappings.keys())]

    def _plot_object_counts(self, confidence=0, target_attribute_dict=None):
        """
        Plot object counts for ground truth and predictions.

        This method filters the dataset based on the specified target attributes, counts the occurrences 
        of each class in the ground truth and predicted annotations, and formats the result for visualization.

        :param confidence: A float representing the confidence threshold for including predictions.
        :param target_attribute_dict: A dictionary for filtering attributes (optional).
        
        :return: A dictionary formatted for bar chart visualization containing class occurrence counts.
            - 'type' (str): The type of plot (e.g., 'bar').
            - 'data' (dict): A dictionary with class names as keys and counts of predicted and ground truth 
              occurrences as values.
        """
        # Filter wrt target attribute dict
        idxs = self.validation_utils.filter_attribute_by_dict(
            target_attribute_dict
        ).index.tolist()

        # Counting the groundtruth
        gt_df = pd.DataFrame(self.coco_[1]["annotations"])
        gt_df = gt_df[gt_df["image_id"].isin(idxs)]

        gt_class_occurrence_df = (
            gt_df.groupby("image_id")["category_id"].value_counts().unstack().fillna(0)
        )
        gt_non_occurring_classes = list(
            set([int(i) for i in self.class_mappings.keys()])
            - set(gt_class_occurrence_df.columns.tolist())
        )
        for class_ in gt_non_occurring_classes:
            gt_class_occurrence_df[class_] = 0

        # Counting the predictions
        pred_df = pd.DataFrame(self.coco_[0])
        pred_df = pred_df[pred_df["score"] > confidence]
        pred_df = pred_df[pred_df["image_id"].isin(idxs)]

        pred_class_occurrence_df = (
            pred_df.groupby("image_id")["category_id"]
            .value_counts()
            .unstack()
            .fillna(0)
        )
        non_occurring_classes = list(
            set([int(i) for i in self.class_mappings.keys()])
            - set(pred_class_occurrence_df.columns.tolist())
        )
        for class_ in non_occurring_classes:
            pred_class_occurrence_df[class_] = 0

        # Sum Occurrences
        pred_count = pred_class_occurrence_df.sum().to_dict()
        gt_count = gt_class_occurrence_df.sum().to_dict()

        result_dict = dict()
        # Format them for frontend
        for cls_ in self.class_idxs:
            result_dict[self.class_mappings[str(cls_)]] = {
                "Predicted": pred_count[int(cls_)],
                "GT": gt_count[int(cls_)],
            }

        payload_dict = {
            "type": "bar",
            "data": result_dict,
        }
        return payload_dict

    def _plot_prediction_distribution(self, target_attribute_dict=None):
        """
        Plot the distribution of predictions across classes.

        This method filters the dataset based on the specified target attributes, counts the occurrences 
        of each predicted class, and formats the result for pie chart visualization.

        :param target_attribute_dict: A dictionary for filtering attributes (optional).

        :return: A dictionary formatted for pie chart visualization containing class occurrence counts.
            - 'type' (str): The type of plot (e.g., 'pie').
            - 'data' (dict): A dictionary with class names as keys and counts of predicted occurrences as values.
        """
        idxs = self.validation_utils.filter_attribute_by_dict(
            target_attribute_dict
        ).index.tolist()

        # Counting the predictions
        pred_df = pd.DataFrame(self.coco_[0])
        pred_df = pred_df[pred_df["image_id"].isin(idxs)]

        pred_class_occurrence_df = (
            pred_df.groupby("image_id")["category_id"]
            .value_counts()
            .unstack()
            .fillna(0)
        )
        non_occurring_classes = list(
            set([int(i) for i in self.class_mappings.keys()])
            - set(pred_class_occurrence_df.columns.tolist())
        )
        for class_ in non_occurring_classes:
            pred_class_occurrence_df[class_] = 0

        # Sum Occurrences
        pred_class_occurrence_df.columns = pred_class_occurrence_df.columns.astype(str)
        pred_class_occurrence_df.rename(columns=self.class_mappings, inplace=True)

        pred_class_occurrence_dict = pred_class_occurrence_df.sum().to_dict()

        payload_dict = {
            "type": "pie",
            "data": pred_class_occurrence_dict,
        }

        return payload_dict
