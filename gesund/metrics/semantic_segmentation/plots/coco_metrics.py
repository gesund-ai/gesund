import pickle

import numpy as np
import pandas as pd

from ..metrics.coco_metrics import COCOMetrics
from gesund.utils.validation_data_utils import ValidationUtils, Statistics



class PlotCOCOMetrics:
    def __init__(
        self,
        class_mappings,
        meta,
        ground_truth_dict=None,
        prediction_dict=None,
        artifacts_path=None,
        study_list=None,
    ):
        """
        Initialize the PlotCOCOMetrics class.

        This constructor initializes the necessary attributes for calculating and plotting COCO metrics.

        :param class_mappings: A dictionary mapping class IDs to class names.
        :param meta: Metadata for the validation process.
        :param ground_truth_dict: (optional) A dictionary containing ground truth annotations.
        :param prediction_dict: (optional) A dictionary containing predictions.
        :param artifacts_path: (optional) Path to save or load artifacts.
        :param study_list: (optional) List of studies to filter metrics.
        """
        if ground_truth_dict:
            self.ground_truth_dict = ground_truth_dict
            self.prediction_dict = prediction_dict
            self.coco_metrics = COCOMetrics(class_mappings, study_list)
            self.study_list = study_list
            self.class_mappings = class_mappings
            self.validation_utils = ValidationUtils(meta)
            self.artifacts_path = artifacts_path

    def highlighted_overall_metrics(self):
        """
        Calculate highlighted overall metrics.

        This function calculates the overall metrics based on the ground truth and prediction dictionaries
        and formats the output for visualization.

        :return: A dictionary containing the overall metrics for visualization.
        """
        metrics_dict = self.coco_metrics.calculate_highlighted_overall_metrics(
            gt=self.ground_truth_dict, pred=self.prediction_dict
        )
        metrics_dict = {k: {"Validation": v} for k, v in metrics_dict.items()}
        payload_dict = {"type": "overall", "data": metrics_dict}
        return payload_dict
    
    def study_classbased_table(self, target_attribute_dict):
        """
        Generate a class-based metrics table for a specific study.

        This function filters the ground truth and prediction dictionaries based on the target study ID
        and computes the class-based metrics table.

        :param target_attribute_dict: A dictionary containing attributes to filter the studies.
        :return: A dictionary containing the class-based metrics table.
        """
        if target_attribute_dict:
            gt, pred = self.validation_utils._filter_dict_by_study(self.ground_truth_dict,
                                                                    self.prediction_dict,
                                                                    target_attribute_dict["study_id"])
            metrics_dict = self.coco_metrics.calculate_statistics_classbased_table(
                    gt=gt, pred=pred, target_attribute_dict=target_attribute_dict
                )
            
        else:
            metrics_dict = self.coco_metrics.calculate_statistics_classbased_table(
                gt=self.ground_truth_dict, pred=self.prediction_dict, target_attribute_dict=target_attribute_dict
            )
            
        payload_dict = {"type": "table", "data": {"Validation": metrics_dict}}
        
        return payload_dict
        
    def statistics_classbased_table(self, target_attribute_dict):
        """
        Generate a statistics table based on class metrics.

        This function calculates class-based statistics for the ground truth and predictions,
        filtering by target attributes if provided.

        :param target_attribute_dict: A dictionary containing attributes to filter the statistics.
        :return: A dictionary containing the statistics table.
        """
        if self.study_list:
            return {}
            
        if target_attribute_dict:

            filtered_image_rows = self.validation_utils.filter_attribute_by_dict(
                target_attribute_dict
            )
            filtered_image_ids = filtered_image_rows.index.tolist()

            gt = dict(
                (image_id, self.ground_truth_dict[image_id])
                for image_id in filtered_image_ids
                if image_id in self.ground_truth_dict
            )
            pred = dict(
                (image_id, self.prediction_dict[image_id])
                for image_id in filtered_image_ids
                if image_id in self.prediction_dict
            )
            
            metrics_dict = self.coco_metrics.calculate_statistics_classbased_table(
                gt=gt, pred=pred, target_attribute_dict=target_attribute_dict
            )

        else:
            metrics_dict = self.coco_metrics.calculate_statistics_classbased_table(
                gt=self.ground_truth_dict, pred=self.prediction_dict, target_attribute_dict=target_attribute_dict
            )
        
        payload_dict = {"type": "table", "data": {"Validation": metrics_dict}}
        return payload_dict

    
    def metrics_by_meta_data(self, target_attribute_dict=None):
        """
        Calculate metrics based on metadata attributes.

        This function filters the ground truth and prediction dictionaries based on metadata
        and computes various metrics, returning them in a structured format for visualization.

        :param target_attribute_dict: (optional) A dictionary containing attributes to filter by metadata.
        :return: A dictionary containing metrics for visualization.
        """
        # Filter by meta
        image_ids = self.validation_utils.filter_attribute_by_dict(
            target_attribute_dict
        ).index.tolist()

        try:
            artifacts = self._load_pickle(self.artifacts_path)
        except:
            coco_metrics = COCOMetrics(self.class_mappings)
            artifacts = self.coco_metrics.create_artifacts(
                self.ground_truth_dict, self.prediction_dict
            )

        imagewise_iou = artifacts["iou"]["imagewise_iou"]
        imagewise_pixel_accuracy = artifacts["pAccs"]["imagewise_acc"]
        imagewise_fwiou = artifacts["fwiou"]["imagewise_fwiou"]
        imagewise_misevals = artifacts["misevals"]["imagewise_metrics"]

        for image_id in imagewise_iou:
            imagewise_misevals[image_id]["iou"] = imagewise_iou[image_id]
            imagewise_misevals[image_id]["pAccs"] = imagewise_iou[image_id]
            imagewise_misevals[image_id]["fwiou"] = imagewise_iou[image_id]

        imagewise_metrics_df = pd.DataFrame(imagewise_misevals)
        imagewise_metrics_df = imagewise_metrics_df.T.loc[image_ids]
        # Delete repetitives
        imagewise_metrics_df = imagewise_metrics_df.drop(
            columns=[
                "IoU",
                "MCC_Normalized",
                "MCC_Absolute",
                "Precision_Sets",
                "Precision_CM",
                "Specificity_Sets",
                "Specificity_CM",
                "Sensitivity_Sets",
                "Sensitivity_CM",
                "IoU_Sets",
                "IoU_CM",
                "DSC_Enhanced",
                "DSC_Sets",
                "DSC_CM",
                "Accuracy",
                "Accuracy_Sets",
                "Accuracy_CM",
            ]
        )
        # Rename
        imagewise_metrics_df = imagewise_metrics_df.rename(
            columns={
                "iou": "meanIoU",
                "Kapp": "mean Kappa",
                "DSC": "Dice Score",
                "Sensitivity": "mean Sensitivity",
                "Specificity": "mean Specificity",
                "AUC": "mean AUC",
                "fwiou": "mean fwIoU",
                "pAccs": "Pixel Accuracy",
                "MCC": "Matthew's Classwise C C",
            }
        )

        metrics = imagewise_metrics_df.mean().to_dict()

        payload_dict = {"type": "bar", "data": {}}

        graph_1_metrics = ["TruePositive", "TrueNegative", "FalsePositive", "FalseNegative"]
        graph_2_metrics = ["AverageHausdorffDistance", "SimpleHausdorffDistance"]

        graph_1_dict = {key: int(metrics[key]) for key in graph_1_metrics if key in metrics}
        payload_dict["data"]["graph_1"] = graph_1_dict

        graph_2_dict = {key: metrics[key] for key in graph_2_metrics if key in metrics}
        payload_dict["data"]["graph_2"] = graph_2_dict

        graph_3_dict = {key: metrics[key] for key in metrics if key not in graph_1_metrics and key not in graph_2_metrics}
        payload_dict["data"]["graph_3"] = graph_3_dict   # Remaining metrics to graph_3 (specified by frontend)

        return payload_dict

    def main_metric(self):
        """
        Calculate the main validation metric.

        This function computes the main metric (mean IoU) based on the ground truth and predictions,
        returning it in a structured format for visualization.

        :return: A dictionary containing the main metric for visualization.
        """
        metrics_dict = (
            metrics_dict
        ) = self.coco_metrics.calculate_highlighted_overall_metrics(
            gt=self.ground_truth_dict, pred=self.prediction_dict
        )
        payload_dict = {
            "type": "main_metric",
            "data": {
                "mean IoU": metrics_dict["Vol.mIoU"] if self.study_list else metrics_dict["mean IoU"],
            },
        }
        return payload_dict

    def blind_spot_metrics(self, target_attribute_dict, threshold):
        """
        Calculate blind spot metrics for target attributes.

        This function computes class-based metrics and overall metrics for filtered ground truth and predictions,
        identifying potential blind spots based on the target attributes.

        :param target_attribute_dict: A dictionary containing attributes to filter the metrics.
        :param threshold: Threshold value to determine significant metrics.
        :return: A dictionary containing blind spot metrics for visualization.
        """
        filtered_image_ids = self.validation_utils.filter_attribute_by_dict(
            target_attribute_dict).index.tolist()

        filtered_gt = dict(
            (image_id, self.ground_truth_dict[image_id])
            for image_id in filtered_image_ids
            if image_id in self.ground_truth_dict
        )
        filtered_pred = dict(
            (image_id, self.ground_truth_dict[image_id])
            for image_id in filtered_image_ids
            if image_id in self.prediction_dict
        )

        class_metrics = self.coco_metrics.calculate_statistics_classbased_table(
            gt=filtered_gt, pred=filtered_pred
        )

        overall_metrics = self.coco_metrics.calculate_highlighted_overall_metrics(
            gt=filtered_gt, pred=filtered_pred
        )

        blind_spot_metrics_dict = {str(i): value 
                                   for i, value in enumerate(class_metrics.values(), 0)}

        blind_spot_metrics_dict["Average"] = overall_metrics
        
        return blind_spot_metrics_dict

    def create_artifacts(self, artifacts_path):
        """
        Create artifacts based on ground truth and prediction data.

        This function uses the COCO metrics object to generate artifacts from the 
        ground truth and prediction dictionaries. The artifacts are stored at the 
        specified path. If the artifact creation fails, a message is printed, 
        and the metrics values will need to be recalculated on each request.

        :param artifacts_path: (str) Path where the artifacts should be created.
        
        :return: (bool) Status indicating whether the artifact creation was successful.
        """
        status = False
        try:
            self.coco_metrics.create_artifacts(
                gt=self.ground_truth_dict,
                pred=self.prediction_dict,
                artifacts_path=artifacts_path,
            )
        except:
            print(
                f"Artifacts can't be created to, {artifacts_path}, metrics values will be for recalculated in each request"
            )
        return status

    def _load_pickle(self, file_path):
        """
        Load a pickle file from the specified file path.

        This helper function attempts to load a pickle file, returning its contents. 
        It opens the file in binary read mode and deserializes the object stored in it.

        :param file_path: (str) The path to the pickle file to load.

        :return: The contents of the pickle file.

        :raises FileNotFoundError: If the file at the specified path does not exist.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)
