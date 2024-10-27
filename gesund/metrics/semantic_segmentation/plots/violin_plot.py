import pickle

import pandas as pd 
import numpy as np

from ..metrics.coco_metrics import COCOMetrics

class PlotViolin:
    def __init__(self, class_mappings, ground_truth_dict=None, prediction_dict=None, artifacts_path=None, study_list=None):
        """
        Initialize the PlotViolin class with necessary parameters.

        This constructor initializes the class with mappings, ground truth, predictions, artifacts path, and study list.
        If a ground truth dictionary is provided, it also initializes the COCOMetrics object.

        :param class_mappings: A dictionary mapping class IDs to class names.
        :param ground_truth_dict: A dictionary containing ground truth data (default is None).
        :param prediction_dict: A dictionary containing prediction data (default is None).
        :param artifacts_path: Path to the artifacts file (default is None).
        :param study_list: A list of studies to analyze (default is None).
        """
        if ground_truth_dict:
            self.ground_truth_dict = ground_truth_dict
            self.prediction_dict = prediction_dict
            self.class_mappings = class_mappings
            self.coco_metrics = COCOMetrics(class_mappings)
            self.artifacts_path = artifacts_path
            self.study_list = study_list

    def violin_graph(self):
        """
        Generate violin plots for various metrics.

        This function attempts to load existing artifacts from a pickle file. If loading fails, it computes the
        artifacts using the ground truth and prediction dictionaries. It then constructs and returns a payload
        dictionary containing the IoU, FWIoU, accuracy, Dice coefficient, specificity, sensitivity, Kappa, 
        and AUC values as lists for plotting.

        :return: A dictionary containing metric types and their corresponding lists of values for plotting:
            - 'type' (str): The type of graph, in this case, "violin".
            - 'data' (dict): A dictionary containing metric names as keys and lists of corresponding values.
        """
        try:
            artifacts = self._load_pickle(self.artifacts_path)
        except:
            coco_metrics = COCOMetrics(self.class_mappings)
            artifacts = coco_metrics.create_artifacts(self.ground_truth_dict, self.prediction_dict)

        # TO DO: add imagewise APD in its function
        # Add in coco_metrics.py plots as well
        iou_series = pd.DataFrame.from_dict(artifacts["iou"]["imagewise_iou"], orient='index',
                                            columns=['imagewise_iou']).sort_values(by='imagewise_iou', ascending=True)
        fwiou_series = pd.DataFrame.from_dict(artifacts["fwiou"]["imagewise_fwiou"], orient='index',
                                            columns=['imagewise_fwiou']).sort_values(by='imagewise_fwiou', ascending=True)
        acc_series = pd.DataFrame.from_dict(artifacts["pAccs"]["imagewise_acc"], orient='index',
                                            columns=['imagewise_acc']).sort_values(by='imagewise_acc', ascending=True)
        

        dice_series = self.create_miseval_imagewise_df(artifacts, 'DSC')
        spec_series = self.create_miseval_imagewise_df(artifacts, 'Specificity')
        sens_series = self.create_miseval_imagewise_df(artifacts, 'Sensitivity')
        kapp_series = self.create_miseval_imagewise_df(artifacts, 'Kapp')
        auc_series = self.create_miseval_imagewise_df(artifacts, 'AUC')

        payload_dict = {"type": "violin",
                        "data": { "IoU": iou_series['imagewise_iou'].tolist(),
                                "FWIoU": fwiou_series['imagewise_fwiou'].tolist(),
                                "Acc": acc_series['imagewise_acc'].tolist(),
                                "Dice": dice_series['DSC'].tolist(),
                                "Spec": spec_series['Specificity'].tolist(),
                                "Sens": sens_series['Sensitivity'].tolist(),
                                "Kapp": kapp_series['Kapp'].tolist(),
                                "AUC": auc_series['AUC'].tolist()}}
                                                    
        return payload_dict



    def create_miseval_imagewise_df(self, artifacts, metric_name):
        """
        Create a DataFrame for image-wise evaluation metrics.

        This function constructs a DataFrame containing image IDs and their corresponding metric values
        for a specific metric. The DataFrame is sorted by the metric values.

        :param artifacts: A dictionary containing various artifacts generated from the ground truth and predictions.
        :param metric_name: The name of the metric to extract (e.g., 'DSC', 'Specificity').

        :return: A DataFrame with two columns: 'Image ID' and the specified metric name, sorted by the metric.
        """
        metric_values = [(img_id, metrics[metric_name]) for img_id, metrics in artifacts['misevals']['imagewise_metrics'].items()]
        df = pd.DataFrame(metric_values, columns=['Image ID', metric_name]).sort_values(metric_name)
        return df

    def _load_pickle(self, file_path):
        """
        Load data from a pickle file.

        This function opens a pickle file and loads its content, returning the data.

        :param file_path: The path to the pickle file to be loaded.

        :return: The data loaded from the pickle file.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)