import pickle

import pandas as pd 
import numpy as np

from ..metrics.coco_metrics import COCOMetrics

class PlotViolin:
    def __init__(self, class_mappings, ground_truth_dict=None, prediction_dict=None, artifacts_path=None, study_list=None):
        if ground_truth_dict:
            self.ground_truth_dict = ground_truth_dict
            self.prediction_dict = prediction_dict
            self.class_mappings = class_mappings
            self.coco_metrics = COCOMetrics(class_mappings)
            self.artifacts_path = artifacts_path
            self.study_list = study_list

    def violin_graph(self):

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
        metric_values = [(img_id, metrics[metric_name]) for img_id, metrics in artifacts['misevals']['imagewise_metrics'].items()]
        df = pd.DataFrame(metric_values, columns=['Image ID', metric_name]).sort_values(metric_name)
        return df

    def _load_pickle(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)