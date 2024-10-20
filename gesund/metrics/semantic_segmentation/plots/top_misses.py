import pandas as pd
import numpy as np
import pickle
import math
import os

from ..metrics import COCOMetrics

class PlotTopMisses:
    def __init__(self, ground_truth_dict, prediction_dict, class_mappings, artifacts_path, study_list=None):
        self.ground_truth_dict = ground_truth_dict
        self.prediction_dict = prediction_dict
        self.class_mappings=class_mappings
        self.artifacts_path = artifacts_path
        self.study_list = study_list

    def top_misses(self, metric, sort_by, top_k=150):
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
        with open(file_path, "rb") as f:
            return pickle.load(f)
