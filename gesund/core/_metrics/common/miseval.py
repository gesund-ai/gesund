from typing import Union, Optional
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager
from gesund.core._metrics.miseval.plot_miseval import PlotMisEval


class SemanticSegmentation:
    def _validate_data(self, data: dict) -> bool:
        """
        Validates the data required for metric calculation and plotting.

        :param data: The input data required for calculation, {"prediction":, "ground_truth": }
        :type data: dict

        :return: Status if the data is valid
        :rtype: bool
        """
        
        # commenting at the moment, to avoid subcohort metadata step, because it splitting the data 
        # and then giving final results seperately for so many image ids
        data["ground_truth"] = data["annotation"]
        '''# Basic validation checks
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        required_keys = ["prediction", "ground_truth"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Data must contain '{key}'.")

        if len(data["prediction"]) != len(data["ground_truth"]):
            raise ValueError("Prediction and ground_truth must have the same length.")

        # check for image ids samples in the ground truth and prediction
        if (
            len(
                set(list(data["prediction"].keys())).difference(
                    set(list(data["ground_truth"].keys()))
                )
            )
            > 0
        ):
            raise ValueError("Prediction and ground truth samples does not match.")'''

        return True

    def __preprocess(self, data: dict, get_logits=False) -> tuple:
        """
        Preprocesses the data

        :param data: dictionary containing the data prediction, ground truth
        :type data: dict
        :param get_logits: in case of multi class classification set to True
        :type get_logits: boolean

        :return: data tuple
        :rtype: tuple
        """      
        combined_data = []
        
        for image_id in data["prediction"]:
            pred_item = data["prediction"][image_id]
            annotation_item = data["ground_truth"][image_id]
            item_info_dict = {}
            item_info_dict["image_id"] = pred_item["image_id"]
            item_info_dict["shape"] = [
                pred_item["shape"][0],
                pred_item["shape"][1],
            ]
            '''metadata_row = data["metadata"][data["metadata"]["image_id"] == image_id]
            if not metadata_row.empty:
                item_info_dict["meta_data"] = metadata_row.iloc[0].to_dict()
            else:
                item_info_dict["meta_data"] = {}'''
            item_info_dict["meta_data"] = data["metadata"][image_id]["metadata"] if data["metadata"] else {}
            item_info_dict["ground_truth"] = annotation_item["annotation"]
            item_info_dict["objects"] = pred_item["masks"]
            item_info_dict["created_timestamp"] = time.time()
            combined_data.append(item_info_dict)
            
        prepped_data = dict()
        data_df = pd.DataFrame(combined_data)
        
        gt_dict = data_df[["image_id", "ground_truth", "shape"]].values
        ground_truth_dict = {}
        for image_id, ground_truth_list, shape_list in gt_dict:
            shape = shape_list
            rle_dict = {"rles": []}
            if len(ground_truth_list) > 1:
                for item in ground_truth_list:
                    label = item["label"]
                    rle = {
                        "rle": item["mask"]["mask"], 
                        "shape": shape, 
                        "class": label
                    }
                    rle_dict["rles"].append(rle)
                ground_truth_dict[image_id] = rle_dict
                
            else:
                ground_truth = ground_truth_list[0]
                label = ground_truth["label"]
                rles_str = ground_truth["mask"]["mask"]
                rles = {
                    "rles": [{
                        "rle": rles_str, 
                        "shape": shape, 
                        "class": label
                    }]
                }
                ground_truth_dict[image_id] = rles

        # Prediction dict
        pred_dict = data_df[["image_id", "objects", "shape"]].values
        prediction_dict = {}
        for image_id, objects, shape in pred_dict:
            rles = objects["rles"]
            for rle_dict in rles:
                rle_dict["shape"] = shape
            prediction_dict[image_id] = objects
        
        try:
            loss_dict = (
                data_df[["image_id", "loss"]]
                .set_index("image_id")
                .to_dict()["loss"]
            )
        except:
            pass
        
        # Metdata dict
        meta_data_dict = data_df[["image_id", "meta_data"]].values
        meta_data_dict = dict(zip(meta_data_dict[:, 0], meta_data_dict[:, 1]))
        
        prepped_data["ground_truth"] = ground_truth_dict
        prepped_data["prediction"] = prediction_dict        
        prepped_data["metadata"] = meta_data_dict
        prepped_data["class_mapping"] = data["class_mapping"]

          
        return prepped_data

    def __calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        """
        A function to calculate the metrics

        :param data: data dictionary containing data
        :type data: dict
        :param class_mapping: a dictionary with class mapping labels
        :type class_mapping: dict

        :return: results calculated
        :rtype: dict
        """
        
        preprocessed_data = self.__preprocess(data)
        
        self.plot_miseval = PlotMisEval(
            ground_truth_dict=preprocessed_data["ground_truth"],
            meta=preprocessed_data["metadata"],
            prediction_dict=preprocessed_data["prediction"],
            class_mappings=preprocessed_data["class_mapping"]
        )
        
        metrics = self.plot_miseval.highlighted_overall_metrics()

        return metrics

    def calculate(self, data: dict) -> dict:
        """
        Calculates the AUC metric for the given dataset.

        :param data: The input data required for calculation and plotting
                     {"prediction":, "ground_truth": , "class_mappings":}
        :type data: dict

        :return: Calculated metric results
        :rtype: dict
        """
        result = {}

        # Validate the data
        self._validate_data(data)

        # calculate results
        result["miseval"] = self.__calculate_metrics(data, data.get("class_mapping"))

        return result


class Classification:
    pass


class ObjectDetection:
    pass


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("semantic_segmentation.miseval")
def calculate_miseval_metrics(data: dict, problem_type: str):
    """
    A wrapper function to calculate the AUC metric.

    :param data: Dictionary of data: {"prediction": , "ground_truth": , 'metadata': , 'class_mapping': }
    :type data: dict
    :param problem_type: Type of the problem
    :type problem_type: str

    :return: Dict of calculated results
    :rtype: dict
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result
