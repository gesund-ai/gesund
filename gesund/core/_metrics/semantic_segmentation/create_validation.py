from typing import Any, Dict, List, Optional
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from .plots.plot_driver import SemanticSegmentationPlotDriver
from gesund.core._metrics.semantic_segmentation.segmentation_metric_plot import (
    Semantic_Segmentation_Plot,
)


class ValidationCreation:
    """
    Class responsible for creating validation data and generating semantic segmentation metrics and plots.

    This class handles the creation of validation collections and metrics for semantic segmentation.

    :param batch_job_id: (str) Unique identifier for the batch job.
    :param filter_field: (str, optional) The field to filter the data. Defaults to "image_url".
    :param generate_metrics: (bool, optional) Whether to generate metrics during validation creation. Defaults to True.

    Attributes:
        batch_job_id (str): Unique identifier for the batch job.
        filter_field (str): The field to filter the data.
        generate_metrics (bool): Whether to generate metrics during validation creation.
    """
    
    def __init__(self, batch_job_id, filter_field="image_url", generate_metrics=True):
        """
        Initialize the ValidationCreation class.

        :param batch_job_id: (str) Identifier for the batch job.
        :param filter_field: (str, optional) Field used for filtering the data. Defaults to "image_url".
        :param generate_metrics: (bool, optional) Flag to control whether metrics should be generated. Defaults to True.
        """
        self.batch_job_id = batch_job_id
        self.filter_field = filter_field
        self.generate_metrics = generate_metrics

    def create_validation_collection_data(self, successful_batch_data, annotation_data, format=None, meta_data=None):
        """
        Create a list of validation collection data from batch and annotation data.

        :param successful_batch_data: (dict) Dictionary containing data for successfully processed images.
        :param annotation_data: (dict) Dictionary containing annotation data for the images.
        :param format: (str) Format of the annotations (e.g., 'coco').
        :param meta_data: (dict, optional) Additional metadata for each image. Defaults to None.

        :return: (list) A list of dictionaries with validation data for each image.
        """
        validation_collection_data = []

        for image_id in successful_batch_data:
            batch_item = successful_batch_data[image_id]
            annotation_item = annotation_data[image_id]
            image_information_dict = {}
            image_information_dict["batch_job_id"] = self.batch_job_id
            image_information_dict["image_id"] = batch_item["image_id"]
            image_information_dict["shape"] = [
                batch_item["shape"][0],
                batch_item["shape"][1],
            ]
            image_information_dict["meta_data"] = (
                meta_data[image_id]["metadata"] if meta_data else {}
            )
            image_information_dict["ground_truth"] = annotation_item["annotation"]
            image_information_dict["objects"] = batch_item["masks"]
            image_information_dict["created_timestamp"] = time.time()
            validation_collection_data.append(image_information_dict)

        return validation_collection_data
    
    def load(self, validation_collection_data, class_mappings, filtering_meta=None):
        """
        Load the validation collection data and class mappings to generate metrics and plots.

        :param validation_collection_data: (list) List of validation collection data.
        :param class_mappings: (dict) Mapping between class IDs and class names.
        :param filtering_meta: (dict, optional) Metadata used for filtering. Defaults to None.

        :return: (dict) Dictionary containing the overall computed metrics.
        """
        generate_metrics = True

        # study_list = self.validation_cruds.get_study_list_by_batch_job_id(batch_job_id)
        # self.study_list = study_list
        self.study_list = []

        plotting_data = self._load_plotting_data(
            validation_collection_data=validation_collection_data,
            generate_metrics=generate_metrics,
        )

        # Create per image variables
        ground_truth_dict = plotting_data["per_image"]["ground_truth"]
        prediction_dict = plotting_data["per_image"]["prediction"]

        meta_data_dict = None

        meta_data_dict = plotting_data["per_image"]["meta_data"]

        loss_dict = None

        try:
            loss_dict = plotting_data["per_image"]["loss"]
        except:
            print("Loss not found.")

        self.plot_driver = SemanticSegmentationPlotDriver(
            class_mappings=class_mappings,
            ground_truth_dict=ground_truth_dict,
            prediction_dict=prediction_dict,
            meta_data_dict=meta_data_dict,
            loss_dict=loss_dict,
            batch_job_id=self.batch_job_id,
            filtering_meta=filtering_meta,
        )

        overall_metrics = self.plot_driver._calling_all_plots()

        return overall_metrics

    def plot_metrics(self, metrics, json_output_dir,plot_outputs_dir, plot_configs):
        """
        Generate and save various types of plots based on the provided metrics and configurations.

        This function creates different types of plots (e.g., violin graphs, plots by metadata, overall metrics, etc.)
        and saves them to the specified output directories.

        :param metrics: (dict) Dictionary containing the computed metrics.
        :param json_output_dir: (str) Directory path where JSON files for the plots will be saved.
        :param plot_outputs_dir: (str) Directory path where the generated plot images will be saved.
        :param plot_configs: (dict) Configuration dictionary specifying the types of plots to generate and their parameters.

        :return: None
        """
        
        file_name_patterns = {
            "violin_graph": ("violin_path", "plot_{}.json"),
            "plot_by_meta_data": (
                "plot_by_meta_data",
                "plot_metrics_by_meta_data.json",
            ),
            "overall_metrics": ("overall_data", "plot_highlighted_{}.json"),
            "classbased_table": ("classed_table", "plot_statistics_{}.json"),
            "blind_spot": ("blind_spot", "plot_{}_metrics.json"),
        }

        draw_params = {
            "violin_graph": lambda c: {
                "metrics": c.get("metrics"),
                "threshold": c.get("threshold"),
            },
            "plot_by_meta_data": lambda c: {"meta_data_args": c.get("meta_data_args")},
            "overall_metrics": lambda c: {"overall_args": c.get("overall_args")},
            "classbased_table": lambda c: {
                "classbased_table_args": c.get("classbased_table_args")
            },
            "blind_spot": lambda c: {"blind_spot_args": c.get("blind_spot_args")},
        }

        for draw_type, config in plot_configs.items():
            arg_name, file_pattern = file_name_patterns.get(
                draw_type, (None, "plot_{}.json")
            )
            if arg_name is None:
                print(f"Warning: Unknown draw type '{draw_type}'. Skipping.")
                continue

            file_name = file_pattern.format(draw_type)
            file_path = os.path.join(json_output_dir, file_name)
            plot = Semantic_Segmentation_Plot(**{arg_name: file_path})
            save_path = os.path.join(plot_outputs_dir, f"{draw_type}.png")

            params = draw_params.get(draw_type, lambda _: {})(config)
            plot.draw(draw_type, save_path=save_path, **params)

    def _load_plotting_data(
        self, validation_collection_data=None, generate_metrics=True, study_list=None,
    ):
        """
        Load plotting data for per-image and per-dataset plots.

        :param validation_collection_data: (list) List of validation collection data.
        :param generate_metrics: (bool) Flag indicating whether to generate metrics. Defaults to True.
        :param study_list: (list, optional) List of studies. Defaults to None.

        :return: (dict) Dictionary containing plotting data for per-image and per-dataset plots.
        """
        plotting_data = dict()
        plotting_data["per_image"] = self._craft_per_image_plotting_data(
            validation_collection_data,
            generate_metrics=generate_metrics,
            study_list=study_list,
        )

        return plotting_data

    def _craft_per_image_plotting_data(
        self, 
        validation_collection_data: Optional[List[Dict[str, Any]]], 
        generate_metrics: bool, 
        study_list: Optional[List[Any]] = None
        ) -> Dict[str, Any]:
        """
        Create data for per-image plots.

        :param validation_collection_data: (list) List of validation collection data.
        :param generate_metrics: (bool) Flag indicating whether to generate metrics.
        :param study_list: (list, optional) List of studies. Defaults to None.

        :return: (dict) Dictionary containing ground truth, predictions, and metadata for each image.
        """
        data = dict()
        validation_df = pd.DataFrame(validation_collection_data)

        # Ground truth dict

        gt_dict = validation_df[["image_id", "ground_truth", "shape"]].values
        ground_truth_dict = {}
        for image_id, ground_truth_list, shape_list in gt_dict:
            shape = shape_list
            rle_dict = {"rles": []}
            if len(ground_truth_list) > 1:
                for item in ground_truth_list:
                    label = item["label"]
                    rle = {"rle": item["mask"]["mask"], "shape": shape, "class": label}
                    rle_dict["rles"].append(rle)
                ground_truth_dict[image_id] = rle_dict

            else:
                ground_truth = ground_truth_list[0]
                label = ground_truth["label"]
                rles_str = ground_truth["mask"]["mask"]
                rles = {"rles": [{"rle": rles_str, "shape": shape, "class": label}]}
                ground_truth_dict[image_id] = rles

        # Prediction dict
        pred_dict = validation_df[["image_id", "objects", "shape"]].values
        prediction_dict = {}
        for image_id, objects, shape in pred_dict:
            rles = objects["rles"]
            for rle_dict in rles:
                rle_dict["shape"] = shape
            prediction_dict[image_id] = objects

        # Loss dict
        if generate_metrics:
            try:
                loss_dict = (
                    validation_df[["image_id", "loss"]]
                    .set_index("image_id")
                    .to_dict()["loss"]
                )
            except:
                pass

        # Meta Data dict
        meta_data_dict = validation_df[["image_id", "meta_data"]].values
        meta_data_dict = dict(zip(meta_data_dict[:, 0], meta_data_dict[:, 1]))

        data["ground_truth"] = ground_truth_dict
        data["prediction"] = prediction_dict

        data["meta_data"] = meta_data_dict

        if generate_metrics:
            try:
                data["loss"] = loss_dict
            except:
                pass
        return data
