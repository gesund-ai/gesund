import time
import datetime
import pandas as pd
import json
import numpy as np
import os
import matplotlib.pyplot as plt

from .plots.plot_driver import ObjectDetectionPlotDriver
from gesund.core._metrics.object_detection.object_detection_metric_plot import (
    Object_Detection_Plot,
)


class ValidationCreation:
    """
    Class responsible for creating validation data and generating classification metrics and plots.

    Attributes:
        batch_job_id (str): Unique identifier for the batch job.
        filter_field (str): The field to filter the data, default is "image_url".
        generate_metrics (bool): Whether to generate metrics during validation creation, default is True.
    """

    def __init__(self, batch_job_id, filter_field="image_url", generate_metrics=True):
        """
        Initialize the ValidationCreation class.

        :param batch_job_id: Identifier for the batch job (str).
        :param filter_field: Field used for filtering the data (str, optional). Defaults to "image_url".
        :param generate_metrics: Flag to control whether metrics should be generated (bool, optional). Defaults to True.
        """
        self.batch_job_id = batch_job_id
        self.filter_field = filter_field
        self.generate_metrics = generate_metrics

    def create_validation_collection_data(
        self, successful_batch_data, annotation_data, format, meta_data=None
    ):
        """
        Create a list of validation collection data from batch and annotation data.

        :param successful_batch_data: Dictionary containing data for successfully processed images (dict).
        :param annotation_data: Dictionary containing annotation data for the images (dict).
        :param format: Format of the annotations (str, e.g., 'coco').
        :param meta_data: Additional metadata for each image (dict, optional). Defaults to None.

        :return: A list of dictionaries with validation data for each image (list).
        """

        validation_collection_data = list()
        # To Do: Optimize this
        for image_id in successful_batch_data:
            batch_item = successful_batch_data[image_id]
            annotation_item = annotation_data[image_id]
            image_information_dict = {}
            image_information_dict["batch_job_id"] = self.batch_job_id
            image_information_dict["image_id"] = batch_item["image_id"]
            if format == "coco":
                image_information_dict["shape"] = annotation_item["shape"]
            else:
                image_information_dict["shape"] = batch_item["shape"]
            image_information_dict["ground_truth"] = [
                {
                    "class": i["label"],
                    "box": {
                        "x1": i["points"][0]["x"],
                        "x2": i["points"][1]["x"],
                        "y1": i["points"][0]["y"],
                        "y2": i["points"][1]["y"],
                    },
                }
                for i in annotation_item["annotation"]
            ]
            image_information_dict["objects"] = batch_item["objects"]
            image_information_dict["created_timestamp"] = time.time()
            image_information_dict["meta_data"] = (
                meta_data[image_id]["metadata"] if meta_data else {}
            )
            validation_collection_data.append(image_information_dict)
        return validation_collection_data

    def _load_plotting_data(self, validation_collection_data, class_mappings):
        """
        Load plotting data for per-image and per-dataset plots.

        :param validation_collection_data: List of validation collection data (list).
        :param class_mappings: Mapping between class IDs and class names (dict).

        :return: Dictionary containing plotting data for per-image and per-dataset plots (dict).
        """
        plotting_data = dict()
        plotting_data["per_image"] = self._craft_per_image_plotting_data(
            validation_collection_data
        )
        plotting_data["per_dataset"] = self._craft_per_dataset_plotting_data(
            validation_collection_data, class_mappings
        )
        return plotting_data

    def _craft_per_dataset_plotting_data(
        self, validation_collection_data, class_mappings
    ):
        """
        Create data for per-dataset plots (e.g., COCO format).

        :param validation_collection_data: List of validation collection data (list).
        :param class_mappings: Mapping between class IDs and class names (dict).

        :return: Dictionary containing prediction and ground truth data in COCO format (dict).
        """
        per_dataset = dict()
        if validation_collection_data:
            per_dataset["prediction_coco"] = self._load_pred_coco(
                validation_collection_data
            )
            per_dataset["ground_truth_coco"] = self._load_annot_coco(
                validation_collection_data, class_mappings
            )
        return per_dataset

    def _craft_per_image_plotting_data(self, validation_collection_data):
        """
        Create data for per-image plots.

        :param validation_collection_data: List of validation collection data (list).

        :return: Dictionary containing ground truth, predictions, and metadata for each image (dict).
        """
        data = dict()
        validation_df = pd.DataFrame(validation_collection_data)
        # Ground truth dict
        ground_truth_dict = validation_df[["image_id", "ground_truth"]].values
        ground_truth_dict = dict(zip(ground_truth_dict[:, 0], ground_truth_dict[:, 1]))
        # Prediction dict
        prediction_dict = validation_df[["image_id", "objects"]].values
        prediction_dict = dict(zip(prediction_dict[:, 0], prediction_dict[:, 1]))

        try:
            loss_dict = (
                validation_df[["image_id", "loss"]]
                .set_index("image_id")
                .to_dict()["loss"]
            )
        except:
            pass

        data["ground_truth"] = ground_truth_dict
        data["objects"] = prediction_dict
        meta_data_dict = validation_df[["image_id", "meta_data"]].values
        meta_data_dict = dict(zip(meta_data_dict[:, 0], meta_data_dict[:, 1]))

        data["meta_data"] = meta_data_dict

        try:
            data["loss"] = loss_dict
        except:
            pass
        return data

    def load(self, validation_collection_data, class_mappings, filtering_meta=None):
        """
        Load the validation collection data and class mappings to generate metrics and plots.

        :param validation_collection_data: List of validation collection data (list).
        :param class_mappings: Mapping between class IDs and class names (dict).
        :param filtering_meta: Metadata used for filtering (dict, optional). Defaults to None.

        :return: Dictionary containing the overall computed metrics (dict).
        """
        plotting_data = self._load_plotting_data(
            validation_collection_data=validation_collection_data,
            class_mappings=class_mappings,
        )

        # Create per image variables
        ground_truth_dict = plotting_data["per_image"]["ground_truth"]
        prediction_dict = plotting_data["per_image"]["objects"]

        pred_coco = plotting_data["per_dataset"]["prediction_coco"]
        gt_coco = plotting_data["per_dataset"]["ground_truth_coco"]
        coco_ = [pred_coco, gt_coco]
        meta_data_dict = None

        meta_data_dict = plotting_data["per_image"]["meta_data"]

        loss_dict = None

        try:
            loss_dict = plotting_data["per_image"]["loss"]
        except:
            print("Loss not found.")

        self.plot_driver = ObjectDetectionPlotDriver(
            coco_=coco_,
            class_mappings=class_mappings,
            ground_truth_dict=ground_truth_dict,
            prediction_dict=prediction_dict,
            meta_data_dict=meta_data_dict,
            loss_dict=loss_dict,
            batch_job_id=self.batch_job_id,
            filtering_meta=filtering_meta,
        )

        overall_metrics = self.plot_driver._calling_all_plots()

        if "mAP11@10" in overall_metrics["main_metric"]:
            del overall_metrics["main_metric"]["mAP11@10"]

        for key in overall_metrics["plot_blind_spot_metrics"]:
            if "AP11@10" in overall_metrics["plot_blind_spot_metrics"][key]:
                del overall_metrics["plot_blind_spot_metrics"][key]["AP11@10"]

        if "mAP11@10" in overall_metrics["plot_highlighted_overall_metrics"]["data"]:
            del overall_metrics["plot_highlighted_overall_metrics"]["data"]["mAP11@10"]

        for key in overall_metrics["plot_statistics_classbased_table"]["data"][
            "Validation"
        ]:
            if (
                "AP11@10"
                in overall_metrics["plot_statistics_classbased_table"]["data"][
                    "Validation"
                ][key]
            ):
                del overall_metrics["plot_statistics_classbased_table"]["data"][
                    "Validation"
                ][key]["AP11@10"]

        return overall_metrics

    def plot_metrics(self, metrics, json_output_dir, plot_outputs_dir, plot_configs):
        """
        Generate and save various types of plots based on the provided metrics and configurations.

        This function creates different types of plots (e.g., mixed plots, top misses, confidence histograms, etc.)
        and saves them to the specified output directories.

        :param metrics: Dictionary containing the computed metrics (dict).
        :param json_output_dir: Directory path where JSON files for the plots will be saved (str).
        :param plot_outputs_dir: Directory path where the generated plot images will be saved (str).
        :param plot_configs: Configuration dictionary specifying the types of plots to generate and their parameters (dict).

        :return: None
        """
        file_name_patterns = {
            "mixed_plot": ("mixed_json_path", "plot_performance_by_iou_threshold.json"),
            "top_misses": ("top_misses_path", "plot_{}.json"),
            "confidence_histogram": (
                "confidence_histogram_path",
                "plot_{}_scatter_distribution.json",
            ),
            "classbased_table": ("table_json_path", "plot_statistics_{}.json"),
            "overall_metrics": ("overall_json_path", "plot_highlighted_{}.json"),
            "blind_spot": ("blind_spot_path", "plot_{}_metrics.json"),
        }

        draw_params = {
            "mixed_plot": lambda c: {"mixed_args": c},
            "top_misses": lambda c: {"top_misses_args": c},
            "confidence_histogram": lambda c: {"confidence_histogram_args": c},
            "classbased_table": lambda c: {"classbased_table_args": c},
            "overall_metrics": lambda c: {"overall_args": c},
            "blind_spot": lambda c: {"blind_spot_args": c},
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
            plot = Object_Detection_Plot(**{arg_name: file_path})
            save_path = os.path.join(plot_outputs_dir, f"{draw_type}.png")

            try:
                params = draw_params.get(draw_type, lambda _: {})(config)
                plot.draw(draw_type, save_path=save_path, **params)
            except Exception as e:
                print(f"Error creating {draw_type}: {e}")

    def _load_pred_coco(self, validation_collection_data=None):
        """
        Load prediction data in COCO format.

        This function processes a list of validation collection data and transforms it into
        a list of prediction dictionaries formatted according to the COCO specification.

        :param validation_collection_data: (list, optional) List of validation collection data.

        :return: A list of prediction dictionaries in COCO format, each containing:
            - 'image_id' (int): Identifier of the image.
            - 'category_id' (int): Identifier of the predicted category.
            - 'bbox' (list): Bounding box coordinates in the format [x, y, width, height].
            - 'score' (float): Confidence score of the prediction.
        """

        pred_dict_list = []
        predictions_list = validation_collection_data
        for pred in predictions_list:
            objects = pred["objects"]
            image_id = pred["image_id"]

            for object_ in objects:
                bbox = list(object_["box"].values())
                class_label = object_["prediction_class"]
                confidence = object_["confidence"]

                add_dict = {
                    "image_id": image_id,
                    "category_id": class_label,
                    "bbox": [
                        bbox[0],
                        bbox[1],
                        (bbox[2] - bbox[0]),
                        (bbox[3] - bbox[1]),
                    ],
                    "score": confidence,
                }
                pred_dict_list.append(add_dict)

        return pred_dict_list

    def _load_annot_coco(self, validation_collection_data=None, class_mappings=None):
        """
        Load annotation data in COCO format.

        This function processes a list of validation collection data and class mappings to generate
        a COCO-formatted dictionary containing image metadata, annotations, and category information.

        :param validation_collection_data: (list, optional) List of validation collection data.
        :param class_mappings: (dict, optional) Mapping between class IDs and class names.

        :return: A dictionary in COCO format containing:
            - 'info' (dict): Metadata about the dataset.
            - 'licenses' (list): Licensing information for the dataset.
            - 'categories' (list): List of categories used in the dataset.
            - 'images' (list): List of image metadata, each containing:
                - 'id' (int): Identifier of the image.
                - 'license' (int): Identifier of the license.
                - 'file_name' (str): Name of the image file.
                - 'height' (int): Height of the image.
                - 'width' (int): Width of the image.
                - 'date_captured' (str): Date the image was captured.
            - 'annotations' (list): List of annotations, each containing:
                - 'id' (int): Identifier of the annotation.
                - 'image_id' (int): Identifier of the associated image.
                - 'category_id' (int): Identifier of the category for the annotation.
                - 'bbox' (list): Bounding box for the annotation.
                - 'segmentation' (None): Segmentation data (if applicable).
                - 'area' (float): Area of the bounding box.
                - 'iscrowd' (int): Indicator if the annotation is for a crowd.
        """

        coco_dict = {
            "info": {
                "year": "2022",
                "version": "1.0",
                "description": "Exported from Gesund AI",
                "contributor": "Gesund AI",
                "url": "https://gesund.ai",
                "date_created": datetime.datetime.strftime(
                    datetime.datetime.now(), format="%Y-%m-%dT%H:%M:%S%z"
                ),
            },
            "licenses": [
                {
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                    "id": 1,
                    "name": "NonLicensed Annotation",
                },
            ],
            "categories": [
                {"id": 0, "name": "cat", "supercategory": "animal"},
            ],
            "images": [
                {
                    "id": 0,
                    "license": 1,
                    "file_name": None,
                    "height": 480,
                    "width": 640,
                    "date_captured": None,
                },
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 2,
                    "bbox": [260, 177, 231, 199],
                    "segmentation": None,
                    "area": 45969,
                    "iscrowd": 0,
                },
            ],
        }

        categories = []

        for id_, name in class_mappings.items():
            categories.append({"id": int(id_), "name": name, "supercategory": name})

        coco_images_list = []
        images_list = validation_collection_data
        for img in images_list:
            image_id = img["image_id"]
            file_name = f"{image_id}.jpg"
            image_dict = {
                "id": image_id,
                "license": 1,
                "file_name": file_name,
                "height": None,
                "width": None,
                "date_captured": None,
            }
            coco_images_list.append(image_dict)

        coco_annotations_list = []
        annotations_list = validation_collection_data
        obj_id = 0
        for ant in annotations_list:
            image_id = ant["image_id"]

            for p in ant["ground_truth"]:
                category = p["class"]

                ant_dict = {
                    "id": obj_id,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": [
                        p["box"]["x1"],
                        p["box"]["y1"],
                        (p["box"]["x2"] - p["box"]["x1"]),
                        (p["box"]["y2"] - p["box"]["y1"]),
                    ],
                    "segmentation": None,
                    "area": (p["box"]["x2"] - p["box"]["x1"])
                    * (p["box"]["y2"] - p["box"]["y1"]),
                    "iscrowd": 0,
                }
                obj_id += 1
                coco_annotations_list.append(ant_dict)

        coco_dict["categories"] = categories
        coco_dict["images"] = coco_images_list
        coco_dict["annotations"] = coco_annotations_list
        annot_dict = coco_dict

        return annot_dict
