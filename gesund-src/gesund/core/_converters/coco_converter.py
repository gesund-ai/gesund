from typing import Dict, List, Optional, Any, Union, Set, Tuple
import json
import uuid
from collections import defaultdict
import pycocotools.mask as mask_utils

from gesund.core._utils import ValidationUtils


class ClassificationConverter:
    def __init__(self, annotations: dict, predictions: list):
        """
        Initializes the ClassificationConverter with annotations and predictions.

        :param annotations: Annotations data in COCO format.
        :type annotations: dict
        :param predictions: Predictions data in COCO format.
        :type predictions: list
        """
        self.annotations = annotations
        self.predictions = predictions

    def _convert_annotations(self) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
        """
        Convert classification annotations from COCO format to custom format.

        :return: Annotations in custom classification format.
        :rtype: Dict[int, Dict[str, List[Dict[str, Any]]]]
        """
        custom_annotations = {}
        for image in self.annotations["images"]:
            image_id = image["id"]
            custom_annotations[image_id] = {"annotation": []}

        for annotation in self.annotations["annotations"]:
            image_id = annotation["image_id"]
            custom_annotations[image_id]["annotation"].append(
                {"id": annotation["id"], "label": annotation["category_id"]}
            )

        return custom_annotations

    def _convert_predictions(self) -> Dict[int, Dict[str, Any]]:
        """
        Convert classification predictions from COCO format to custom format.

        :return: Predictions in custom classification format.
        :rtype: Dict[int, Dict[str, Any]]
        """
        custom_predictions = {}
        for pred in self.predictions:
            image_id = pred["image_id"]
            category_id = pred["category_id"]
            confidence = pred["score"]
            loss = pred.get("loss", None)

            logits = [0.0, 0.0]
            logits[category_id] = confidence
            logits[1 - category_id] = 1 - confidence

            custom_predictions[image_id] = {
                "image_id": image_id,
                "prediction_class": category_id,
                "confidence": confidence,
                "logits": logits,
                "loss": loss,
            }
        return custom_predictions


class SemanticSegmentationConverter:
    def __init__(self, annotations: dict, predictions: list):
        """
        Initializes the SemanticSegmentationConverter with annotations and predictions.

        :param annotations: Annotations data in COCO format.
        :type annotations: dict
        :param predictions: Predictions data in COCO format.
        :type predictions: list
        """
        self.annotations = annotations
        self.predictions = predictions

    def _convert_annotations(self) -> Dict[int, Dict[str, Any]]:
        """
        Convert semantic segmentation annotations from COCO format to custom format.

        :return: Annotations in custom semantic segmentation format.
        :rtype: Dict[int, Dict[str, Any]]
        """
        custom_annotations = {}
        grouped_annotations = defaultdict(lambda: {"image_id": None, "annotation": []})

        for ann in self.annotations["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            rle = ann["segmentation"]
            size = rle["size"]

            rle_mask = mask_utils.decode(rle)
            rle_, shape = ValidationUtils.mask_to_rle(rle_mask)

            if grouped_annotations[image_id]["image_id"] is None:
                grouped_annotations[image_id]["image_id"] = image_id

            annotation_entry = {
                "image_id": image_id,
                "label": category_id,
                "type": "mask",
                "measurement_info": {
                    "objectName": "mask",
                    "measurement": "Segmentation",
                },
                "mask": {"mask": rle_},
                "mask": {"mask": rle_},  # Use the RLE string generated
                "shape": size,
                "window_level": None,
            }

            grouped_annotations[image_id]["annotation"].append(annotation_entry)

        for image_id, data in grouped_annotations.items():
            custom_annotations[image_id] = {
                "image_id": data["image_id"],
                "annotation": data["annotation"],
            }
        return custom_annotations

    def _convert_predictions(self) -> Dict[int, Dict[str, Any]]:
        """
        Convert semantic segmentation predictions from COCO format to custom format.

        :return: Predictions in custom semantic segmentation format.
        :rtype: Dict[int, Dict[str, Any]]
        """
        custom_predictions = {}
        grouped_predictions = defaultdict(
            lambda: {
                "image_id": None,
                "masks": {"rles": []},
                "shape": None,
                "status": 200,
            }
        )

        for pred in self.predictions:
            image_id = pred["image_id"]
            class_id = pred["category_id"]
            rle = pred["segmentation"]
            size = rle["size"]

            rle_mask = mask_utils.decode([rle])
            rle_, shape = ValidationUtils.mask_to_rle(rle_mask)

            if grouped_predictions[image_id]["image_id"] is None:
                grouped_predictions[image_id]["image_id"] = image_id
                grouped_predictions[image_id]["shape"] = size

            rle_entry = {"rle": rle_, "class": class_id}

            grouped_predictions[image_id]["masks"]["rles"].append(rle_entry)

        for image_id, data in grouped_predictions.items():
            custom_predictions[image_id] = {
                "image_id": data["image_id"],
                "masks": data["masks"],
                "shape": data["shape"],
                "status": data["status"],
            }

        return custom_predictions


class ObjectDetectionConverter:
    def __init__(self, annotations: dict, predictions: list):
        """
        Initializes the ObjectDetectionConverter with annotations and predictions.

        :param annotations: Annotations data in COCO format.
        :type annotations: dict
        :param predictions: Predictions data in COCO format.
        :type predictions: list
        """
        self.annotations = annotations
        self.predictions = predictions

    def _convert_annotations(self) -> Dict[int, Dict[str, Any]]:
        """
        Convert object detection annotations from COCO format to custom format.

        :return: Annotations in custom object detection format.
        :rtype: Dict[int, Dict[str, Any]]
        """
        custom_format = {}
        images = {
            img["id"]: {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            }
            for img in self.annotations["images"]
        }
        for image_id, image_info in images.items():
            file_name = image_info["file_name"]
            width = image_info["width"]
            height = image_info["height"]
            custom_annotations = []
            for ann in self.annotations["annotations"]:
                if ann["image_id"] == image_id:
                    x1, y1, width_bbox, height_bbox = ann["bbox"]
                    x2, y2 = x1 + width_bbox, y1 + height_bbox
                    points = [{"x": x1, "y": y1}, {"x": x2, "y": y2}]
                    custom_annotations.append(
                        {
                            "config_id": "uJ4FNIiXW4JyRCRRX_qzz",
                            "name": "default",
                            "id": str(uuid.uuid4()),
                            "label": ann["category_id"],
                            "points": points,
                            "type": "rect",
                        }
                    )
            custom_format[image_id] = {
                "image_id": image_id,
                "annotation": custom_annotations,
                "image_name": f"{file_name}",
                "shape": [height, width],
                "last_updated_timestamp": 1727595047,
                "config_id": "uJ4FNIiXW4JyRCRRX_qzz",
            }
        return custom_format

    def _convert_predictions(self) -> Dict[int, Dict[str, Any]]:
        """
        Convert object detection predictions from COCO format to custom format.

        :return: Predictions in custom object detection format.
        :rtype: Dict[int, Dict[str, Any]]
        """
        predictions_format = {}
        for prediction in self.predictions:
            image_id = prediction["image_id"]
            bbox = prediction["bbox"]
            score = prediction["score"]
            category_id = prediction["category_id"]

            if image_id not in predictions_format:
                predictions_format[image_id] = {
                    "objects": [],
                    "image_id": image_id,
                    "status": 200,
                }

            predictions_format[image_id]["objects"].append(
                {
                    "box": {
                        "x1": round(bbox[0]),
                        "y1": round(bbox[1]),
                        "x2": round(bbox[0] + bbox[2]),
                        "y2": round(bbox[1] + bbox[3]),
                    },
                    "confidence": round(score, 4),
                    "prediction_class": category_id,
                }
            )
        return predictions_format


class InstanceSegmentationConverter:
    def __init__(self, annotations: dict, predictions: list):
        """
        Initializes the InstanceSegmentationConverter with annotations and predictions.

        :param annotations: Annotations data in COCO format.
        :type annotations: dict
        :param predictions: Predictions data in COCO format.
        :type predictions: list
        """
        self.annotations = annotations
        self.predictions = predictions

    def _convert_annotations(self) -> Dict[int, Dict[str, Any]]:
        """
        Convert instance segmentation annotations from COCO format to custom format.

        :return: Annotations in custom instance segmentation format.
        :rtype: Dict[int, Dict[str, Any]]
        """
        raise NotImplementedError(
            "Instance Segmentation conversion is not implemented."
        )

    def _convert_predictions(self) -> Dict[int, Dict[str, Any]]:
        """
        Convert instance segmentation predictions from COCO format to custom format.

        :return: Predictions in custom instance segmentation format.
        :rtype: Dict[int, Dict[str, Any]]
        """
        raise NotImplementedError(
            "Instance Segmentation conversion is not implemented."
        )


class COCOProblemTypeFactory:
    def get_converter(
        self, problem_type: str
    ) -> Union[
        ClassificationConverter,
        SemanticSegmentationConverter,
        ObjectDetectionConverter,
        InstanceSegmentationConverter,
    ]:
        """
        Factory method to get the appropriate converter based on problem type.

        :param problem_type: Type of the problem (e.g., 'classification', 'semantic_segmentation', 'object_detection', 'instance_segmentation').
        :type problem_type: str

        :return: An instance of the corresponding converter.
        :rtype: Union[ClassificationConverter, SemanticSegmentationConverter, ObjectDetectionConverter, InstanceSegmentationConverter]
        :raises ValueError: If the problem type is unsupported.
        """
        if problem_type == "classification":
            return ClassificationConverter
        elif problem_type == "semantic_segmentation":
            return SemanticSegmentationConverter
        elif problem_type == "object_detection":
            return ObjectDetectionConverter
        elif problem_type == "instance_segmentation":
            return InstanceSegmentationConverter
        else:
            raise ValueError("Unsupported problem type.")


class COCOToGesund:
    def __init__(self, annotations: dict = None, successful_batch_data: list = None):
        """
        Initializes the COCOToGesund converter with annotations and predictions.

        :param annotations: Annotations data in COCO format.
        :type annotations: dict, optional
        :param successful_batch_data: Predictions data in COCO format.
        :type successful_batch_data: list, optional
        """
        self.annotations = annotations
        self.successful_batch_data = successful_batch_data

    def is_annot_coco_format(self) -> bool:
        """
        Check if the annotations are in COCO format. COCO format includes keys like 'images', 'annotations', and 'categories'.

        :return: True if the annotations follow the COCO format, False otherwise.
        :rtype: bool
        """
        required_keys = {"images", "annotations", "categories"}
        return all(key in self.annotations for key in required_keys)

    def is_pred_coco_format(self) -> bool:
        """
        Check if the predictions are in COCO format. COCO predictions typically contain 'image_id', 'category_id', and 'score'. The 'loss' key is optional.

        :return: True if the predictions follow the COCO format, False otherwise.
        :rtype: bool
        """
        if isinstance(self.successful_batch_data, list) and self.successful_batch_data:
            required_keys = {"image_id", "category_id", "score"}
            return all(key in self.successful_batch_data[0] for key in required_keys)
        return False

    def convert_annotations(self, problem_type: str) -> Dict[int, Any]:
        """
        Convert annotations based on the problem type.

        :param problem_type: Type of the problem.
        :type problem_type: str
        :return: Converted annotations.
        :rtype: Dict[int, Any]
        """
        factory = COCOProblemTypeFactory()
        converter_cls = factory.get_converter(problem_type)
        converter = converter_cls(self.annotations, self.successful_batch_data)
        return converter._convert_annotations()

    def convert_predictions(self, problem_type: str) -> Dict[int, Any]:
        """
        Convert predictions based on the problem type.

        :param problem_type: Type of the problem.
        :type problem_type: str
        :return: Converted predictions.
        :rtype: Dict[int, Any]
        """
        factory = COCOProblemTypeFactory()
        converter_cls = factory.get_converter(problem_type)
        converter = converter_cls(self.annotations, self.successful_batch_data)
        return converter._convert_predictions()

    def convert_if_needed(self, problem_type: str) -> Dict[str, Any]:
        """
        Convert annotations and predictions to custom format if they are in COCO format.

        :param problem_type: Type of the problem.
        :type problem_type: str
        :return: Dictionary containing converted annotations and predictions.
        :rtype: Dict[str, Any]
        """
        annotations = self.annotations
        predictions = self.successful_batch_data

        if self.is_annot_coco_format():
            annotations = self.convert_annotations(problem_type)
        if self.is_pred_coco_format():
            predictions = self.convert_predictions(problem_type)

        return {"annotations": annotations, "predictions": predictions}
