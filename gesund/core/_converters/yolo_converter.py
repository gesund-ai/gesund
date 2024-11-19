from typing import Union, Dict, List, Optional
import numpy as np


def mask_to_rle(mask: np.ndarray) -> List[int]:
    """Convert a 2D binary mask to RLE format."""
    try:
        pixels = mask.flatten()
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs = np.concatenate([[0], runs, [len(pixels)]])
        rle = []
        for start, stop in zip(runs[:-1], runs[1:]):
            if pixels[start] == 1:
                rle.extend([start, stop - start])
        return rle
    except Exception as e:
        print(e)


class ClassificationConverter:
    def __init__(
        self,
        annotations: list,
        predictions: list,
        image_width: int = 512,
        image_height: int = 512,
    ):
        """
        The initialization function of the class

        :param annotations: list of annotations
        :param predictions: list of predictions
        :param image_width: integer value representing the image width
        :param image_height: integer value representing the image height
        :return: None
        """
        self.annotations = annotations
        self.predictions = predictions
        self.image_width = image_width
        self.image_height = image_height

    def _convert_predictions(self) -> dict:
        """
        A function to convert the yolo predictions to gesund predictions format

        :return: a dictionary of predictions in the gesund predictions format
        """
        custom_json = {}

        for item in self.predictions:
            image_id = item["image_id"]

            for prediction in item["predictions"]:
                class_id = prediction["class"]
                confidence = prediction["confidence"]
                loss = prediction["loss"]
                logits = [0.0, 0.0]
                logits[class_id] = confidence
                logits[1 - class_id] = 1 - confidence

                pred = {
                    "image_id": image_id,
                    "prediction_class": class_id,
                    "confidence": confidence,
                    "logits": logits,
                    "loss": loss,
                }

                # if image_id in custom_json:
                #     custom_json[image_id].append(pred)
                # else:
                custom_json[image_id] = pred
        return custom_json

    def _convert_annotations(self) -> dict:
        """
        A function to convert the yolo annotations to gesund annotations format

        :return: a dictionary of annotations in the gesund annotations format
        """
        custom_json = {}
        for item in self.annotations:
            image_id = item["image_id"]

            for annotation in item["annotations"]:
                class_id = annotation["class"]

                if image_id in custom_json:
                    custom_json[image_id]["annotation"].append({"label": class_id})
                else:
                    custom_json[image_id] = {
                        "image_id": image_id,
                        "annotation": [
                            {
                                "label": class_id,
                            }
                        ],
                    }
        return custom_json


class ObjectDetectionConverter:
    def __init__(
        self,
        annotations: Union[list, dict],
        predictions: Union[list, dict],
        image_width: int,
        image_height: int,
    ):
        """
        The initialization function of the class


        :param annotations: a list of annotation in the yolo format to convert to gesund format
        :type annotations: list
        :param predictions: a list of predictions in the yolo format to convert into the gesund format
        :type predictions: list
        :param image_width: The width of the image
        :type image_width: int
        :param image_height: The height of the image
        :type image_height: int

        :return: None
        """
        self.annotation = annotations
        self.predictions = predictions
        self.image_width = image_width
        self.image_height = image_height

    def _convert_predictions(self) -> dict:
        """
        A function to convert the yolo predictions to gesund predictions format

        :return: a dictionary of predictions in the gesund predictions format
        :rtype: dict
        """
        custom_json = {}

        for item in self.predictions:
            image_id = item["image_id"]
            annotations = item["annotations"]

            for annotation in annotations:
                class_id = annotation["class"]
                x_center = annotation["x_center"]
                y_center = annotation["y_center"]
                width = annotation["width"]
                height = annotation["height"]

                # convert normalized to pixel values
                x1 = int((x_center - width / 2) * self.image_width)
                y1 = int((y_center - height / 2) * self.image_height)
                x2 = int((x_center + width / 2) * self.image_width)
                y2 = int((y_center + height / 2) * self.image_height)

                custom_prediction = {
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "confidence": annotation["confidence"],
                    "prediction_class": class_id,
                }

                if image_id in custom_json:
                    custom_json[image_id]["objects"].append(custom_prediction)
                else:
                    custom_json[image_id] = {
                        "objects": [],
                        "shape": [self.image_width, self.image_height],
                        "status": 200,
                        "image_id": image_id,
                    }
                    custom_json[image_id]["objects"].append(custom_prediction)
        return custom_json

    def _convert_annotations(self) -> dict:
        """
        A function to convert the yolo predictions to gesund predictions format

        :return: a dictionary of predictions in the gesund predictions format
        :rtype: dict
        """
        custom_json = {}

        for item in self.annotation:
            image_id = item["image_id"]
            annotations = item["annotations"]

            custom_annotations = []
            for annotation in annotations:
                class_id = annotation["class"]
                x_center = annotation["x_center"]
                y_center = annotation["y_center"]
                width = annotation["width"]
                height = annotation["height"]

                # convert normalized values to pixel values
                x1 = int((x_center - width / 2) * self.image_width)
                y1 = int((y_center - height / 2) * self.image_height)
                x2 = int((x_center + width / 2) * self.image_width)
                y2 = int((y_center + height / 2) * self.image_height)

                custom_annotation = {
                    "label": f"class_{class_id}",
                    "points": [{"x": x1, "y": y1}, {"x": x2, "y": y2}],
                    "type": "rect",
                }
                custom_annotations.append(custom_annotation)

            custom_json[image_id] = {
                "image_id": image_id,
                "annotation": custom_annotations,
            }
        return custom_json


class SemanticSegmentationConverter:
    def __init__(
        self,
        annotations: Union[list, dict],
        predictions: Union[list, dict],
        image_width: int,
        image_height: int,
    ):
        """
        the initialization function of the class

        :param annotations: a list of annotation in the yolo format to convert to gesund format
        :type annotations: Union[list, dict]
        :param predictions: a list of predictions in the yolo format to convert into the gesund format
        :type predictions: Union[list, dict]
        :param image_width: The width of the image
        :type image_width: int
        :param image_height: The height of the image
        :type image_height: int

        :return: None
        """
        self.annotation = annotations
        self.predictions = predictions
        self.image_width = image_width
        self.image_height = image_height

    def _convert_predictions(self) -> dict:
        """
        A function to convert the yolo predictions to gesund predictions format

        :return: a list of objects in the gesund predictions format
        :rtype: dict
        """
        custom_json = {}

        for item in self.predictions:
            image_id = item["image_id"]
            predictions = item["predictions"]
            for pred in predictions:
                class_id = pred["class"]
                segmentation = pred["segmentation"]

                # convert to pixel values
                pixel_values = [
                    {
                        "x": int(val["x"] * self.image_width),
                        "y": int(val["y"] * self.image_height),
                    }
                    for val in segmentation
                ]

                # convert the pixel values to binary masks
                binary_mask = np.zeros(
                    (self.image_height, self.image_width), dtype=np.uint8
                )
                for point in pixel_values:
                    x, y = point["x"], point["y"]
                    if 0 <= x < self.image_width and 0 <= y < self.image_height:
                        binary_mask[y, x] = 1

                # convert the binary masks to RLE masks
                rle = mask_to_rle(binary_mask)
                rle = " ".join([str(i) for i in rle])
                rle = {"rle": rle, "class": class_id}

                # save the created items in the json format
                if image_id in custom_json:
                    custom_json[image_id]["masks"]["rles"].append(rle)
                else:
                    custom_json[image_id] = {
                        "image_id": image_id,
                        "masks": {"rles": [rle]},
                        "shape": [self.image_width, self.image_height],
                        "status": 200,
                    }
            return custom_json

    def _convert_annotations(self) -> dict:
        """
        A function to convert the yolo annotations to gesund predictions format

        :return: a list of objects in the gesund predictions format
        :rtype: dict
        """
        custom_json = {}

        for item in self.annotation:
            image_id = item["image_id"]
            annotations = item["annotations"]

            custom_annotations = []
            for annotation in annotations:
                class_id = annotation["class"]
                segmentation = annotation["segmentation"]

                # convert the normalized values to pixel values
                pixel_values = [
                    {
                        "x": int(val["x"] * self.image_width),
                        "y": int(val["y"] * self.image_height),
                    }
                    for val in segmentation
                ]

                # convert the pixel values to binary masks
                binary_mask = np.zeros(
                    (self.image_height, self.image_width), dtype=np.uint8
                )

                for point in pixel_values:
                    x, y = point["x"], point["y"]
                    if 0 <= x < self.image_width and 0 <= y < self.image_height:
                        binary_mask[y, x] = 1

                # convert the binary masks to RLE masks
                rle = mask_to_rle(binary_mask)
                rle = " ".join([str(i) for i in rle])
                # save the created items in the json format
                custom_annotation = {
                    "image_id": image_id,
                    "label": class_id,
                    "type": "mask",
                    "measurement_info": {
                        "objectName": "mask",
                        "measurement": "Segmentation",
                    },
                    "mask": {"mask": rle},
                    "shape": [self.image_width, self.image_height],
                    "window_level": None,
                }
                custom_annotations.append(custom_annotation)
            custom_json[image_id] = {
                "image_id": image_id,
                "annotation": custom_annotations,
            }

        return custom_json


class InstanceSegmentationConverter:
    def __init__(
        self,
        annotations: Union[list, dict],
        predictions: Union[list, dict],
        image_width: int,
        image_height: int,
    ):
        """
        The initialization function of the class


        :param annotations: a list of annotation in the yolo format to convert to gesund format
        :type annotations: Union[list, dict]
        :param predictions: a list of predictions in the yolo format to convert into the gesund format
        :type predictions: Union[list, dict]
        :param image_width: The width of the image
        :type image_width: int
        :param image_height: The height of the image
        :type image_height: int

        :return: None
        """
        self.annotation = annotations
        self.predictions = predictions
        self.image_width = image_width
        self.image_height = image_height

    def _convert_predictions(self) -> dict:
        """
        A function to convert the yolo predictions to gesund predictions format

        :return: a dict of objects in the gesund predictions format
        :rtype: dict
        """
        pass

    def _convert_annotations(self) -> dict:
        """
        A function to convert the yolo predictions to gesund predictions format

        :return: a dict of objects in the gesund predictions format
        :rtype: dict
        """
        pass


class YoloProblemTypeFactory:
    def get_yolo_converter(self, problem_type: str):
        """
        A factory method to get the yolo converter

        :param problem_type: type of the problem
        :type problem_type: str

        :return: class of the converter
        :rtype: class
        """
        if problem_type == "object_detection":
            return ObjectDetectionConverter
        elif problem_type == "semantic_segmentation":
            return SemanticSegmentationConverter
        elif problem_type == "classification":
            return ClassificationConverter
        else:
            raise NotImplementedError


class YoloToGesund:
    def convert(
        self,
        annotation: Union[List[Dict], Dict],
        prediction: Union[List[Dict], Dict],
        problem_type: str,
        image_width: Optional[int] = 512,
        image_height: Optional[int] = 512,
    ) -> Dict:
        """
        A run method to execute the pipeline and convert the given format to gesund format

        :param annotation: the annotations data
        :type annotation: Union[List[Dict], Dict]
        :param prediction: the prediction data
        :type prediction: Union[List[Dict], Dict]
        :param problem_type: problem type could be classification | object detection | segmentation
        :type problem_type: str
        :param image_width: width of the image
        :type image_width: int
        :param image_height: height of the image
        :type image_height: int

        :return: dictionary of converted format
        :rtype: dict
        """
        _converter_cls = YoloProblemTypeFactory().get_yolo_converter(
            problem_type=problem_type
        )
        _converter = _converter_cls(
            annotations=annotation,
            predictions=prediction,
            image_width=image_width,
            image_height=image_height,
        )
        return _converter._convert_annotations(), _converter._convert_predictions()
