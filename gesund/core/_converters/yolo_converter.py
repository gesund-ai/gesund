from typing import Union, Dict, List
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
    def __init__(self, annotations: list, predictions: list, image_width: int=512, image_height: int = 512):
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
                        "loss": loss
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
                    custom_json[image_id]["annotation"].append(
                        {"label": class_id}
                    )
                else:    
                    custom_json[image_id] = {
                        "image_id": image_id,
                        "annotation": [
                            {
                                "label": class_id,
                            }
                        ]
                    }
        return custom_json



class ObjectDetectionConverter:
    def __init__(self, annotations: list, predictions: list, image_width: int = 512, image_height: int = 512):
        """
        The initialization function of the class

        
        :param annotations: list: a list of annotation in the yolo format to convert to gesund format
        :param predictions: list: a list of predictions in the yolo format to convert into the gesund format
        :parma image_width: int: The width of the image
        :param image_height: int: The height of the image
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
                    "box": {
                        "x1": x1, 
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "confidence": annotation["confidence"],
                    "prediction_class": class_id
                }

                if image_id in custom_json:
                    custom_json[image_id]["objects"].append(custom_prediction)
                else:
                    custom_json[image_id] = {
                        "objects": [],
                        "shape": [self.image_width, self.image_height], "status": 200,
                        "image_id": image_id
                    }
                    custom_json[image_id]["objects"].append(custom_prediction)
        return custom_json 

    def _convert_annotations(self) -> dict:
        """
        A function to convert the yolo predictions to gesund predictions format

        :return: a dictionary of predictions in the gesund predictions format
        """
        custom_json = {}

        for item in self.annotation:
            image_id = item["image_id"]
            annotations = item["annotations"]

            custom_annotations = []
            for annotation in annotations:
                class_id = annotation['class']
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
                    "points": [
                        {"x": x1, "y": y1},
                        {"x": x2, "y": y2}
                    ],
                    "type": "rect"
                }
                custom_annotations.append(custom_annotation)
            
            custom_json[image_id] = {
                "image_id": image_id,
                "annotation": custom_annotations
            }
        return custom_json

class SemanticSegmentationConverter:
    def __init__(self, annotations: list, predictions: list, image_width: int = 512, image_height: int = 512):
        """
        the initialization function of the class

        :param annotations: list: a list of annotation in the yolo format to convert to gesund format
        :param predictions: list: a list of predictions in the yolo format to convert into the gesund format
        :param image_width: int: The width of the image
        :param image_height: int: The height of the image
        
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
                    {"x": int(val["x"] * self.image_width), "y": int(val["y"] * self.image_height)}
                    for val in segmentation
                ]

                # convert the pixel values to binary masks
                binary_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
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
                        "masks":{
                            "rles": [rle]
                        },
                        "shape": [self.image_width, self.image_height],
                        "status": 200
                    }
            return custom_json

    def _convert_annotations(self) -> dict:
        """
        A function to convert the yolo annotations to gesund predictions format
        
        :return: a list of objects in the gesund predictions format
        """
        custom_json = {}

        for item in self.annotation:
            image_id = item["image_id"]
            annotations = item["annotations"]

            custom_annotations  = []
            for annotation in annotations:
                class_id = annotation["class"]
                segmentation = annotation["segmentation"]

                # convert the normalized values to pixel values
                pixel_values = [
                    {"x": int(val["x"] * self.image_width), 'y': int(val["y"] * self.image_height)}
                    for val in segmentation
                ]

                # convert the pixel values to binary masks
                binary_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

                for point in pixel_values: 
                    x, y = point["x"], point["y"]
                    if 0<= x < self.image_width and 0 <= y < self.image_height:
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
                        "measurement": "Segmentation"
                    },
                    "mask": {
                        "mask": rle
                    },
                    "shape": [self.image_width, self.image_height],
                    "window_level": None
                }
                custom_annotations.append(custom_annotation)
            custom_json[image_id] = {
                "image_id": image_id,
                "annotation": custom_annotations
            }
        
        return custom_json

class InstanceSegmentationConverter:
    def __init__(self, annotations: list, predictions: list, image_width: int = 512, image_height: int = 512):
        """
        The initialization function of the class

        
        :param annotations: list: a list of annotation in the yolo format to convert to gesund format
        :param predictions: list: a list of predictions in the yolo format to convert into the gesund format
        :param image_width: int: The width of the image
        :param image_height: int: The height of the image
        
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
        """
        pass

    def _convert_annotations(self) -> dict:
        """
        A function to convert the yolo predictions to gesund predictions format

        :return: a list of objects in the gesund predictions format
        """
        pass



class YoloToGesund:
    def __init__(self, annotations: list, predictions: list, image_width: int = 512, image_height: int = 512):
        """
        The initialization function of the class

        
        :param annotations: list: a list of annotation in the yolo format to convert to gesund format
        :param predictions: list: a list of predictions in the yolo format to convert into the gesund format
        :param image_width: int: The width of the image
        :param image_height: int: The height of the image
        
        :return: None
        """
        self.annotation = annotations
        self.predictions = predictions
        self.image_width = image_width
        self.image_height = image_height
        self.classification_converter = ClassificationConverter(
            annotations=annotations,
            predictions=predictions,
            image_width=image_width,
            image_height=image_height
        )
        self.obj_converter = ObjectDetectionConverter(
            annotations=annotations,
            predictions=predictions,
            image_width=image_width,
            image_height=image_height
        )
        self.semantic_segmentation_converter = SemanticSegmentationConverter(
            annotations=annotations,
            predictions=predictions,
            image_width=image_width,
            image_height=image_height
        )
        self.instance_segmentation_converter = InstanceSegmentationConverter(
            annotations=annotations,
            predictions=predictions,
            image_width=image_width,
            image_height=image_height
        )

    def run(self, problem_type: str= "object_detection", input_type: str="prediction") -> Dict:
        """
        A run method to execute the pipeline and convert the jsons

        
        :param problem_type: str: object detection 
        :param input_type: str: to indicate if the input is prediction or annotation
        
        :return: dictionary of converted format
        """
        _class_problems = {
            "classification": self.classification_converter,
            "object_detection": self.obj_converter,
            "semantic_segmentation": self.semantic_segmentation_converter,
            "instance_segmentation": self.instance_segmentation_converter
        }
        _func = {
            "prediction": _class_problems[problem_type]._convert_predictions, 
            "annotation": _class_problems[problem_type]._convert_annotations
        }
        return _func[input_type]()
    
