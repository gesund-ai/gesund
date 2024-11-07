import json
import uuid
from collections import defaultdict
import pycocotools.mask as mask_utils

from gesund.core._utils import ValidationUtils

class COCOToGesund:
    """
    A class to convert annotations and predictions between COCO format and custom format
    for various problem types such as classification, object detection, and segmentation.
    
    :param problem_type: The type of problem being addressed, e.g., 'classification', 
                         'semantic_segmentation', 'object_detection', 'instance_segmentation'.
    :type problem_type: str
    :param annotations: Annotations data in COCO format (optional).
    :type annotations: dict, optional
    :param successful_batch_data: Batch data containing successful predictions in COCO format (optional).
    :type successful_batch_data: list, optional
    """
    def __init__(self, problem_type, annotations=None, successful_batch_data=None):
        self.problem_type = problem_type
        if annotations:
            self.annotations = annotations
        if successful_batch_data:
            self.successful_batch_data = successful_batch_data

    def is_annot_coco_format(self):
        """
        Check if the annotations are in COCO format. COCO format includes keys like
        'images', 'annotations', and 'categories'.

        :return: True if the annotations follow the COCO format, False otherwise.
        :rtype: bool
        """
        return all(key in self.annotations for key in ['images', 'annotations', 'categories'])
    
    def is_pred_coco_format(self):
        """
        Check if the predictions are in COCO format. COCO predictions typically contain 
        'image_id', 'category_id', and 'score'. The 'loss' key is optional.

        :return: True if the predictions follow the COCO format, False otherwise.
        :rtype: bool
        """
        if isinstance(self.successful_batch_data, list) and len(self.successful_batch_data) > 0:
            # Mandatory keys
            required_keys = {'image_id', 'category_id', 'score'}
            # Check if the first item contains all the required keys
            return all(key in self.successful_batch_data[0] for key in required_keys)
        return False

    def convert_annotations(self):
        """
        Convert annotations to the custom format based on the problem type.

        :return: Annotations in the custom format.
        :rtype: dict
        :raises ValueError: If the problem type is unsupported.
        """
        if self.problem_type == 'classification':
            return self.convert_classification_annotations()
        elif self.problem_type == 'semantic_segmentation':
            return self.convert_semantic_segmentation_annotations()
        elif self.problem_type == 'object_detection':
            return self.convert_object_detection_annotations()
        elif self.problem_type == 'instance_segmentation':
            return self.convert_instance_segmentation_annotations()
        else:
            raise ValueError("Unsupported problem type.")

    def convert_predictions(self):
        """
        Convert predictions to the custom format based on the problem type.

        :return: Predictions in the custom format.
        :rtype: dict
        :raises ValueError: If the problem type is unsupported.
        """
        if self.problem_type == 'classification':
            return self.convert_classification_predictions()
        elif self.problem_type == 'semantic_segmentation':
            return self.convert_semantic_segmentation_predictions()
        elif self.problem_type == 'object_detection':
            return self.convert_object_detection_predictions()
        elif self.problem_type == 'instance_segmentation':
            return self.convert_instance_segmentation_predictions()
        else:
            raise ValueError("Unsupported problem type.")

    def convert_classification_annotations(self):
        """
        Convert classification annotations from COCO format to custom format.

        :return: Annotations in custom classification format.
        :rtype: dict
        """
        custom_annotations = {}
        for image in self.annotations['images']:
            image_id = image['id']
            custom_annotations[image_id] = {"annotation": []}

        for annotation in self.annotations['annotations']:
            image_id = annotation['image_id']
            custom_annotations[image_id]['annotation'].append({
                'id': annotation['id'],
                'label': annotation['category_id']
            })

        return custom_annotations

    def convert_semantic_segmentation_annotations(self):
        """
        Convert semantic segmentation annotations from COCO format to custom format.

        :return: Annotations in custom semantic segmentation format.
        :rtype: dict
        """
        # Initialize a dictionary to hold final annotations
        custom_annotations = {}

        # Group annotations by image_id
        grouped_annotations = defaultdict(lambda: {
            "image_id": None,
            "annotation": []
        })

        for ann in self.annotations["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            rle = ann["segmentation"]
            size = rle["size"]

            # Convert RLE to mask string
            rle_mask = mask_utils.decode(rle)
            rle_, shape = ValidationUtils.mask_to_rle(rle_mask)

            # Populate the grouped_annotations structure
            if grouped_annotations[image_id]["image_id"] is None:
                grouped_annotations[image_id]["image_id"] = image_id

            # Create an annotation entry
            annotation_entry = {
                "image_id": image_id,
                "label": category_id,
                "type": "mask",
                "measurement_info": {
                    "objectName": "mask",
                    "measurement": "Segmentation"
                },
                "mask": {
                    "mask": rle_  # Use the RLE string generated
                },
                "shape": size,
                "window_level": None
            }

            # Append the annotation entry to the list for this image_id
            grouped_annotations[image_id]["annotation"].append(annotation_entry)

        # Transform grouped annotations into the desired format
        for image_id, data in grouped_annotations.items():
            custom_annotations[image_id] = {
                "image_id": data["image_id"],
                "annotation": data["annotation"]
            }

        return custom_annotations

            
    def convert_object_detection_annotations(self):
        """
        Convert object detection annotations from COCO format to custom format.

        :return: Annotations in custom object detection format.
        :rtype: dict
        """
        custom_format = {}
        
        # Create a mapping of image IDs to file names and shapes
        images = {img['id']: {'file_name': img['file_name'], 'width': img['width'], 'height': img['height']} 
                for img in self.annotations['images']}
        annotations = self.annotations['annotations']
        
        for image_id, image_info in images.items():
            file_name = image_info['file_name']
            width = image_info['width']
            height = image_info['height']
            custom_annotations = []
            
            # Filter annotations for the current image
            for ann in annotations:
                if ann['image_id'] == image_id:
                    # Convert COCO bbox format (x, y, width, height) to custom points format
                    x1, y1, width_bbox, height_bbox = ann['bbox']
                    x2, y2 = x1 + width_bbox, y1 + height_bbox
                    points = [{"x": x1, "y": y1}, {"x": x2, "y": y2}]

                    # Append annotation to the list
                    custom_annotations.append({
                        "config_id": "uJ4FNIiXW4JyRCRRX_qzz",
                        "name": "default",
                        "id": str(uuid.uuid4()),
                        "label": ann['category_id'],  # You can adjust this mapping as needed
                        "points": points,
                        "type": "rect"
                    })
            
            # Add image and its annotations to the custom format
            custom_format[image_id] = {
                "image_id": image_id,
                "annotation": custom_annotations,
                "image_name": f"{file_name}",
                "shape": [height, width], 
                "last_updated_timestamp": 1727595047,
                "config_id": "uJ4FNIiXW4JyRCRRX_qzz"
            }
        
        return custom_format


    def convert_instance_segmentation_annotations(self):
        """
        Convert instance segmentation annotations from COCO format to custom format.

        :return: Annotations in custom instance segmentation format.
        :rtype: dict
        """
        pass

    def convert_classification_predictions(self):
        """
        Convert classification predictions from COCO format to custom format.

        :return: Predictions in custom classification format.
        :rtype: dict
        """
        custom_predictions = {}
        for pred in self.successful_batch_data:
            image_id = pred['image_id']
            category_id = pred['category_id']
            confidence = pred['score']
            loss = pred.get('loss', None)

            logits = [0.0, 0.0]
            logits[category_id] = confidence
            logits[1 - category_id] = 1 - confidence

            custom_predictions[image_id] = {
                'image_id': image_id,
                'prediction_class': category_id,
                'confidence': confidence,
                'logits': logits,
                'loss': loss
            }

        return custom_predictions
    
    def convert_semantic_segmentation_predictions(self):
        """
        Convert semantic segmentation predictions from COCO format to custom format.

        :return: Predictions in custom semantic segmentation format.
        :rtype: dict
        """
        custom_predictions = {}

        # Group by image_id
        grouped_predictions = defaultdict(lambda: {
            "image_id": None,
            "masks": {
                "rles": []
            },
            "shape": None,
            "status": 200
        })

        for pred in self.successful_batch_data:
            image_id = pred["image_id"]
            class_id = pred["category_id"]
            rle = pred["segmentation"]
            size = rle["size"]

            # Convert RLE to mask string
            rle_mask = mask_utils.decode([rle])
            rle_, shape = ValidationUtils.mask_to_rle(rle_mask)
            
            # Populate the grouped_annotations structure
            if grouped_predictions[image_id]["image_id"] is None:
                grouped_predictions[image_id]["image_id"] = image_id
                grouped_predictions[image_id]["shape"] = size
            
            # Create an RLE entry for the current annotation
            rle_entry = {
                "rle": rle_,  # Use the RLE string generated
                "class": class_id
            }
            
            # Append the RLE entry to the masks
            grouped_predictions[image_id]["masks"]["rles"].append(rle_entry)
            
            
        # Transform grouped annotations into the desired format
        for image_id, data in grouped_predictions.items():
            custom_predictions[image_id] = {
                "image_id": data["image_id"],
                "masks": data["masks"],
                "shape": data["shape"],
                "status": data["status"]
            }

        return custom_predictions

    def convert_object_detection_predictions(self):
        """
        Convert object detection predictions from COCO format to custom format.

        :return: Predictions in custom object detection format.
        :rtype: dict
        """
        predictions_format = {}
    
        # Loop through each item in the custom format data
        for prediction in self.successful_batch_data:
            image_id = prediction['image_id']
            bbox = prediction['bbox']
            score = prediction['score']
            category_id = prediction['category_id']

            # Add to the predictions format
            if image_id not in predictions_format:
                predictions_format[image_id] = {
                    "objects": [],
                    "image_id": image_id,
                    "status": 200
                }
            
            # Append the object to the objects list
            predictions_format[image_id]['objects'].append({
                "box": {
                    "x1": round(bbox[0]),
                    "y1": round(bbox[1]),
                    "x2": round(bbox[0] + bbox[2]),  # Calculate x2 based on width
                    "y2": round(bbox[1] + bbox[3])   # Calculate y2 based on height
                },
                "confidence": round(score, 4),  # Use the provided score
                "prediction_class": category_id   # Use category_id directly
            })

        return predictions_format

    def convert_instance_segmentation_predictions(self):
        """
        Convert instance segmentation predictions from COCO format to custom format.

        :return: Predictions in custom instance segmentation format.
        :rtype: dict
        """
        pass

    def convert_annot_if_needed(self):
        """
        Convert annotations to custom format if they are in COCO format. 
        If they are already in custom format, no conversion is performed.

        :return: Annotations in custom format.
        :rtype: dict
        """
        if self.is_annot_coco_format():
            print("Annotations are in COCO format. Converting to custom format.")
            return self.convert_annotations()
        else:
            print("Annotations are already in custom format. No conversion needed.")
            return self.annotations
        
    def convert_pred_if_needed(self):
        """
        Convert predictions to custom format if they are in COCO format. 
        If they are already in custom format, no conversion is performed.

        :return: Predictions in custom format.
        :rtype: dict
        """
        if self.is_pred_coco_format():
            print("Predictions are in COCO format. Converting to custom format.")
            return self.convert_predictions()
        else:
            print("Predictions are already in custom format. No conversion needed.")
            return self.successful_batch_data
