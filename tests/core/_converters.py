import pytest
from gesund.core._converters.coco_converter import COCOToGesund
import numpy as np
from gesund.core._converters.yolo_converter import ClassificationConverter, ObjectDetectionConverter, SemanticSegmentationConverter
from typing import List, Dict, Any
from unittest.mock import MagicMock
from pydantic import BaseModel, ValidationError


class AnnotationModel(BaseModel):
    image_id: str
    annotation: List[Dict[str, Any]]

class PredictionModel(BaseModel):
    image_id: str
    prediction_class: int
    confidence: float
    logits: List[float]
    loss: float

class SemanticAnnotationModel(BaseModel):
    image_id: str
    annotation: List[Dict[str, Any]]

class SemanticPredictionModel(BaseModel):
    image_id: str
    masks: Dict[str, Any]
    shape: List[int]
    status: int


# Sample data for ClassificationConverter
class_annotations: List[Dict[str, Any]] = [
    {
        "image_id": "image_1",
        "annotations": [{"class": 0}, {"class": 1}]
    }
]

class_predictions: List[Dict[str, Any]] = [
    {
        "image_id": "image_1",
        "predictions": [
            {"class": 0, "confidence": 0.9, "loss": 0.1},
            {"class": 1, "confidence": 0.8, "loss": 0.2}
        ]
    }
]

# Sample data for SemanticSegmentationConverter
semantic_annotations: List[Dict[str, Any]] = [
    {
        "image_id": "image_1",
        "annotations": [
            {
                "class": 0,
                "segmentation": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.2, "y": 0.2},
                    {"x": 0.3, "y": 0.3}
                ]
            }
        ]
    }
]

semantic_predictions: List[Dict[str, Any]] = [
    {
        "image_id": "image_1",
        "predictions": [
            {
                "class": 0,
                "segmentation": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.2, "y": 0.2},
                    {"x": 0.3, "y": 0.3}
                ]
            }
        ]
    }
]

def test_classification_converter() -> None:
    """
    A function to test the ClassificationConverter to transform classification data to gesund format given the sample annotations and predictions.

    :return: None
    :rtype: None
    """
    converter = ClassificationConverter(
        annotations=class_annotations,
        predictions=class_predictions,
        image_width=512,
        image_height=512
    )
    annotations = converter._convert_annotations()
    predictions = converter._convert_predictions()

    try:
        for image_id in annotations:
            AnnotationModel(**annotations[image_id])
        for image_id in predictions:
            PredictionModel(**predictions[image_id])
    except ValidationError as e:
        pytest.fail(f"Validation failed: {e}")

def test_semantic_segmentation_converter() -> None:
    """
    A function to test the SemanticSegmentationConverter to transform semantic segmentation data to gesund format given the sample annotations and predictions.

    :return: None
    :rtype: None
    """
    converter = SemanticSegmentationConverter(
        annotations=semantic_annotations,
        predictions=semantic_predictions,
        image_width=512,
        image_height=512
    )
    annotations = converter._convert_annotations()
    predictions = converter._convert_predictions()

    try:
        for image_id in annotations:
            SemanticAnnotationModel(**annotations[image_id])
        for image_id in predictions:
            SemanticPredictionModel(**predictions[image_id])
    except ValidationError as e:
        pytest.fail(f"Validation failed: {e}")

def test_get_converter_yolo() -> None:
    """
    A function to get the YoloToGesund converter to transform the data to gesund format given the 'yolo' converter type.

    :return: None
    :rtype: None
    """
    from gesund.core._converters.converter_factory import ConverterFactory
    from gesund.core._converters.yolo_converter import YoloToGesund

    factory = ConverterFactory()
    converter = factory.get_converter("yolo")
    assert isinstance(converter, YoloToGesund)

def test_get_converter_invalid() -> None:
    """
    A function to get an invalid converter from the ConverterFactory and verify that it returns None.

    :return: None
    :rtype: None
    """
    from gesund.core._converters.converter_factory import ConverterFactory

    factory = ConverterFactory()
    converter = factory.get_converter("invalid")
    assert converter is None
