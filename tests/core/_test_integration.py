# FILE: _test_integration.py

import pytest
from gesund.core._converters.yolo_converter import (
    ClassificationConverter,
    SemanticSegmentationConverter,
)
from typing import List, Dict, Any


def test_classification_converter_integration() -> None:
    """
    Integration test for the ClassificationConverter.
    """
    class_annotations: List[Dict[str, Any]] = [
        {"image_id": "image_1", "annotations": [{"class": 0}, {"class": 1}]}
    ]

    class_predictions: List[Dict[str, Any]] = [
        {
            "image_id": "image_1",
            "predictions": [
                {"class": 0, "confidence": 0.9, "loss": 0.1},
                {"class": 1, "confidence": 0.8, "loss": 0.2},
            ],
        }
    ]

    converter = ClassificationConverter(
        annotations=class_annotations,
        predictions=class_predictions,
        image_width=512,
        image_height=512,
    )
    annotations = converter._convert_annotations()
    predictions = converter._convert_predictions()

    assert annotations == {
        "image_1": {"image_id": "image_1", "annotation": [{"label": 0}, {"label": 1}]}
    }

    expected_predictions = {
        "image_1": {
            "image_id": "image_1",
            "prediction_class": 1,
            "confidence": pytest.approx(0.8),
            "logits": pytest.approx([0.2, 0.8]),
            "loss": pytest.approx(0.2),
        }
    }
    assert predictions == expected_predictions


def test_semantic_segmentation_converter_integration() -> None:
    """
    Integration test for the SemanticSegmentationConverter.
    """
    semantic_annotations: List[Dict[str, Any]] = [
        {
            "image_id": "image_1",
            "annotations": [
                {
                    "class": 0,
                    "segmentation": [
                        {"x": 0.1, "y": 0.1},
                        {"x": 0.2, "y": 0.2},
                        {"x": 0.3, "y": 0.3},
                    ],
                }
            ],
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
                        {"x": 0.3, "y": 0.3},
                    ],
                }
            ],
        }
    ]

    converter = SemanticSegmentationConverter(
        annotations=semantic_annotations,
        predictions=semantic_predictions,
        image_width=512,
        image_height=512,
    )
    annotations = converter._convert_annotations()
    predictions = converter._convert_predictions()

    actual_annotation_rle = annotations["image_1"]["annotation"][0]["mask"]["mask"]
    actual_prediction_rle = predictions["image_1"]["masks"]["rles"][0]["rle"]

    expected_annotations = {
        "image_1": {
            "image_id": "image_1",
            "annotation": [
                {
                    "image_id": "image_1",
                    "label": 0,
                    "type": "mask",
                    "measurement_info": {
                        "objectName": "mask",
                        "measurement": "Segmentation",
                    },
                    "mask": {"mask": actual_annotation_rle},
                    "shape": [512, 512],
                    "window_level": None,
                }
            ],
        }
    }

    expected_predictions = {
        "image_1": {
            "image_id": "image_1",
            "masks": {"rles": [{"rle": actual_prediction_rle, "class": 0}]},
            "shape": [512, 512],
            "status": 200,
        }
    }

    assert annotations == expected_annotations
    assert predictions == expected_predictions
