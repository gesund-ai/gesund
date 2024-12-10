import os
from pydantic import BaseModel, field_validator
from typing import List, Dict, Union, Optional, ClassVar

from ._exceptions import InputError


class UserInputParams(BaseModel):
    annotations_path: str
    predictions_path: str
    class_mapping: Union[str, dict]
    problem_type: str
    json_structure_type: str
    data_format: str
    plot_config: dict
    metadata_path: Optional[str] = None

    allowed_values: ClassVar[dict] = {
        "problem_type": ["classification", "object_detection", "semantic_segmentation"],
        "json_structure_type": ["gesund", "coco", "yolo"],
        "data_format": ["json"],
    }

    @field_validator("annotations_path")
    def validate_annotations_path(cls, annotations_path):
        if os.path.exists(annotations_path):
            print("annotations path validated !")
        else:
            raise InputError(msg="Annotations path is invalid")
        return annotations_path

    @field_validator("predictions_path")
    def validate_predictions_path(cls, predictions_path):
        if os.path.exists(predictions_path):
            print("predictions path validated !")
        else:
            raise InputError(msg="Predictions path is invalid")
        return predictions_path

    @field_validator("metadata_path")
    def validate_metadata_path(cls, metadata_path):
        if metadata_path:
            if os.path.exists(metadata_path):
                print("metadata path validated !")
            else:
                raise InputError(msg="Metadata path is invalid")
        else:
            print("No metadata path provided.")
        return metadata_path

    @field_validator("class_mapping")
    def validate_class_mapping(cls, class_mapping):
        if isinstance(class_mapping, str):
            if os.path.exists(class_mapping):
                print("class mapping file validated !")
            else:
                raise InputError(msg="Class mapping path is invalid")
        else:
            print("Class mapping is dict input")
        return class_mapping

    @field_validator("problem_type")
    def validate_problem_type(cls, problem_type):
        if problem_type not in cls.allowed_values["problem_type"]:
            raise InputError("Invalid problem type")
        else:
            print("Problem Type validated !")
        return problem_type

    @field_validator("json_structure_type")
    def validate_json_structure_type(cls, json_structure_type):
        if json_structure_type not in cls.allowed_values["json_structure_type"]:
            raise InputError("Invalid json structure type")
        else:
            print("JSON structure type validated!")
        return json_structure_type

    @field_validator("data_format")
    def validate_data_format(cls, data_format):
        if data_format not in cls.allowed_values["data_format"]:
            raise InputError("Invalid data format")
        else:
            print("Data format validated!")
        return data_format


class UserInputData(BaseModel):
    prediction: Union[List[Dict], Dict]
    annotation: Union[List[Dict], Dict]
    class_mapping: Dict
    metadata: Optional[Union[List[Dict], Dict]] = None
    converted_prediction: Optional[Union[List[Dict], Dict]] = None
    converted_annotation: Optional[Union[List[Dict], Dict]] = None
    was_converted: Optional[bool] = False
