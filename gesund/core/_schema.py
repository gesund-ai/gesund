import os
from pydantic import BaseModel, field_validator
from typing import List, Dict, Union, Optional, ClassVar

from ._exceptions import InputError


class UserInputParams(BaseModel):
    annotations_path: str
    predictions_path: str
    problem_type: str
    json_structure_type: str
    data_format: str
    allowed_values: dict
    metadata_path: Optional[str] = None
    return_dict: Optional[bool] = False
    store_json: Optional[bool] = False
    display_plots: Optional[bool] = False
    store_plots: Optional[bool] = False

    @field_validator("annotations_path")
    def validate_annotations_path(cls, annotations_path):
        if os.path.exists(annotations_path):
            print("annotations path validated !")
        else:
            raise InputError(msg="Annotations path is invalid")
    
    @field_validator("predictions_path")
    def validate_predictions_path(cls, predictions_path):
        if os.path.exists(predictions_path):
            print("predictions path validated !")
        else:
            raise InputError(msg="Predictions path is invalid")
    
    @field_validator("metadata_path")
    def validate_metadata_path(cls, metadata_path):
        if metadata_path:
            if os.path.exists(metadata_path):
                print("metadata path validated !")
            else:
                raise InputError(msg="Metadata path is invalid")
        else:
            print("No metadata path provided.")
    
    @field_validator("problem_type")
    def validate_problem_type(cls, problem_type):
        if problem_type not in cls.allowed_values["problem_type"]:
            raise InputError("Invalid problem type")
        else:
            print("Problem Type validated !")
    
    @field_validator("json_structure_type")
    def validate_json_structure_type(cls, json_structure_type):
        if json_structure_type not in cls.allowed_values["json_structure_type"]:
            raise InputError("Invalid json structure type")
        else:
            print(" JSON structure type validated !")
    
    @field_validator("data_format")
    def validate_json_structure_type(cls, data_format):
        if data_format not in cls.allowed_values["data_format"]:
            raise InputError("Invalid data format")
        else:
            print("Data format validated!")


class UserInputData(BaseModel):
    prediction: Union[List[Dict], Dict]
    annotation: Union[List[Dict], Dict]
    class_mapping: Dict
    converted_prediction: Optional[Union[List[Dict], Dict]]
    converted_annotation: Optional[Union[List[Dict], Dict]]
    was_converted: bool = False


class ResultData(BaseModel):
    pass