import os
from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Dict, Union


class InputParams(BaseModel):
    annotations_path: str
    predictions_path: str
    metadata_path: str
    problem_type: bool
    json_structure_type: str
    return_dict: bool
    store_json: bool
    display_plots: bool
    store_plots: bool

    @field_validator("annotations_path")
    def validate_annotations_path(cls, annotations_path):
        if os.path.exists(annotations_path):
            print("annotations path validated !")



input_data = {
    "annotations_path": "run_meta_validation.py",
    "predictions_path": "asdasd",
    "metadata_path": "asdas",
    "problem_type": False,
    "json_structure_type": "asdasd", 
    "return_dict": False,
    "store_json": False,
    "display_plots": False,
    "store_plots": False
}



input_model = InputParams(**input_data)

print(input_model)



input_data = {
    "annotations_path": "_schema.py",
    "predictions_path": "_schema.py",
    "metadata_path": "_schema.py",
    "problem_type": "classification",
    "json_structure_type": "gesund", 
    "return_dict": False,
    "store_json": False,
    "display_plots": False,
    "store_plots": False
}

gesund_validation = Validation(**input_data)

print(gesund_validation.params)