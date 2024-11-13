import sys
import os
import pytest

print(os.getcwd().split("tests")[0])
os.environ["PYTHONPATH"] = os.getcwd().split("tests")[0]


from gesund.validation import Validation
from gesund.core import UserInputParams, UserInputData

def test_validation_initialization():
    data_dir = "./tests/_data/classification"
    classification_validation = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        problem_type="classification",
        data_format="json",
        json_structure_type="gesund",
        metadata_path=f"{data_dir}/test_metadata.json",
        store_json=False,
        return_dict=False,
        display_plots=False,
        store_plots=False
    )

    assert isinstance(classification_validation.user_params, UserInputParams)
    assert isinstance(classification_validation.data, UserInputData)
