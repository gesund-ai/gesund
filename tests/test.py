#!/usr/bin/env python
import json
from gesund.utils.io_utils import read_json, validate_json_format
from gesund.metrics.classification.create_validation import ValidationCreation

def check_json_fields(data, required_fields):
    """
    Check if each item in the JSON data contains all required fields.
    
    Args:
    data (dict): The JSON data to check.
    required_fields (list): List of fields that should be present in each item.
    
    Raises:
    AssertionError: If any field is missing in the data.
    """
    for item_id, item in data.items():
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            raise AssertionError(f"Item ID {item_id} is missing fields: {missing_fields}")

def test_create_validation_collection_data():
    batch_job_id = "test_batch_job_id"
    successful_batch_data_file = 'test_data/test_predictions_classification.json'
    annotation_data_file = 'test_data/test_annotations_classification.json'
    class_mappings_file = 'test_data/class_mappings.json'
    
    # Read data from JSON files
    successful_batch_data = read_json(successful_batch_data_file)
    annotation_data = read_json(annotation_data_file)
    class_mappings = read_json(class_mappings_file)

    # Define required fields for each type of data
    successful_batch_fields = ['image_id', 'confidence', 'logits', 'prediction_class', 'loss']
    annotation_fields = ['image_id', 'annotation']

    # Validate the format of the JSON data
    validate_json_format(successful_batch_data, successful_batch_fields)
    validate_json_format(annotation_data, annotation_fields)
    # Check if all required fields are present in the JSON files
    check_json_fields(successful_batch_data, successful_batch_fields)
    check_json_fields(annotation_data, annotation_fields)
    
    # Create ValidationCreation instance and process data
    validation = ValidationCreation(batch_job_id)
    validation_data = validation.create_validation_collection_data(successful_batch_data, annotation_data)
    metrics = validation.load(validation_data, class_mappings)

    # Print or assert results for further validation
    print(metrics)
    # Example assertions (customize based on your requirements)
    assert isinstance(validation_data, list)
    assert len(validation_data) > 0
    assert all("image_id" in item for item in validation_data)
    assert all("confidence" in item for item in validation_data)
    assert all("logits" in item for item in validation_data)
    assert all("prediction_class" in item for item in validation_data)
    assert all("ground_truth" in item for item in validation_data)
    assert all("created_timestamp" in item for item in validation_data)
    assert all("loss" in item for item in validation_data)

if __name__ == "__main__":
    test_create_validation_collection_data()

