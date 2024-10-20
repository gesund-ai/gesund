# Validation Metrics Library

## Overview

This library provides tools for calculating validation metrics for predictions and annotations in machine learning workflows. It includes a command-line tool for computing and displaying validation metrics.

## Installation

To use this library, ensure you have the necessary dependencies installed in your environment. You can install them via `pip`:

```sh
pip install .
```

## Usage

### Command-Line Tool

The primary script for running validation metrics is `run_metrics.py`. This script calculates validation metrics based on JSON files containing predictions and annotations.

#### Arguments

- `annotations` (required): Path to the JSON file containing annotation data.
- `predictions` (required): Path to the JSON file containing prediction data.
- `class_mappings` (required): Path to the JSON file containing class_mappings data.
- `problem_type` (required): Problem type that Validation is being run for .e.g. `classification`, `semantic_segmentation`, `instance_segmentation`, `object_detection`


#### Example

**Basic Usage**:

   ```sh
   run_metrics --annotations test_data/gesund_custom_format/gesund_custom_format_annotations_classification.json --predictions test_data/gesund_custom_format/gesund_custom_format_predictions_classification.json --class_mappings test_data/test_class_mappings.json --problem_type classification --format gesund_custom_format
   ```

### Example JSON Inputs

The library supports annotations and predictions in the following formats:
- `COCO`
- `YOLO`
- `Gesund Custom Format`

The format for `Gesund Custom Format` is shown below under `Example JSON Inputs`.

- **Annotations (`test_data/gesund_custom_format/gesund_custom_format_annotations_classification.json`)**:
  ```json
  {
  "664df1bf782d9eb107789013": {
    "image_id": "664df1bf782d9eb107789013",
    "annotation": [
      {
        "id": "664dfb2085d8059c72b7b24a",
        "label": 0
      }
    ]
  },

  "664df1bf782d9eb107789015": {
    "image_id": "664df1bf782d9eb107789015",
    "annotation": [
      {
        "id": "664dfb2085d8059c72b7b24d",
        "label": 1
      }
    ]
  },
  ...
  }
  ```

- **Predictions (`test_data/gesund_custom_format/gesund_custom_format_predictions_classification.json`)**:
  ```json
  {
  "664df1bf782d9eb107789013": {
    "image_id": "664df1bf782d9eb107789013",
    "prediction_class": 1,
    "confidence": 0.731047693767988,
    "logits": [
      0.2689523062320121,
      0.731047693767988
    ],
    "loss": 16.11764907836914
  },

  "664df1bf782d9eb107789015": {
    "image_id": "664df1bf782d9eb107789015",
    "prediction_class": 1,
    "confidence": 0.7308736572776326,
    "logits": [
      0.26912634272236735,
      0.7308736572776326
    ],
    "loss": 0.007578411139547825
  },
  ...
  }
  ```

- **Class Mappings (`test_data/test_class_mappings.json`)**:
  ```json
  {"0": "normal", "1": "pneumonia"}
  ```


### Example Outputs
#### Console Output

Only the Highlighted Overall Metrics are printed to the console. The output on the consol should look like so:

```json
Validation Metrics:
----------------------------------------
Accuracy:
    Validation: 0.4375
    Confidence_Interval: 0.2656 to 0.6094
----------------------------------------
Micro F1:
    Validation: 0.4375
    Confidence_Interval: 0.2656 to 0.6094
----------------------------------------
Macro F1:
    Validation: 0.4000
    Confidence_Interval: 0.2303 to 0.5697
----------------------------------------
AUC:
    Validation: 0.3996
    Confidence_Interval: 0.2299 to 0.5693
----------------------------------------
Precision:
    Validation: 0.4343
    Confidence_Interval: 0.2625 to 0.6060
----------------------------------------
Sensitivity:
    Validation: 0.4549
    Confidence_Interval: 0.2824 to 0.6274
----------------------------------------
Specificity:
    Validation: 0.4549
    Confidence_Interval: 0.2824 to 0.6274
----------------------------------------
Matthews C C:
    Validation: -0.1089
    Confidence_Interval: 0.0010 to 0.2168
----------------------------------------
----------------------------------------
All Graphs and Plots Metrics saved in JSONs.
----------------------------------------
```

#### Output JSON Files

All output JSON files for all graphs and plots will be present in the `outputs` dir, under the randomly assigned `{batch_job_id}` dir.

#### COCO Format

It is to be noted that COCO format is traditionally used for object detection, instance segmentation, and keypoint detection, but it is not designed for image classification. Therefore, we have adapted COCO-like structures for classification tasks. 

Sample format can be seen below:

- **Annotations (`test_data/coco/coco_annotations_classification.json`)**:
```json
{
    "info": {},
    "licenses": [],
    "categories": [
        {
            "id": 0,
            "name": "normal",
            "supercategory": "medical conditions"
        },
        {
            "id": 1,
            "name": "pneumonia",
            "supercategory": "medical conditions"
        }
    ],
    "images": [
        {
            "id": "664df1bf782d9eb107789013",
            "file_name": "image_1.jpg",
            "width": 240,
            "height": 240
        },
        {
            "id": "664df1bf782d9eb107789015",
            "file_name": "image_2.jpg",
            "width": 240,
            "height": 240
        },
        {
            "id": "664df1bf782d9eb107789014",
            "file_name": "image_3.jpg",
            "width": 240,
            "height": 240
        },
        ...
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": "664df1bf782d9eb107789013",
            "category_id": 0,
            "bbox": [],
            "area": 224,
            "iscrowd": 0
        },
        {
            "id": 2,
            "image_id": "664df1bf782d9eb107789015",
            "category_id": 1,
            "bbox": [],
            "area": 224,
            "iscrowd": 0
        },
        {
            "id": 3,
            "image_id": "664df1bf782d9eb107789014",
            "category_id": 1,
            "bbox": [],
            "area": 224,
            "iscrowd": 0
        },
        ...
    ]
  }
  ```


- **Predictions (`test_data/coco_predictions_classification.json`)**:
```json
[
    {
        "image_id": "664df1bf782d9eb107789013",
        "category_id": 1,
        "score": 0.731047693767988,
        "loss": 16.11764907836914
      },
      {
        "image_id": "664df1bf782d9eb107789015",
        "category_id": 1,
        "score": 0.7308736572776326,
        "loss": 0.007578411139547825
      },
      {
        "image_id": "664df1bf782d9eb107789014",
        "category_id": 1,
        "score": 0.7310579660592649,
        "loss": 0.000025339495550724678
      },
      ...
      ]
  ```