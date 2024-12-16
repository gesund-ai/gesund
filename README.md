<h1 align="center">
  <img src="gesund/assets/gesund_logo.png" width="300" alt="Gesund Logo">
</h1><br>

# Validation Metrics Library

[![PyPi](https://img.shields.io/pypi/v/gesund)](https://pypi.org/project/gesund)
[![PyPI Downloads](https://img.shields.io/pypi/dm/gesund.svg?label=PyPI%20downloads)](
https://pypi.org/project/gesund/)




This library provides tools for calculating validation metrics for predictions and annotations in machine learning workflows. It includes a command-line tool for computing and displaying validation metrics.

- **Documentation:**  https://gesund-ai.github.io
- **Source code:** https://github.com/gesund-ai/gesund
- **Bug reports:** https://github.com/gesund-ai/gesund/issues
- **Examples :** https://github.com/gesund-ai/gesund/tree/main/gesund/examples


## Installation

To use this library, ensure you have the necessary dependencies installed in your environment. You can install them via `pip`:

```sh
pip install gesund==latest_version
pip install pycocotools@git+https://github.com/HammadK44/cocoapi.git@Dev#subdirectory=PythonAPI/
```

## Basic Usage

```python

# import the library
from gesund.validation import Validation

# set up the configs
data_dir = "./tests/_data/classification"
plot_config = {
    "classification": {
        "class_distributions": {
            "metrics": ["normal", "pneumonia"],
            "threshold": 10,
        },
        "blind_spot": {"class_type": ["Average", "1", "0"]},
        "performance_by_threshold": {
            "graph_type": "graph_1",
            "metrics": [
                "F1",
                "Sensitivity",
                "Specificity",
                "Precision",
                "FPR",
                "FNR",
            ],
            "threshold": 0.2,
        },
        "roc": {"roc_class": ["normal", "pneumonia"]},
        "precision_recall": {"pr_class": ["normal", "pneumonia"]},
        "confidence_histogram": {"metrics": ["TP", "FP"], "threshold": 0.5},
        "overall_metrics": {"metrics": ["AUC", "Precision"], "threshold": 0.2},
        "confusion_matrix": {},
        "prediction_dataset_distribution": {},
        "most_confused_bar": {},
        "confidence_histogram_scatter_distribution": {},
        "lift_chart": {},
    }
}

# create a class instance
classification_validation = Validation(
    annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
    predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
    problem_type="classification",
    class_mapping=f"{data_dir}/test_class_mappings.json",
    data_format="json",
    json_structure_type="gesund",
    metadata_path=f"{data_dir}/test_metadata.json",
    return_dict=True,
    display_plots=False,
    store_plots=False,
    plot_config=plot_config,
    run_validation_only=True
)

# run the validation workflow
results = classification_validation.run()

# explore the results
print(results)

```


## Code of Conduct


We are committed to fostering a welcoming and inclusive community. Please adhere to the following guidelines when contributing to this project:

- **Respect**: Treat everyone with respect and consideration. Harassment or discrimination of any kind is not tolerated.
- **Collaboration**: Be open to collaboration and constructive criticism. Offer feedback gracefully and accept feedback in the same manner.
- **Inclusivity**: Use inclusive language and be mindful of different perspectives and experiences.
- **Professionalism**: Maintain a professional attitude in all project interactions.

By participating in this project, you agree to abide by this Code of Conduct. If you witness or experience any behavior that violates these guidelines, please contact the project maintainers.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

