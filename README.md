<h1 align="center">
  <img src="gesund/assets/gesund_logo.png" width="300" alt="Gesund Logo">
</h1><br>

# Validation Metrics Library

[![Test  Workflow](https://github.com/gesund-ai/gesund/actions/workflows/test.yml/badge.svg)](https://github.com/gesund-ai/gesund/actions/workflows/test.yml)
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

from gesund import Validation
from gesund.validation._result import ValidationResult
from gesund.core._managers.metric_manager import metric_manager
from gesund.core._managers.plot_manager import plot_manager

# Call the default configuration from utils
from utils import callable_plot_config

# Select your problem type {"classification", "object_detection", "semantic_segmentation"}

# example usage for problem type
problem_type = "classification"
plot_configuration = callable_plot_config(problem_type)
metric_name = "lift_chart"
cohort_id = None
data_dir = f"./tests/_data/{problem_type}"

# create a class instance
validator = Validation(
    annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
    predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
    class_mapping=f"{data_dir}/test_class_mappings.json",
    problem_type=problem_type,
    data_format="json",
    json_structure_type="gesund",
    plot_config=plot_configuration,
    cohort_args={"selection_criteria": "random"},
    metric_args={"threshold": [0.25, 0.5, 0.75]},
)

# run the validation workflow
validation_results = validator.run()
# save the results 
validation_results.save(metric_name)
# plot the results
validation_results.plot(metric_name="auc", save_plot=False, cohort_id=cohort_id)

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

