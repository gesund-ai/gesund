# file to write the common logic
import pytest
import os

import warnings

warnings.filterwarnings("ignore")


@pytest.mark.parametrize(
    "plot_config", [{"problem_type": "object_detection"}], indirect=True
)
def test_validation_initialization(plot_config, setup_and_teardown):
    from gesund import Validation
    from gesund.core.schema import UserInputParams

    data_dir = "./tests/_data/object_detection"

    validator = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type="object_detection",
        data_format="json",
        json_structure_type="gesund",
        metadata_path=f"{data_dir}/test_metadata.json",
        plot_config=plot_config,
    )

    assert isinstance(validator.user_params, UserInputParams)


@pytest.mark.parametrize(
    "plot_config", [{"problem_type": "object_detection"}], indirect=True
)
def test_validation_dataload(plot_config, setup_and_teardown):
    from gesund import Validation
    from gesund.core.schema import UserInputData

    data_dir = "./tests/_data/object_detection"

    validator = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type="object_detection",
        data_format="json",
        json_structure_type="gesund",
        metadata_path=f"{data_dir}/test_metadata.json",
        plot_config=plot_config,
    )

    assert isinstance(validator.data, UserInputData)


@pytest.mark.parametrize(
    "plot_config, problem_type, threshold",
    [
        ({"problem_type": "object_detection"}, "classification", []),
        ({"problem_type": "object_detection"}, "object_detection", []),
        ({"problem_type": "object_detection"}, "semantic_segmentation", []),
    ],
)
def test_metrics_manager(plot_config, setup_and_teardown, problem_type, threshold):
    from gesund import Validation
    from gesund.validation._result import ValidationResult
    from gesund.core._managers.metric_manager import metric_manager

    data_dir = f"./tests/_data/{problem_type}"
    validator = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type=problem_type,
        data_format="json",
        json_structure_type="gesund",
        plot_config=plot_config,
        cohort_args={"selection_criteria": "random"},
        metric_args={"threshold": threshold},
    )

    validation_results = validator.run()

    assert isinstance(validation_results, ValidationResult) is True

    result_list = []
    for _metric in metric_manager.get_names(problem_type=problem_type):
        if _metric in validation_results.result:
            result_list.append(True)
        else:
            result_list.append(False)

    assert any(result_list) is True


@pytest.mark.parametrize(
    "plot_config, metric_name, cohort_id, threshold, problem_type",
    [
        (
            {"problem_type": "classification"},
            "lift_chart",
            None,
            [0.25, 0.5, 0.75],
            "classification",
        ),
        (
            {"problem_type": "classification"},
            "auc",
            None,
            [0.25, 0.5, 0.75],
            "classification",
        ),
        (
            {"problem_type": "classification"},
            "confusion_matrix",
            None,
            [0.25, 0.5, 0.75],
            "classification",
        ),
        (
            {"problem_type": "classification"},
            "most_confused",
            None,
            [0.25, 0.5, 0.75],
            "classification",
        ),
        (
            {"problem_type": "classification"},
            "stats_tables",
            None,
            [0.25, 0.5, 0.75],
            "classification",
        ),
        (
            {"problem_type": "classification"},
            "top_losses",
            None,
            [0.25, 0.5, 0.75],
            "classification",
        ),
        (
            {"problem_type": "classification"},
            "threshold",
            None,
            [0.25, 0.5, 0.75],
            "classification",
        ),
        (
            {"problem_type": "semantic_segmentation"},
            "iou_distribution",
            None,
            [],
            "semantic_segmentation",
        ),
        (
            {"problem_type": "semantic_segmentation"},
            "predicted_distribution",
            None,
            [],
            "semantic_segmentation",
        ),
        (
            {"problem_type": "semantic_segmentation"},
            "dice_distribution",
            None,
            [],
            "semantic_segmentation",
        ),
        (
            {"problem_type": "semantic_segmentation"},
            "top_losses",
            None,
            [],
            "semantic_segmentation",
        ),
        (
            {"problem_type": "object_detection"},
            "average_precision",
            None,
            0.5,
            "object_detection",
        ),
        (
            {"problem_type": "object_detection"},
            "average_precision",
            None,
            [0, 0.25, 0.5, 0.75, 1],
            "object_detection",
        ),
        (
            {"problem_type": "object_detection"},
            "top_losses",
            None,
            [],
            "object_detection",
        ),
        (
            {"problem_type": "object_detection"},
            "predicted_distribution",
            None,
            [],
            "object_detection",
        ),
        (
            {"problem_type": "object_detection"},
            "confidence_distribution",
            None,
            [],
            "object_detection",
        ),
    ],
)
def test_plot_manager_single_metric(
    plot_config, metric_name, cohort_id, threshold, setup_and_teardown, problem_type
):
    """
    A function to test validation run and plotting of the metrics
    """
    from gesund import Validation
    from gesund.validation._result import ValidationResult
    from gesund.core._managers.metric_manager import metric_manager
    from gesund.core._managers.plot_manager import plot_manager

    data_dir = f"./tests/_data/{problem_type}"
    validator = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type=problem_type,
        data_format="json",
        json_structure_type="gesund",
        plot_config=plot_config,
        cohort_args={"selection_criteria": "random"},
        metric_args={"threshold": threshold},
    )

    validation_results = validator.run()
    assert isinstance(validation_results, ValidationResult) is True
    assert metric_name in metric_manager.get_names(problem_type=problem_type)
    assert metric_name in plot_manager.get_names(problem_type=problem_type)
    assert metric_name in validation_results.result

    file_name = f"{problem_type}_{metric_name}.png"
    validation_results.plot(
        metric_name=metric_name,
        save_plot=True,
        cohort_id=cohort_id,
        file_name=file_name,
    )

    if metric_name != "stats_tables":
        if cohort_id:
            path_to_check = f"plots/{cohort_id}_{file_name}"
        else:
            path_to_check = f"plots/{file_name}"
        assert os.path.exists(path_to_check) is True
