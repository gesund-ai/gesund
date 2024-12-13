# file to write the common logic
import pytest
import os

from gesund import Validation


@pytest.mark.parametrize(
    "plot_config", [{"problem_type": "classification"}], indirect=True
)
def test_validation_initialization(plot_config, setup_and_teardown):
    from gesund.core.schema import UserInputParams

    data_dir = "./tests/_data/classification"

    validator = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type="classification",
        data_format="json",
        json_structure_type="gesund",
        metadata_path=f"{data_dir}/test_metadata.json",
        plot_config=plot_config,
    )

    assert isinstance(validator.user_params, UserInputParams)


@pytest.mark.parametrize(
    "plot_config", [{"problem_type": "classification"}], indirect=True
)
def test_validation_dataload(plot_config, setup_and_teardown):
    from gesund.core.schema import UserInputData

    data_dir = "./tests/_data/classification"

    validator = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type="classification",
        data_format="json",
        json_structure_type="gesund",
        metadata_path=f"{data_dir}/test_metadata.json",
        plot_config=plot_config,
    )

    assert isinstance(validator.data, UserInputData)


@pytest.mark.parametrize(
    "plot_config", [{"problem_type": "classification"}], indirect=True
)
def test_metrics_manager(plot_config, setup_and_teardown):
    from gesund import Validation
    from gesund.validation._result import ValidationResult
    from gesund.core._managers.metric_manager import metric_manager

    problem_type = "classification"
    data_dir = f"./tests/_data/{problem_type}"
    validator = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type=problem_type,
        data_format="json",
        json_structure_type="gesund",
        # metadata_path=f"{data_dir}/test_metadata_new.json",
        plot_config=plot_config,
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
    "plot_config", [{"problem_type": "classification"}], indirect=True
)
def test_plot_manager(plot_config, setup_and_teardown):
    from gesund import Validation
    from gesund.validation._result import ValidationResult
    from gesund.core._managers.metric_manager import metric_manager

    problem_type = "classification"
    data_dir = f"./tests/_data/{problem_type}"
    validator = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type=problem_type,
        data_format="json",
        json_structure_type="gesund",
        metadata_path=f"{data_dir}/test_metadata_new.json",
        plot_config=plot_config,
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

    validation_results.plot(save_plot=True)
    for _metric in metric_manager.get_names(problem_type=problem_type):
        assert os.path.exists(f"plots/{_metric}.png") is True


@pytest.mark.parametrize(
    "plot_config", [{"problem_type": "classification"}], indirect=True
)
def test_plot_manager_single_metric(plot_config, setup_and_teardown):
    from gesund import Validation
    from gesund.validation._result import ValidationResult
    from gesund.core._managers.metric_manager import metric_manager
    from gesund.core._managers.plot_manager import plot_manager

    problem_type = "classification"
    data_dir = f"./tests/_data/{problem_type}"
    validator = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type=problem_type,
        data_format="json",
        json_structure_type="gesund",
        metadata_path=f"{data_dir}/test_metadata_new.json",
        plot_config=plot_config,
    )

    metric_name = "confusion_matrix"
    validation_results = validator.run()
    assert isinstance(validation_results, ValidationResult) is True

    assert metric_name in metric_manager.get_names(problem_type=problem_type)
    assert metric_name in plot_manager.get_names(problem_type=problem_type)

    validation_results.plot(metric_name=metric_name, save_plot=True)
    assert os.path.exists(f"plots/{metric_name}.png") is True


"""
Usage

from gesund import Validation

validator = Validation(...)

results = validator.run()

results.plot(metric_name="str")

results.save(metric_name="str")

"""
