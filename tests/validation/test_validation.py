# file to write the common logic
import pytest
import os

from gesund import Validation


@pytest.fixture
def plot_config(request):
    plot_configs = {
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
        },
        "object_detection": {
            "mixed_plot": {"mixed_plot": ["map10", "map50", "map75"], "threshold": 0.5},
            "top_misses": {"min_miou": 0.70, "top_n": 10},
            "confidence_histogram": {"confidence_histogram_labels": ["TP", "FP"]},
            "classbased_table": {
                "classbased_table_metrics": ["precision", "recall", "f1"],
                "threshold": 0.2,
            },
            "overall_metrics": {
                "overall_metrics_metrics": ["map", "mar"],
                "threshold": 0.5,
            },
            "blind_spot": {
                "blind_spot_Average": ["mAP@50", "mAP@10", "mAR@max=10", "mAR@max=100"],
                "threshold": 0.5,
            },
        },
        "semantic_segmentation": {
            "violin_graph": {"metrics": ["Acc", "Spec", "AUC"], "threshold": 0.5},
            "plot_by_meta_data": {
                "meta_data_args": [
                    "FalsePositive",
                    "Dice Score",
                    "mean Sensitivity",
                    "mean AUC",
                    "Precision",
                    "AverageHausdorffDistance",
                    "SimpleHausdorffDistance",
                ]
            },
            "overall_metrics": {
                "overall_args": ["mean AUC", "fwIoU", "mean Sensitivity"]
            },
            "classbased_table": {"classbased_table_args": 0.5},
            "blind_spot": {
                "blind_spot_args": [
                    "fwIoU",
                    "mean IoU",
                    "mean Sensitivity",
                    "mean Specificity",
                    "mean Kappa",
                    "mean AUC",
                    "",
                ]
            },
        },
    }
    return plot_configs[request.param["problem_type"]]


@pytest.fixture(scope="function")
def setup_and_teardown():
    # Setup code (optional)
    print("Setting up...")
    # ... your setup code here

    yield  # Yield control to the test function

    # Teardown code
    work_dir = os.getcwd()
    outputs_dir = os.path.join(work_dir, "outputs")
    shutil.rmtree(outputs_dir)


def test_validation_initialization(plot_config):
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
    )

    assert isinstance(validator.user_params, UserInputParams)
