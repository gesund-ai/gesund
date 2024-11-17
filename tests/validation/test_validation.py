import os
import pytest

from gesund.validation import Validation


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


@pytest.mark.parametrize(
    "plot_config", [{"problem_type": "classification"}], indirect=True
)
def test_validation_initialization(plot_config):
    from gesund.core import UserInputParams

    data_dir = "./tests/_data/classification"
    classification_validation = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        problem_type="classification",
        data_format="json",
        json_structure_type="gesund",
        metadata_path=f"{data_dir}/test_metadata.json",
        return_dict=False,
        display_plots=False,
        store_plots=False,
        plot_config=plot_config,
        run_validation_only=True,
    )

    assert isinstance(classification_validation.user_params, UserInputParams)


@pytest.mark.parametrize(
    "plot_config", [{"problem_type": "classification"}], indirect=True
)
def test_validation_dataload(plot_config):
    from gesund.core import UserInputData

    data_dir = "./tests/_data/classification"
    classification_validation = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type="classification",
        data_format="json",
        json_structure_type="gesund",
        metadata_path=f"{data_dir}/test_metadata.json",
        return_dict=False,
        display_plots=False,
        store_plots=False,
        plot_config=plot_config,
        run_validation_only=True,
    )
    assert isinstance(classification_validation.data, UserInputData)


@pytest.mark.parametrize(
    "plot_config", [{"problem_type": "classification"}], indirect=True
)
def test_validation_plotmetrics_classification(plot_config):
    from gesund.core import ResultDataClassification

    data_dir = "./tests/_data/classification"
    classification_validation = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type="classification",
        data_format="json",
        json_structure_type="gesund",
        metadata_path=f"{data_dir}/test_metadata.json",
        return_dict=True,
        display_plots=True,
        store_plots=True,
        plot_config=plot_config,
        run_validation_only=False,
    )
    results = classification_validation.run()

    assert os.path.exists(classification_validation.output_dir) is True
    assert isinstance(results, ResultDataClassification)


@pytest.mark.parametrize(
    "plot_config", [{"problem_type": "object_detection"}], indirect=True
)
def test_validation_plotmetrics_object_detection(plot_config):

    data_dir = "./tests/_data/object_detection"
    obj_det_validation = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type="object_detection",
        data_format="json",
        json_structure_type="gesund",
        metadata_path=f"{data_dir}/test_metadata.json",
        return_dict=False,
        display_plots=True,
        store_plots=True,
        plot_config=plot_config,
        run_validation_only=False,
    )
    results = obj_det_validation.run()
    assert os.path.exists(obj_det_validation.output_dir) is True


@pytest.mark.parametrize(
    "plot_config", [{"problem_type": "semantic_segmentation"}], indirect=True
)
def test_validation_plotmetrics_segmentation(plot_config):
    data_dir = "./tests/_data/semantic_segmentation"
    seg_validation = Validation(
        annotations_path=f"{data_dir}/gesund_custom_format/annotation.json",
        predictions_path=f"{data_dir}/gesund_custom_format/prediction.json",
        class_mapping=f"{data_dir}/test_class_mappings.json",
        problem_type="semantic_segmentation",
        data_format="json",
        json_structure_type="gesund",
        metadata_path=f"{data_dir}/test_metadata.json",
        return_dict=False,
        display_plots=True,
        store_plots=True,
        plot_config=plot_config,
        run_validation_only=False,
    )
    results = seg_validation.run()
    assert os.path.exists(seg_validation.output_dir) is True
