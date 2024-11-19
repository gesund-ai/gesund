import pprint
from gesund.validation import run_metrics, plotting_metrics
import json


def main():
    import os

    print(os.getcwd())
    parent_folder = os.getcwd()
    problem_type = "semantic_segmentation"
    format_name = "gesund_custom_format"
    file_name_ant = "gesund_custom_format_annotations_sem_segm"
    file_name_pred = "gesund_custom_format_predictions_sem_segm"
    args = {
        "annotations_json_path": "{}/test_data/{}/{}/{}.json".format(
            parent_folder, problem_type, format_name, file_name_ant
        ),
        "predictions": "{}/test_data/{}/{}/{}.json".format(
            parent_folder, problem_type, format_name, file_name_pred
        ),
        "class_mappings": "{}/test_data/{}/test_class_mappings.json".format(
            parent_folder, problem_type
        ),
        "problem_type": problem_type,
        "format": format_name,
        "write_results_to_json": True,
    }

    result = run_metrics(args)
    with open("{}_results.json".format(problem_type), "w") as f:
        json.dump(result, f, indent=4)

    args["plot_configs"] = {
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
        "overall_metrics": {"overall_args": ["mean AUC", "fwIoU", "mean Sensitivity"]},
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
    }
    plotting_metrics(result, args)


if __name__ == "__main__":
    main()
