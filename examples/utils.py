import os
from typing import Dict, Any

def callable_plot_config(problem_type: str) -> Dict[str, Any]:
    """
    Returns a dictionary of plot configurations for a given problem type.

    Args:
        problem_type (str): The type of problem (e.g., "classification", "object_detection", "semantic_segmentation").

    Returns:
        Dict[str, Any]: A dictionary containing plot configurations for the specified problem type.

    Raises:
        ValueError: If the problem_type is not supported.
    """
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
            "prediction_dataset_distribution": {},
            "most_confused_bar": {},
            "confidence_histogram_scatter_distribution": {},
            "lift_chart": {},
            "auc": {},
            "threshold": {},
            "top_losses": {},
            "most_confused": {},
            "confusion_matrix": {},
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
            "top_losses": {},
            "predicted_distribution": {},
            "confidence_distribution": {},
            "average_precision": {},
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
            "predicted_distribution": {},
            "iou_distribution": {},
            "dice_distribution": {},
            "top_losses": {},
        },
    }
    if problem_type not in plot_configs:
        raise ValueError(
            f"Unsupported problem_type: {problem_type}. "
            f"Supported types are: {list(plot_configs.keys())}"
        )

    return plot_configs[problem_type]