from typing import Union, Any, Dict, List, Optional
from gesund.core import metric_manager, plot_manager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix


class Classification:
    def _validate_data(self, data: dict) -> bool:
        """
        Validates the data required for metric calculation and plotting.
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        required_keys = ["prediction", "ground_truth"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Data must contain '{key}'.")
        if len(data["prediction"]) != len(data["ground_truth"]):
            raise ValueError("Prediction and ground_truth must have the same length.")
        return True

    def apply_metadata(self, data: dict, metadata: dict) -> dict:
        """
        Applies metadata to the data for metric calculation and plotting.
        """
        return data

    def calculate(self, data: dict) -> dict:
        """
        Calculates the most confused classes for the given dataset.
        """
        self._validate_data(data)
        true = np.array(data["ground_truth"])
        pred_categorical = np.array(data["prediction"])
        metadata = data.get("metadata", None)

        if metadata is not None:
            data = self.apply_metadata(data, metadata)
            true = np.array(data["ground_truth"])
            pred_categorical = np.array(data["prediction"])

        class_mappings = data.get("class_mappings", None)
        if class_mappings:
            class_order = [int(i) for i in list(class_mappings.keys())]
        else:
            classes = np.unique(np.concatenate((true, pred_categorical)))
            class_order = classes.tolist()
            class_mappings = {int(i): str(i) for i in class_order}

        cm = confusion_matrix(true, pred_categorical, labels=class_order)
        confused_pairs = []
        for i, actual_class in enumerate(class_order):
            for j, pred_class in enumerate(class_order):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append(
                        {
                            "true": class_mappings[actual_class],
                            "predicted": class_mappings[pred_class],
                            "count": cm[i, j],
                        }
                    )
        confused_pairs = sorted(confused_pairs, key=lambda x: x["count"], reverse=True)

        result = {
            "confused_pairs": confused_pairs,
            "class_mappings": class_mappings,
            "class_order": class_order,
        }
        return result


class PlotMostConfused:
    def __init__(self, data: dict):
        self.data = data
        self.confused_pairs = data["confused_pairs"]

    def _validate_data(self):
        """
        Validates the data required for plotting.
        """
        if "confused_pairs" not in self.data:
            raise ValueError("Data must contain 'confused_pairs'.")

    def save(self, filepath: str = "most_confused.png") -> str:
        """
        Saves the plot to a file.
        """
        plt.savefig(filepath)
        return filepath

    def plot(self, top_k: int = 5):
        """
        Plots the most confused classes as a bar chart.
        """
        self._validate_data()
        df = pd.DataFrame(self.confused_pairs[:top_k])
        plt.figure(figsize=(10, 6))
        sns.barplot(x="count", y=df.index, data=df, orient="h")
        plt.yticks(
            ticks=df.index,
            labels=[
                f"True: {row['true']} | Predicted: {row['predicted']}"
                for _, row in df.iterrows()
            ],
        )
        plt.xlabel("Count")
        plt.ylabel("Confusion Pairs")
        plt.title("Most Confused Classes")
        plt.tight_layout()
        plt.show()


class SemanticSegmentation(Classification):
    pass


class ObjectDetection:
    pass


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("classification.most_confused")
def calculate_most_confused_metric(data: dict, problem_type: str):
    """
    Calculates the most confused classes metric.
    """
    metric_calculator = problem_type_map[problem_type]()
    result = metric_calculator.calculate(data)
    return result


@plot_manager.register("classification.most_confused")
def plot_most_confused(results: dict, save_plot: bool) -> Union[str, None]:
    """
    Plots the most confused classes.
    """
    plotter = PlotMostConfused(data=results)
    plotter.plot()
    if save_plot:
        return plotter.save()
