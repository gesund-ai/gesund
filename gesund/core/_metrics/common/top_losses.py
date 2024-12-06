from typing import Union, Any, Dict, List, Optional
from gesund.core import metric_manager, plot_manager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Classification:
    def _validate_data(self, data: dict) -> bool:
        """
        Validates the data required for metric calculation and plotting.
        """
        # Basic validation checks
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")

        required_keys = ["prediction", "ground_truth", "loss"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Data must contain '{key}'.")

        if len(data["prediction"]) != len(data["ground_truth"]) or len(
            data["prediction"]
        ) != len(data["loss"]):
            raise ValueError(
                "Prediction, ground_truth, and loss must have the same length."
            )

        return True

    def apply_metadata(self, data: dict, metadata: dict) -> dict:
        """
        Applies metadata to the data for metric calculation and plotting.
        """
        return data

    def calculate(self, data: dict) -> dict:
        """
        Calculates the top losses for the given dataset.
        """
        # Validate the data
        self._validate_data(data)

        # Extract data
        predictions = pd.DataFrame(data["prediction"])
        ground_truth = pd.Series(data["ground_truth"])
        loss = pd.Series(data["loss"])
        metadata = pd.DataFrame(data.get("metadata", {}))
        class_mappings = data.get("class_mappings", {})

        # Apply metadata if provided
        if metadata is not None:
            data = self.apply_metadata(data, metadata)
            predictions = pd.DataFrame(data["prediction"])
            ground_truth = pd.Series(data["ground_truth"])
            loss = pd.Series(data["loss"])

        # Create meta_pred_true DataFrame
        meta_pred_true = pd.DataFrame(
            {"pred_categorical": predictions.idxmax(axis=1), "true": ground_truth}
        )

        # Calculate top losses
        top_losses_calculator = TopLosses(loss=loss, meta_pred_true=meta_pred_true)
        top_losses_df = top_losses_calculator.calculate_top_losses()

        result = {
            "top_losses": top_losses_df,
            "meta_pred_true": meta_pred_true,
            "predictions": predictions,
            "loss": loss,
            "metadata": metadata,
            "class_mappings": class_mappings,
        }

        return result


class TopLosses:
    def __init__(self, loss: pd.Series, meta_pred_true: pd.DataFrame) -> None:
        self.loss = loss
        self.meta_pred_true = meta_pred_true

    def calculate_top_losses(
        self, predicted_class: Optional[int] = None, top_k: int = 9
    ) -> pd.Series:
        """
        Calculates the top losses.

        :param predicted_class: Class of interest to calculate top losses.
        :type predicted_class: Optional[int]
        :param top_k: Number of top losses to return
        :type top_k: int
        :return: Series of top losses
        :rtype: pd.Series
        """
        if predicted_class is not None:
            indices = self.meta_pred_true[
                self.meta_pred_true["pred_categorical"] == predicted_class
            ].index
            sorted_top_loss = self.loss.loc[indices].sort_values(ascending=False)
        else:
            sorted_top_loss = self.loss.sort_values(ascending=False)
        return sorted_top_loss.head(top_k)


class PlotTopLosses:
    def __init__(self, data: dict):
        self.data = data
        self.top_losses = data["top_losses"]
        self.meta_pred_true = data["meta_pred_true"]
        self.predictions = data["predictions"]
        self.loss = data["loss"]
        self.metadata = data.get("metadata", pd.DataFrame())
        self.class_mappings = data.get("class_mappings", {})

    def _validate_data(self):
        """
        Validates the data required for plotting the top losses.
        """
        if "top_losses" not in self.data:
            raise ValueError("Data must contain 'top_losses'.")

    def save(self, filepath: str = "top_losses.png") -> str:
        """
        Saves the plot to a file.

        :param filepath: Path where the plot image will be saved
        :type filepath: str
        :return: Path where the plot image is saved
        :rtype: str
        """
        plt.savefig(filepath)
        return filepath

    def plot(self, top_k: int = 9):
        """
        Plots the top losses.

        :param top_k: Number of top losses to display
        :type top_k: int
        """
        # Validate the data
        self._validate_data()

        top_losses_indices = self.top_losses.index
        top_losses_values = self.top_losses.values

        # Prepare data for plotting
        predicted_logits = self.predictions.loc[top_losses_indices]
        if self.class_mappings:
            predicted_classes = predicted_logits.idxmax(axis=1).map(self.class_mappings)
            true_classes = self.meta_pred_true.loc[top_losses_indices]["true"].map(
                self.class_mappings
            )
        else:
            predicted_classes = predicted_logits.idxmax(axis=1)
            true_classes = self.meta_pred_true.loc[top_losses_indices]["true"]

        confidences = predicted_logits.max(axis=1)
        losses = self.top_losses.values

        # Create a DataFrame for easy plotting
        plot_data = pd.DataFrame(
            {
                "Loss": losses,
                "Ground Truth": true_classes.values,
                "Prediction": predicted_classes.values,
                "Confidence": confidences.values,
                "Index": top_losses_indices,
            }
        )

        # Plotting
        plt.figure(figsize=(12, 6))
        bars = plt.bar(
            plot_data["Index"].astype(str), plot_data["Loss"], color="salmon"
        )
        plt.xlabel("Sample Index")
        plt.ylabel("Loss")
        plt.title("Top Losses")

        # Annotate bars with Ground Truth and Prediction
        for bar, gt, pred in zip(
            bars, plot_data["Ground Truth"], plot_data["Prediction"]
        ):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"True: {gt}\nPred: {pred}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.xticks(rotation=90)
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


@metric_manager.register("classification.top_losses")
def calculate_top_losses(data: dict, problem_type: str):
    """
    A wrapper function to calculate the top losses metric.

    :param data: Dictionary of data: {"prediction": , "ground_truth": , "loss": }
    :type data: dict
    :param problem_type: Type of the problem
    :type problem_type: str
    :return: Calculated results
    :rtype: dict
    """
    metric_calculator = problem_type_map[problem_type]()
    result = metric_calculator.calculate(data)
    return result


@plot_manager.register("classification.top_losses")
def plot_top_losses(results: dict, save_plot: bool) -> Union[str, None]:
    """
    A wrapper function to plot the top losses.

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: Boolean value to save plot
    :type save_plot: bool
    :return: None or path to the saved plot
    :rtype: Union[str, None]
    """
    plotter = PlotTopLosses(data=results)
    plotter.plot()
    if save_plot:
        return plotter.save()
