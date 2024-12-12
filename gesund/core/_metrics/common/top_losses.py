import os
from typing import Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gesund.core import metric_manager, plot_manager

COHORT_SIZE_LIMIT = 2


def categorize_age(age):
    if age < 18:
        return "Child"
    elif 18 <= age < 30:
        return "Young Adult"
    elif 30 <= age < 60:
        return "Adult"
    else:
        return "Senior"


class Classification:
    def _validate_data(self, data: dict) -> bool:
        """
        A function to validate the data that is required for metric calculation and plotting.

        :param data: The input data required for calculation, {"prediction":, "ground_truth": , "metadata":}
        :type data: dict

        :return: Status if the data is valid
        :rtype: bool
        """
        # Basic validation checks
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        required_keys = ["prediction", "ground_truth"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Data must contain '{key}'.")

        if len(data["prediction"]) != len(data["ground_truth"]):
            raise ValueError("Prediction and ground_truth must have the same length.")

        if (
            len(
                set(list(data["prediction"].keys())).difference(
                    set(list(data["ground_truth"].keys()))
                )
            )
            > 0
        ):
            raise ValueError("Prediction and ground_truth must have the same keys.")

        return True

    def __preprocess(self, data: dict, get_logits=False):
        """
        Preprocesses the data

        :param data: dictionary containing the data prediction, ground truth, metadata
        :type data: dict
        :param get_logits: in case of multi class classification set to True
        :type get_logits: boolean

        :return: data tuple
        :rtype: tuple
        """
        prediction, ground_truth = [], []
        for image_id in data["ground_truth"]:
            sample_gt = data["ground_truth"][image_id]
            sample_pred = data["prediction"][image_id]
            ground_truth.append(sample_gt["annotation"][0]["label"])

            if get_logits:
                prediction.append(sample_pred["logits"])
            else:
                prediction.append(sample_pred["prediction_class"])
        return (np.asarray(prediction), np.asarray(ground_truth))

    def apply_metadata(self, data: dict) -> dict:
        """
        Applies metadata to the data for metric calculation and plotting.

        :param data: The input data required for calculation, {"prediction":, "ground_truth": , "metadata":}
        :type data: dict

        :return: Filtered dataset
        :rtype: dict
        """
        # TODO:: This function could be global to be applied across metrics

        df: pd.DataFrame = pd.DataFrame.from_records(data["metadata"])
        cohorts_data = {}

        metadata_columns = df.columns.tolist()
        metadata_columns.remove("image_id")
        lower_case = {i: i.lower() for i in metadata_columns}
        df = df.rename(columns=lower_case)

        if "age" in list(lower_case.values()):
            df["age"] = df["age"].apply(categorize_age)

        for grp, subset_data in df.groupby(list(lower_case.values())):
            grp_str = ",".join([str(i) for i in grp])

            if subset_data.shape[0] < COHORT_SIZE_LIMIT:
                print(
                    f"Warning - grp excluded - {grp_str} cohort size < {COHORT_SIZE_LIMIT}"
                )
            else:
                image_ids = set(subset_data["image_id"].to_list())
                filtered_data = {
                    "prediction": {
                        i: data["prediction"][i]
                        for i in data["prediction"]
                        if i in image_ids
                    },
                    "ground_truth": {
                        i: data["ground_truth"][i]
                        for i in data["ground_truth"]
                        if i in image_ids
                    },
                    "metadata": subset_data,
                    "class_mapping": data["class_mapping"],
                }
                cohorts_data[grp_str] = filtered_data

        return data

    def __calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        """
        A function to calculate the metrics

        :param data: data dictionary containing data
        :type data: dict
        :param class_mapping: a dictionary with class mapping labels
        :type class_mapping: dict

        :return: results calculated
        :rtype: dict
        """
        class_order = [int(i) for i in class_mapping.keys()]

        if len(class_order) > 2:
            prediction, ground_truth = self.__preprocess(data, get_logits=True)
        else:
            prediction, ground_truth = self.__preprocess(data)

        # TODO: Check the prev code for the below

        data = self.apply_metadata(data, metadata)
        predictions = pd.DataFrame(data["prediction"])
        ground_truth = pd.Series(data["ground_truth"])
        loss = pd.Series(data["loss"])
        metadata = pd.DataFrame(data.get("metadata", {}))

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
            "class_mapping": class_mapping,
        }

        return result

    def calculate(self, data: dict) -> dict:
        """
        Calculates the top losses for the given dataset.

        :param data: The input data required for calculation and plotting
                {"prediction":, "ground_truth": , "metadata":, "class_mappings":}
        :type data: dict

        :return: Calculated metric results
        :rtype: dict
        """
        result = {}

        # Validate the data
        self._validate_data(data)
        metadata = data.get("metadata")

        if metadata:
            cohort_data = self.apply_metadata(data)
            for _cohort_key in cohort_data:
                result[_cohort_key] = self.__calculate_metrics(
                    cohort_data[_cohort_key], data.get("class_mapping")
                )
        else:
            result = self.__calculate_metrics(data, data.get("class_mapping"))

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

    def save(self, fig: Figure, filename: str) -> str:
        """
        Saves the plot to a file.

        :param filename: Path where the plot image will be saved
        :type filename: str

        :return: Path where the plot image is saved
        :rtype: str
        """
        dir_path = "plots"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filepath = f"{dir_path}/{filename}"
        fig.savefig(filepath, format="png")
        return filepath

    def plot(self, top_k: int = 9) -> Figure:
        """
        Plots the top losses.

        :param top_k: Number of top losses to display
        :type top_k: int
        :return: Matplotlib Figure object
        :rtype: Figure
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
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(plot_data["Index"].astype(str), plot_data["Loss"], color="salmon")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Loss")
        ax.set_title("Top Losses")

        # Annotate bars with Ground Truth and Prediction
        for bar, gt, pred in zip(
            bars, plot_data["Ground Truth"], plot_data["Prediction"]
        ):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"True: {gt}\nPred: {pred}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.xticks(rotation=90)
        plt.tight_layout()
        return fig


class SemanticSegmentation(Classification):
    pass


class ObjectDetection(Classification):
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
def plot_top_losses(
    results: dict, save_plot: bool, file_name: str = "top_losses.png"
) -> Union[str, None]:
    """
    A wrapper function to plot the top losses chart.

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: Boolean value to save plot
    :type save_plot: bool
    :param file_name: Name of the file to save the plot
    :type file_name: str

    :return: None or path to the saved plot
    :rtype: Union[str, None]
    """
    plotter = PlotTopLosses(data=results)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
