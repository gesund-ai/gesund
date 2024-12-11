from typing import Union
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager


def categorize_age(age):
    if age < 18:
        return "Child"
    elif 18 <= age < 30:
        return "Young Adult"
    elif 30 <= age < 60:
        return "Adult"
    else:
        return "Senior"


COHORT_SIZE_LIMIT = 2
DEBUG = True


class Classification:
    def _validate_data(self, data: dict) -> bool:
        """
        Validates the data required for metric calculation and plotting.

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

        # check for image ids samples in the ground truth and prediction
        if (
            len(
                set(list(data["prediction"].keys())).difference(
                    set(list(data["ground_truth"].keys()))
                )
            )
            > 0
        ):
            raise ValueError("Prediction and ground truth samples does not match.")

        # if metadata is given then need to check if the metadata fields are present for all the
        # images ids if not then simply state a warning and not apply a

        return True

    def __preprocess(self, data: dict, get_logits=False) -> tuple:
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

        # convert the metadata into pandas dataframe for better access
        df: pd.DataFrame = pd.DataFrame.from_records(data["metadata"])
        cohorts_data = {}

        # collect only the metadata columns
        metadata_columns = df.columns.tolist()
        metadata_columns.remove("image_id")
        lower_case = {i: i.lower() for i in metadata_columns}
        df = df.rename(columns=lower_case)

        # categorize age in metadata
        if "age" in list(lower_case.values()):
            df["age"] = df["age"].apply(categorize_age)

        # loop over group by to form cohorts with unique metadata
        for grp, subset_data in df.groupby(list(lower_case.values())):
            grp_str = ",".join([str(i) for i in grp])

            # it seems that
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
        class_order = [int(i) for i in list(class_mapping.keys())]

        # Calculate ROC and AUC
        # TODO: class wise auc and roc

        if len(class_order) > 2:
            # implementation of multi class classification
            # logits are required for multi class classification
            prediction, ground_truth = self.__preprocess(data, get_logits=True)
            # TODO: multi class auc - roc calculation

        else:
            prediction, ground_truth = self.__preprocess(data)
            fpr, tpr, thresholds = roc_curve(prediction, ground_truth)
            auc_score = auc(fpr, tpr)
            fpr = [float(round(value, 4)) for value in fpr]
            tpr = [float(round(value, 4)) for value in tpr]

        result = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_score,
            "class_mapping": class_mapping,
            "class_order": class_order,
            "thresholds": thresholds,
        }

        return result

    def calculate(self, data: dict) -> dict:
        """
        Calculates the AUC metric for the given dataset.

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

        # enforce metadata turn off until more solid understanding of cohort creation
        if DEBUG:
            metadata = None

        # Apply metadata if given
        if metadata:
            # when the metadata is applied it would stratify the data based on combination
            # to form cohorts
            # the validation is calculated on the cohorts
            cohort_data = self.apply_metadata(data)

        # preprocess the data
        if metadata:
            for _cohort_key in cohort_data:
                result[_cohort_key] = self.__calculate_metrics(
                    cohort_data[_cohort_key], data.get("class_mapping")
                )
        else:
            result = self.__calculate_metrics(data, data.get("class_mapping"))

        return result


class SemanticSegmentation(Classification):
    pass


class ObjectDetection:
    pass


class PlotAuc:
    def __init__(self, data: dict):
        self.data = data
        self.fpr = data["fpr"]
        self.tpr = data["tpr"]
        self.aucs = data["auc"]
        self.class_mappings = data["class_mapping"]
        self.class_order = data["class_order"]

    def _validate_data(self):
        """
        Validates the data required for plotting the AUC.
        """
        required_keys = ["fpr", "tpr", "auc"]
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Data must contain '{key}'.")

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

    def plot(self) -> Figure:
        """
        Plots the AUC curves.
        """
        sns.set_style("whitegrid")
        # Validate the data
        self._validate_data()

        fig, ax = plt.subplots(figsize=(10, 7))
        # TODO: Class wise plot auc-roc
        sns.lineplot(x=self.fpr, y=self.tpr, ax=ax)
        # ax.plot(self.fpr, self.tpr, "k--")
        ax.set_xlabel(
            "False Positive Rate", fontdict={"fontsize": 14, "fontweight": "medium"}
        )
        ax.set_ylabel(
            "True Positive Rate", fontdict={"fontsize": 14, "fontweight": "medium"}
        )
        ax.set_title(
            "Receiver Operating Characteristic (ROC) Curves",
            fontdict={"fontsize": 16, "fontweight": "medium"},
        )
        ax.legend(loc="lower right")
        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("classification.auc")
def calculate_auc_metric(data: dict, problem_type: str):
    """
    A wrapper function to calculate the AUC metric.

    :param data: Dictionary of data: {"prediction": , "ground_truth": }
    :type data: dict
    :param problem_type: Type of the problem
    :type problem_type: str

    :return: Dict of calculated results
    :rtype: dict
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result


@plot_manager.register("classification.auc")
def plot_auc(
    results: dict, save_plot: bool, file_name: str = "auc_plot.png"
) -> Union[str, None]:
    """
    A wrapper function to plot the AUC curves.

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: Boolean value to save plot
    :type save_plot: bool
    :param file_name: name of the file
    :type file_name: str

    :return: None or path to the saved plot
    :rtype: Union[str, None]
    """
    plotter = PlotAuc(data=results)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
