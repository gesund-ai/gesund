import os
from typing import Union, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager


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

    def _preprocess(self, data: dict, get_logits=False):
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

        prediction, ground_truth = self._preprocess(data, get_logits=True)

        lift_chart_calculator = LiftChart(class_mapping)
        lift_points = lift_chart_calculator.calculate_lift_curve_points(
            ground_truth, prediction
        )

        result = {
            "lift_points": lift_points,
            "class_mapping": class_mapping,
            "class_order": class_order,
        }

        return result

    def calculate(self, data: dict) -> dict:
        """
        Calculates the lift chart for the given data.

        :param data: The input data required for calculation and plotting
                     {"prediction":, "ground_truth": , "metadata":, "class_mapping":}
        :type data: dict

        :return: Calculated metric results
        :rtype: dict
        """
        result = {}

        # Validate the data
        self._validate_data(data)

        # calculate the metrics
        result = self.__calculate_metrics(data, data.get("class_mapping"))
        return result


class LiftChart:
    def __init__(self, class_mapping: Dict[int, str]) -> None:
        self.class_mapping = class_mapping
        self.class_order = [int(i) for i in list(class_mapping.keys())]

    def _decile_table(
        self,
        true: Union[List[int], np.ndarray, pd.Series],
        pred_logits: pd.DataFrame,
        predicted_class: int = 0,
        change_deciles: int = 20,
        labels: bool = True,
        round_decimal: int = 3,
    ) -> pd.DataFrame:
        """
        Generates the Decile Table from labels and probabilities.

        :return: Dataframe containing decile metrics
        """
        df = pd.DataFrame(
            {"true": true, "pred_logits": pred_logits.loc[:, predicted_class]}
        )

        # Sort by predicted logits in descending order
        df.sort_values("pred_logits", ascending=False, inplace=True)

        # Assign deciles
        df["decile"] = pd.qcut(
            df["pred_logits"].rank(method="first"), change_deciles, labels=False
        )
        df["decile"] = df["decile"] + 1  # Deciles start from 1

        # Calculate lift metrics
        lift_df = (
            df.groupby("decile")
            .agg(
                prob_min=("pred_logits", "min"),
                prob_max=("pred_logits", "max"),
                prob_avg=("pred_logits", "mean"),
                cnt_cust=("pred_logits", "count"),
                cnt_resp=("true", "sum"),
            )
            .reset_index()
        )

        lift_df["cnt_non_resp"] = lift_df["cnt_cust"] - lift_df["cnt_resp"]

        total_resp = df["true"].sum()
        total_cust = df["true"].count()

        lift_df["resp_rate"] = round(
            lift_df["cnt_resp"] * 100 / lift_df["cnt_cust"], round_decimal
        )
        lift_df["cum_cust"] = np.cumsum(lift_df["cnt_cust"])
        lift_df["cum_resp"] = np.cumsum(lift_df["cnt_resp"])
        lift_df["cum_non_resp"] = np.cumsum(lift_df["cnt_non_resp"])
        lift_df["cum_cust_pct"] = round(
            lift_df["cum_cust"] * 100 / total_cust, round_decimal
        )
        lift_df["cum_resp_pct"] = round(
            lift_df["cum_resp"] * 100 / total_resp, round_decimal
        )
        lift_df["cum_non_resp_pct"] = round(
            lift_df["cum_non_resp"] * 100 / (total_cust - total_resp), round_decimal
        )
        lift_df["KS"] = round(
            lift_df["cum_resp_pct"] - lift_df["cum_non_resp_pct"], round_decimal
        )
        lift_df["lift"] = round(
            lift_df["cum_resp_pct"] / lift_df["cum_cust_pct"], round_decimal
        )
        return lift_df

    def calculate_lift_curve_points(
        self,
        true: Union[List[int], np.ndarray, pd.Series],
        pred_logits: Union[List[int], np.ndarray, pd.Series],
        predicted_class: Optional[int] = None,
        change_deciles: int = 20,
        labels: bool = True,
        round_decimal: int = 3,
    ) -> Dict[str, List[Dict[str, float]]]:
        pred_df = pd.DataFrame(pred_logits)
        class_lift_dict = dict()
        if predicted_class in [None, "all", "overall"]:
            for class_ in self.class_order:
                lift_df = self._decile_table(
                    true,
                    pred_df,
                    predicted_class=int(class_),
                    change_deciles=change_deciles,
                    labels=labels,
                    round_decimal=round_decimal,
                )
                xy_points = [
                    {"x": avg_prob, "y": lift}
                    for avg_prob, lift in zip(lift_df["prob_avg"], lift_df["lift"])
                ]
                class_name = self.class_mapping[str(class_)]
                class_lift_dict[class_name] = xy_points
        else:
            lift_df = self._decile_table(
                true,
                pred_df,
                predicted_class=predicted_class,
                change_deciles=change_deciles,
                labels=labels,
                round_decimal=round_decimal,
            )
            xy_points = [
                {"x": avg_prob, "y": lift}
                for avg_prob, lift in zip(lift_df["prob_avg"], lift_df["lift"])
            ]
            class_name = self.class_mapping[str(predicted_class)]
            class_lift_dict[class_name] = xy_points

        return class_lift_dict


class SemanticSegmentation(Classification):
    pass


class ObjectDetection:
    pass


class PlotLiftChart:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        self.lift_points = data["lift_points"]
        self.class_mapping = data["class_mapping"]
        self.class_order = data["class_order"]
        self.cohort_id = cohort_id

    def _validate_data(self):
        """
        Validates the data required for plotting the Lift Chart.
        """
        if "lift_points" not in self.data:
            raise ValueError("Data must contain 'lift_points'.")

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

        if self.cohort_id:
            filepath = f"{dir_path}/{self.cohort_id}_{filename}"
        else:
            filepath = f"{dir_path}/{filename}"

        fig.savefig(filepath, format="png")
        return filepath

    def plot(self) -> Figure:
        """
        Plots the Lift Chart.

        :return: Matplotlib Figure object
        :rtype: Figure
        """
        sns.set_style("whitegrid")
        # Validate the data
        self._validate_data()

        fig, ax = plt.subplots(figsize=(10, 7))
        for class_name, points in self.lift_points.items():
            x = [p["x"] for p in points]
            y = [p["y"] for p in points]
            ax.plot(x, y, marker="o", label=f"Class {class_name}")

        ax.set_xlabel(
            "Average Predicted Probability",
            fontdict={"fontsize": 14, "fontweight": "medium"},
        )
        ax.set_ylabel(
            "Calculated Lift", fontdict={"fontsize": 14, "fontweight": "medium"}
        )

        if self.cohort_id:
            title_str = f"Lift Chart : cohort - {self.cohort_id}"
        else:
            title_str = "Lift Chart"

        ax.set_title(title_str, fontdict={"fontsize": 16, "fontweight": "medium"})
        ax.legend()
        return fig


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("classification.lift_chart")
def calculate_lift_chart_metric(data: dict, problem_type: str):
    """
    A wrapper function to calculate the Lift Chart metric.

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


@plot_manager.register("classification.lift_chart")
def plot_lift_chart(
    results: dict,
    save_plot: bool,
    file_name: str = "lift_chart.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to plot the lift chart.

    :param results: Dictionary of the results
    :type results: dict
    :param save_plot: Boolean value to save plot
    :type save_plot: bool
    :param file_name: Name of the file to save the plot
    :type file_name: str
    :param cohort_id: id of the cohort
    :type cohort_id: int

    :return: None or path to the saved plot
    :rtype: Union[str, None]
    """
    plotter = PlotLiftChart(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
