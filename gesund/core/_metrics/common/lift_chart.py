from typing import Union, Callable, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gesund.core import metric_manager, plot_manager


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
        Calculates the Lift Chart metric for the given dataset.
        """
        # Validate the data
        self._validate_data(data)

        # Extract predictions and ground truth
        true = np.array(data["ground_truth"])
        pred_logits = pd.DataFrame(data["prediction"]).T
        metadata = data.get("metadata", None)

        # Apply metadata if given
        if metadata is not None:
            data = self.apply_metadata(data, metadata)
            true = np.array(data["ground_truth"])
            pred_logits = pd.DataFrame(data["prediction"]).T

        # Get class mappings if provided, else infer from data
        class_mappings = data.get("class_mappings", None)
        if class_mappings is not None:
            class_order = [int(i) for i in list(class_mappings.keys())]
        else:
            # Infer class order from data
            classes = np.unique(true)
            class_order = classes.tolist()
            class_mappings = {int(i): str(i) for i in class_order}

        # Run the calculation logic
        lift_chart_calculator = LiftChart(class_mappings)
        lift_points = lift_chart_calculator.calculate_lift_curve_points(
            true, pred_logits
        )

        result = {
            "lift_points": lift_points,
            "class_mappings": class_mappings,
            "class_order": class_order,
        }

        return result


class LiftChart:
    def __init__(self, class_mappings: Dict[int, str]) -> None:
        self.class_mappings = class_mappings
        self.class_order = [int(i) for i in list(class_mappings.keys())]

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
        """
        df = pd.DataFrame(
            {"true": true, "pred_logits": pred_logits.loc[predicted_class]}
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
        pred_logits: pd.DataFrame,
        predicted_class: Optional[int] = None,
        change_deciles: int = 20,
        labels: bool = True,
        round_decimal: int = 3,
    ) -> Dict[str, List[Dict[str, float]]]:
        class_lift_dict = dict()
        if predicted_class in [None, "all", "overall"]:
            for class_ in self.class_order:
                lift_df = self._decile_table(
                    true,
                    pred_logits,
                    predicted_class=int(class_),
                    change_deciles=change_deciles,
                    labels=labels,
                    round_decimal=round_decimal,
                )
                xy_points = [
                    {"x": avg_prob, "y": lift}
                    for avg_prob, lift in zip(lift_df["prob_avg"], lift_df["lift"])
                ]
                class_name = self.class_mappings[class_]
                class_lift_dict[class_name] = xy_points
        else:
            lift_df = self._decile_table(
                true,
                pred_logits,
                predicted_class=predicted_class,
                change_deciles=change_deciles,
                labels=labels,
                round_decimal=round_decimal,
            )
            xy_points = [
                {"x": avg_prob, "y": lift}
                for avg_prob, lift in zip(lift_df["prob_avg"], lift_df["lift"])
            ]
            class_name = self.class_mappings[predicted_class]
            class_lift_dict[class_name] = xy_points

        return class_lift_dict


class PlotLiftChart:
    def __init__(self, data: dict):
        self.data = data
        self.lift_points = data["lift_points"]
        self.class_mappings = data["class_mappings"]
        self.class_order = data["class_order"]

    def _validate_data(self):
        """
        Validates the data required for plotting the Lift Chart.
        """
        if "lift_points" not in self.data:
            raise ValueError("Data must contain 'lift_points'.")

    def save(self, filepath: str = "lift_chart.png") -> str:
        """
        Saves the plot to a file.
        """
        plt.savefig(filepath)
        return filepath

    def plot(self):
        """
        Plots the Lift Chart.
        """
        # Validate the data
        self._validate_data()

        plt.figure(figsize=(10, 7))
        for class_name, points in self.lift_points.items():
            x = [p["x"] for p in points]
            y = [p["y"] for p in points]
            plt.plot(x, y, marker="o", label=f"Class {class_name}")

        plt.xlabel("Average Predicted Probability")
        plt.ylabel("Lift")
        plt.title("Lift Chart")
        plt.legend()
        plt.grid(True)
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


@metric_manager.register("classification.lift_chart")
def calculate_lift_chart_metric(data: dict, problem_type: str):
    """
    A wrapper function to calculate the Lift Chart metric.
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result


@plot_manager.register("classification.lift_chart")
def plot_lift_chart(results: dict, save_plot: bool) -> Union[str, None]:
    """
    A wrapper function to plot the Lift Chart.
    """
    plotter = PlotLiftChart(data=results)
    plotter.plot()
    if save_plot:
        return plotter.save()
