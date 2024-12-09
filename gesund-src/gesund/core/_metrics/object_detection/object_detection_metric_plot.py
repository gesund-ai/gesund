import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class Object_Detection_Plot:
    """
    A class for creating various plots and visualizations for object detection results.

    :param blind_spot_path: (str, optional) Path to the blind spot analysis JSON file.
    :param top_misses_path: (str, optional) Path to the top misses analysis JSON file.
    :param table_json_path: (str, optional) Path to the table metrics JSON file.
    :param mixed_json_path: (str, optional) Path to the mixed metrics JSON file.
    :param overall_json_path: (str, optional) Path to the overall metrics JSON file.
    :param confidence_histogram_path: (str, optional) Path to the confidence histogram JSON file.
    :param output_dir: (str, optional) Directory path for saving output plots.
    """

    def __init__(
        self,
        blind_spot_path=None,
        top_misses_path=None,
        table_json_path=None,
        mixed_json_path=None,
        overall_json_path=None,
        confidence_histogram_path=None,
        output_dir=None,
    ):
        self.output_dir = output_dir
        if blind_spot_path:
            self.result_dict = self._load_json(blind_spot_path)
        elif top_misses_path:
            self.new_json_data = self._load_json(top_misses_path)
        elif table_json_path:
            self.table_json_data = self._load_json(table_json_path)
        elif mixed_json_path:
            self.mixed_json_data = self._load_json(mixed_json_path)
        elif overall_json_path:
            self.overall_json_data = self._load_json(overall_json_path)
        elif confidence_histogram_path:
            self.confidence_histogram_data = self._load_json(confidence_histogram_path)

    def _load_json(self, json_path):
        """
        Load data from a JSON file.

        :param json_path: (str) Path to the JSON file to be loaded.

        :return: (dict) Loaded JSON data as a Python dictionary.
        """
        with open(json_path, "r") as file:
            data = json.load(file)
        return data

    def draw(
        self,
        plot_type,
        blind_spot_args=None,
        top_misses_args=None,
        classbased_table_args=None,
        mixed_args=None,
        overall_args=None,
        confidence_histogram_args=None,
        save_path=None,
    ):
        """
        Draw various types of plots based on the specified plot type and parameters.

        :param plot_type: (str) Type of plot to generate ('blind_spot', 'top_misses', 'classbased_table',
                         'mixed_plot', 'overall_metrics', 'confidence_histogram').
        :param blind_spot_args: (dict, optional) Arguments for blind spot analysis plot.
        :param top_misses_args: (dict, optional) Arguments for top misses plot.
        :param classbased_table_args: (dict, optional) Arguments for class-based table plot.
        :param mixed_args: (dict, optional) Arguments for mixed metrics plot.
        :param overall_args: (dict, optional) Arguments for overall metrics plot.
        :param confidence_histogram_args: (dict, optional) Arguments for confidence histogram plot.
        :param save_path: (str, optional) Path to save the generated plot.

        :raises ValueError: If an unsupported plot type is specified.
        """
        if plot_type == "blind_spot":
            self._plot_blind_spot(blind_spot_args, save_path)
        elif plot_type == "top_misses":
            self._plot_top_misses(top_misses_args, save_path)
        elif plot_type == "classbased_table":
            self._plot_classbased_table_metrics(classbased_table_args, save_path)
        elif plot_type == "mixed_plot":
            self._plot_mixed_metrics(mixed_args, save_path)
        elif plot_type == "overall_metrics":
            self._plot_overall_metrics(overall_args, save_path)
        elif plot_type == "confidence_histogram":
            self._plot_confidence_histogram(confidence_histogram_args, save_path)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

    def _plot_blind_spot(self, blind_spot_args, save_path=None):
        """
        Create a bar plot visualization for blind spot analysis metrics.

        :param blind_spot_args: (dict, optional) Dictionary containing:
            - 'Average': (list) List of specific metrics to include in the plot
            - 'threshold': (float) Minimum threshold value for filtering metrics
        :param save_path: (str, optional) Path where the plot should be saved.
                         If None, saves as 'average_metrics.png'.

        :return: None
        :raises AttributeError: If no valid blind spot data is loaded.
        """

        if not self.result_dict or "Average" not in self.result_dict:
            print("No valid 'Average' data found in the JSON.")
            return

        data = self.result_dict["Average"]

        df = pd.DataFrame(
            [(k, v) for k, v in data.items() if v != "None"],
            columns=["Metric", "Value"],
        )
        df["Value"] = df["Value"].astype(float)

        if blind_spot_args:
            if "Average" in blind_spot_args:
                df = df[df["Metric"].isin(blind_spot_args["Average"])]
            if "threshold" in blind_spot_args:
                df = df[df["Value"] > blind_spot_args["threshold"]]

        df = df.sort_values("Value", ascending=False)

        plt.figure(figsize=(14, 8))
        sns.barplot(
            x="Value",
            y="Metric",
            hue="Metric",
            data=df,
            palette="pastel",
            edgecolor="black",
            legend=False,
        )
        plt.title("Average Metrics", fontsize=20, fontweight="bold", pad=20)
        plt.xlabel("Value", fontsize=14, labelpad=15)
        plt.ylabel("Metric", fontsize=14, labelpad=15)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.grid(True, axis="x", linestyle="--", alpha=0.7)

        for i, value in enumerate(df["Value"]):
            plt.text(
                value + 0.01,
                i,
                f"{value:.4f}",
                va="center",
                ha="left",
                fontsize=12,
                color="black",
                fontweight="bold",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig("average_metrics.png", bbox_inches="tight", dpi=300)

        plt.show()
        plt.close()

    def _plot_overall_metrics(self, overall_args, save_path=None):
        """
        Create a bar plot visualization for overall validation metrics.

        :param overall_args: (dict, optional) Dictionary containing:
            - 'metrics': (list) List of specific metrics to include in the plot
            - 'threshold': (float) Minimum threshold value for filtering metrics
        :param save_path: (str, optional) Path where the plot should be saved.
                        If None, saves as 'overall_metrics.png'.

        :return: None
        :raises AttributeError: If no valid overall data is loaded.
        """

        if (
            not self.overall_json_data
            or self.overall_json_data.get("type") != "overall"
        ):
            print("No valid 'overall' data found in the JSON.")
            return

        data = self.overall_json_data.get("data", {})
        df = pd.DataFrame(
            [(k, v["Validation"]) for k, v in data.items()], columns=["Metric", "Value"]
        )

        if overall_args:
            if "metrics" in overall_args:
                df = df[df["Metric"].isin(overall_args["metrics"])]
            if "threshold" in overall_args:
                df = df[df["Value"] > overall_args["threshold"]]

        df = df.sort_values("Value", ascending=False)

        plt.figure(figsize=(14, 8))
        sns.barplot(
            x="Value",
            y="Metric",
            hue="Metric",
            data=df,
            palette="pastel",
            edgecolor="black",
            legend=False,
        )

        plt.title("Overall Metrics", fontsize=20, fontweight="bold", pad=20)
        plt.xlabel("Metric Value", fontsize=14, labelpad=15)
        plt.ylabel("Metric", fontsize=14, labelpad=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.grid(True, axis="x", linestyle="--", alpha=0.7)

        for i, value in enumerate(df["Value"]):
            plt.text(
                value + 0.01,
                i,
                f"{value:.4f}",
                va="center",
                ha="left",
                fontsize=12,
                color="black",
                fontweight="bold",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig("overall_metrics.png", bbox_inches="tight", dpi=300)

        plt.show()
        plt.close()

    def _plot_top_misses(self, top_misses_args, save_path=None):
        """
        Create a bar plot visualization for top missed detections based on mIoU scores.

        :param top_misses_args: (dict, optional) Dictionary containing:
            - 'min_miou': (float) Minimum mIoU threshold for filtering
            - 'max_miou': (float) Maximum mIoU threshold for filtering
            - 'top_n': (int) Number of top misses to display
        :param save_path: (str, optional) Path where the plot should be saved.
                        If None, saves as 'top_misses.png'.

        :return: None
        :raises AttributeError: If no valid image data is loaded.
        """

        if not self.new_json_data or self.new_json_data.get("type") != "image":
            print("No valid 'image' data found in the JSON.")
            return

        data = self.new_json_data.get("data", [])
        df = pd.DataFrame(data)

        if top_misses_args:
            if "min_miou" in top_misses_args:
                df = df[df["mIoU"] >= top_misses_args["min_miou"]]
            if "max_miou" in top_misses_args:
                df = df[df["mIoU"] <= top_misses_args["max_miou"]]
            if "top_n" in top_misses_args:
                df = df.nsmallest(top_misses_args["top_n"], "rank")

        df = df.sort_values("rank")

        plt.figure(figsize=(14, 8))
        sns.barplot(
            x="mIoU",
            y="image_id",
            hue="image_id",
            data=df,
            palette="pastel",
            edgecolor="black",
            legend=False,
        )
        plt.title(
            "Top Misses: mIoU for Each Image", fontsize=20, fontweight="bold", pad=20
        )
        plt.xlabel("mIoU", fontsize=14, labelpad=15)
        plt.ylabel("Image ID", fontsize=14, labelpad=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.grid(True, axis="x", linestyle="--", alpha=0.7)

        for index, value in enumerate(df["mIoU"]):
            plt.text(
                value + 0.01,
                index,
                f"{value:.4f}",
                va="center",
                ha="left",
                fontsize=12,
                color="black",
                fontweight="bold",
            )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig("top_misses.png", bbox_inches="tight", dpi=300)

        plt.show()
        plt.close()

    def _plot_classbased_table_metrics(self, classbased_table_args, save_path=None):
        """
        Create a grouped bar plot visualization for class-based metrics comparison.

        :param classbased_table_args: (dict, optional) Dictionary containing:
            - 'metrics': (list) List of specific metrics to include in the plot
            - 'threshold': (float) Minimum threshold value for filtering metrics
        :param save_path: (str, optional) Path where the plot should be saved.
                        If None, saves as 'class_table_metrics.png'.

        :return: None
        :raises AttributeError: If no valid table data is loaded.
        """
        if not self.table_json_data or self.table_json_data.get("type") != "table":
            print("No valid 'table' data found in the JSON.")
            return

        data = self.table_json_data.get("data", {}).get("Validation", {})
        df = pd.DataFrame.from_dict(data, orient="index")

        if classbased_table_args:
            if "metrics" in classbased_table_args:
                available_metrics = set(df.columns) & set(
                    classbased_table_args["metrics"]
                )
                df = df[list(available_metrics)]
            if "threshold" in classbased_table_args:
                df = df[
                    df.apply(
                        lambda row: all(
                            float(val) > classbased_table_args["threshold"]
                            for val in row
                            if val != "None"
                        ),
                        axis=1,
                    )
                ]

        df = df.reset_index().rename(columns={"index": "Class"})
        df = df.melt(id_vars=["Class"], var_name="Metric", value_name="Value")

        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            x="Value", y="Class", hue="Metric", data=df, palette="Set2", dodge=True
        )
        plt.title("Class-Based Table Metrics", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Metric Value", fontsize=12, labelpad=10)
        plt.ylabel("Class", fontsize=12, labelpad=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, axis="x", linestyle="--", alpha=0.7)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="center", fontsize=8)

        plt.legend(
            title="Metric",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=10,
            title_fontsize=12,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig("class_table_metrics.png", bbox_inches="tight", dpi=300)

        plt.show()
        plt.close()

    def _plot_mixed_metrics(self, mixed_args, save_path=None):
        """
        Create a bar plot visualization for mixed metrics data.

        :param mixed_args: (dict, optional) Dictionary containing:
            - 'metrics': (list) List of specific metrics to include in the plot
            - 'threshold': (float) Minimum threshold value for filtering metrics
        :param save_path: (str, optional) Path where the plot should be saved.
                          If None, saves as 'mixed_metrics_new.png'.

        :return: None
        :raises AttributeError: If no valid mixed data is loaded.
        """
        if not self.mixed_json_data or self.mixed_json_data.get("type") != "mixed":
            print("No valid 'mixed' data found in the JSON.")
            return

        data = self.mixed_json_data.get("data", {}).get("ap_results", {})
        df = (
            pd.DataFrame.from_dict(data, orient="index")
            .reset_index()
            .rename(columns={"index": "Metric", 0: "Value"})
        )

        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        if mixed_args:
            if "metrics" in mixed_args:
                df = df[df["Metric"].isin(mixed_args["metrics"])]
            if "threshold" in mixed_args:
                df = df[df["Value"] > mixed_args["threshold"]]

        df = df.dropna(subset=["Value"])

        plt.figure(figsize=(12, 8))
        sns.barplot(
            x="Value",
            y="Metric",
            hue="Metric",
            data=df,
            palette="pastel",
            dodge=False,
            legend=False,
        )
        plt.title("Mixed Metrics", fontsize=18, fontweight="bold", pad=20)
        plt.xlabel("Metric Value", fontsize=14, labelpad=15)
        plt.ylabel("Metric Name", fontsize=14, labelpad=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, axis="x", linestyle="--", alpha=0.7)

        for index, value in enumerate(df["Value"]):
            plt.text(
                value + 0.02,
                index,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=12,
                color="black",
                fontweight="bold",
            )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig("mixed_metrics_new.png", bbox_inches="tight", dpi=300)

        plt.show()
        plt.close()

    def _plot_confidence_histogram(self, confidence_histogram_args, save_path=None):
        """
        Create scatter and histogram plots for confidence histogram data.

        :param confidence_histogram_args: (dict, optional) Dictionary containing:
            - 'labels': (list) List of labels to include in the scatter plot
        :param save_path: (str, optional) Path where the plots should be saved.
                          If None, saves as 'scatter_plot_points.png' and 'histogram.png'.

        :return: None
        :raises AttributeError: If no valid confidence histogram data is loaded.
        """
        if (
            not self.confidence_histogram_data
            or self.confidence_histogram_data.get("type") != "mixed"
        ):
            print("No valid 'confidence_histogram' data found in the JSON.")
            return
        points_data = self.confidence_histogram_data.get("data", {}).get("points", [])
        histogram_data = self.confidence_histogram_data.get("data", {}).get(
            "histogram", []
        )
        points_df = pd.DataFrame(points_data)
        histogram_df = pd.DataFrame(histogram_data)

        if confidence_histogram_args and "labels" in confidence_histogram_args:
            points_df = points_df[
                points_df["labels"].isin(confidence_histogram_args["labels"])
            ]

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x="x",
            y="y",
            hue="labels",
            data=points_df,
            palette="rocket",
            s=100,
            alpha=0.7,
        )
        plt.title("Scatter Plot of Points", fontsize=18, fontweight="bold", pad=20)
        plt.xlabel("X ", fontsize=14, labelpad=15)
        plt.ylabel("Y ", fontsize=14, labelpad=15)

        plt.grid(True, linestyle="--", alpha=0.7)

        plt.legend(
            title="Labels",
            fontsize=12,
            title_fontsize=14,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

        scatter_save_path = (
            save_path.replace(".png", "_scatter.png")
            if save_path
            else "scatter_plot_points.png"
        )
        plt.savefig(scatter_save_path, bbox_inches="tight", dpi=300)

        plt.show()

        plt.close()

        ### Histogram ###
        plt.figure(figsize=(12, 8))

        sns.barplot(
            x="category",
            y="value",
            hue="category",
            data=histogram_df,
            palette="pastel",
            legend=False,
        )

        plt.title("Confidence Histogram", fontsize=18, fontweight="bold", pad=20)
        plt.xlabel("Confidence Category", fontsize=14, labelpad=15)
        plt.ylabel("Frequency", fontsize=14, labelpad=15)

        plt.grid(True, axis="y", linestyle="--", alpha=0.7)

        for index, row in histogram_df.iterrows():
            plt.text(
                index,
                row["value"] + 0.2,
                f'{row["value"]}',
                ha="center",
                fontsize=12,
                color="black",
                fontweight="bold",
            )

        histogram_save_path = (
            save_path.replace(".png", "_histogram.png")
            if save_path
            else "histogram.png"
        )
        plt.savefig(histogram_save_path, bbox_inches="tight", dpi=300)

        plt.show()
        plt.close()
