from typing import Union, Optional
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager

class Classification:
    pass

class SemanticSegmentation:
    """
    Performs semantic segmentation metric calculations.
    """
    def __init__(self):
        pass

    def _validate_data(self, data: dict) -> bool:
        """
        Validates the data required for semantic segmentation metric calculation.

        :param data: The input data required for calculation, {"ground_truth":, "prediction":}
        :type data: dict

        :return: Status if the data is valid
        :rtype: bool

        :raises ValueError: If data is not a dictionary or missing required keys.
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        if "ground_truth" not in data:
            raise ValueError("Data must contain 'ground_truth' key.")
        return True

    def _preprocess(self, data: dict, get_class_only=False) -> pd.DataFrame:
        """
        Preprocesses the input data to extract ground truth class labels.

        :param data: Dictionary containing 'ground_truth' data.
        :type data: dict
        :param get_class_only: Flag to include only class labels, defaults to False.
        :type get_class_only: bool

        :return: DataFrame with ground truth class labels.
        :rtype: pd.DataFrame
        """
        results = {"gt_class_label": []}  
        for image_id in data["ground_truth"]:
            for annotation in data["ground_truth"][image_id]["annotation"]:
                if annotation["type"] == "mask":  
                    results["gt_class_label"].append(annotation["label"])

        results = pd.DataFrame(results)
        return results

    def _calculate_metrics(self, data: dict) -> dict:
        """
        Calculates dataset population distribution metrics.

        :param data: Dictionary containing preprocessed data.
        :type data: dict

        :return: Dictionary with dataset population distribution.
        :rtype: dict
        """
        results = {}
        dataset_pop = self._preprocess(data)
        results["dataset_population_distribution"] = dataset_pop
        return results

    def calculate(self, data: dict) -> dict:
        """
        Validates data and calculates semantic segmentation metrics.

        :param data: The input data required for calculation, {"ground_truth":, "prediction":}
        :type data: dict

        :return: Calculated metric results.
        :rtype: dict
        """
        self._validate_data(data)
        return self._calculate_metrics(data)

class ObjectDetection(SemanticSegmentation):
    """
    Performs object detection metric calculations.
    """
    def __init__(self):
        """
        Initializes the ObjectDetection metric calculator.
        """
        super().__init__()

    def _preprocess(self, data: dict, get_class_only=False) -> pd.DataFrame:
        """
        Preprocesses the input data to extract ground truth bounding box class labels.

        :param data: Dictionary containing 'ground_truth' data.
        :type data: dict
        :param get_class_only: Flag to include only class labels, defaults to False.
        :type get_class_only: bool

        :return: DataFrame with ground truth bounding box class labels.
        :rtype: pd.DataFrame
        """
        results = {"gt_class_label": []}  
        for image_id in data["ground_truth"]:
            for annotation in data["ground_truth"][image_id]["annotation"]:
                if annotation["type"] == "bbox": 
                    results["gt_class_label"].append(annotation["label"])

        results = pd.DataFrame(results)
        return results

class PlotDatasetPopulationDistributionGender:
    """
    Class to generate and save plots for dataset population distribution based on gender.
    """
    def __init__(self, data=None, cohort_id=None):
        """
        Initializes the PlotDatasetPopulationDistributionGender with data and optional cohort identifier.

        :param data: List of dictionaries containing 'Gender' information, defaults to None.
        :type data: Optional[List[Dict]], optional
        :param cohort_id: Optional identifier for the cohort, defaults to None.
        :type cohort_id: Optional[int], optional
        """
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self, json_file):
        """
        Validates and loads data from a JSON file.

        :param json_file: Path to the JSON file containing gender data.
        :type json_file: str

        :raises ValueError: If the JSON file cannot be read or has an invalid format.
        """
        try:
            with open(json_file, 'r') as f:
                self.data = json.load(f)
            if not isinstance(self.data, list) or not all('Gender' in item for item in self.data):
                raise ValueError("Invalid data format: Must be a list of dictionaries with 'Gender' field")
        except Exception as e:
            raise ValueError(f"Error reading JSON file: {str(e)}")

    def calculate(self):
        """
        Calculates gender distribution metrics such as mean count and unique genders.

        :return: Dictionary containing gender counts, mean count, and unique genders.
        :rtype: dict

        :raises ValueError: If no data is loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call _validate_data first.")

        genders = [item['Gender'] for item in self.data]
        return {
            'genders': genders,
            'mean_gender_count': np.mean([genders.count(g) for g in set(genders)]),
            'unique_genders': list(set(genders))
        }

    def save(self, fig: Figure, filename: str) -> str:
        """
        Saves the matplotlib figure to a local directory.

        :param fig: A Matplotlib Figure object.
        :type fig: Figure
        :param filename: Name of the file to save the figure.
        :type filename: str

        :return: File path where the plot is saved.
        :rtype: str
        """
        dir_path = "plots"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        filepath = f"{dir_path}/{self.cohort_id}_{filename}" if self.cohort_id else f"{dir_path}/{filename}"
        fig.savefig(filepath, format="png")
        return filepath


    def _setup_plot(self, figsize=(12, 7)):
        """
        Sets up the matplotlib plot with predefined styles.

        :param figsize: Size of the figure, defaults to (12, 7).
        :type figsize: tuple, optional

        :return: Tuple containing the figure and axes objects.
        :rtype: Tuple[Figure, plt.Axes]
        """
        sns.set_theme(style="whitegrid", font_scale=1.2)
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')
        return fig, ax

    def _annotate_bars(self, ax):
        """
        Annotates bars in the bar plot with their respective counts.

        :param ax: Matplotlib Axes object where the bars are plotted.
        :type ax: plt.Axes
        """
        for p in ax.patches:
            count = int(p.get_height())
            x = p.get_x() + p.get_width()/2
            y = p.get_height()
            ax.annotate(f'{count}', (x, y),
                        ha='center', va='bottom',
                        fontsize=10, color='black')

    def _customize_plot(self, ax, title, xlabel, ylabel):
        """
        Customizes the plot with titles and labels.

        :param ax: Matplotlib Axes object to customize.
        :type ax: plt.Axes
        :param title: Title of the plot.
        :type title: str
        :param xlabel: Label for the x-axis.
        :type xlabel: str
        :param ylabel: Label for the y-axis.
        :type ylabel: str
        """
        ax.set_title(title, fontsize=16, pad=20, color='black')
        ax.set_xlabel(xlabel, fontsize=12, color='black')
        ax.set_ylabel(ylabel, fontsize=12, color='black')
        ax.tick_params(colors='black')
        for spine in ax.spines.values():
            spine.set_color('black')



    def plot_gender(self, json_file, suffix=''):
        """
        Generates a bar plot for gender distribution based on the provided JSON data.

        :param json_file: Path to the JSON file containing gender data.
        :type json_file: str
        :param suffix: Optional suffix to add to the plot title, defaults to ''.
        :type suffix: str, optional

        :return: Matplotlib Figure object with the gender distribution plot.
        :rtype: Figure

        :raises ValueError: If data validation fails.
        """
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        self._validate_data(json_file)
        df = pd.DataFrame(self.data)

        fig, ax = self._setup_plot()

        sns.countplot(
            data=df,
            x='Gender',
            hue="Gender",
            ax=ax,
            palette='coolwarm',
            edgecolor='white',
            alpha=0.8,
            legend=False
        )

        self._annotate_bars(ax)
        self._customize_plot(ax, 'Gender Distribution', 'Gender', 'Count')
        fig.tight_layout()

        return fig

problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}

def _get_json_file_path(problem_type: str) -> str:
    """
    Constructs the JSON file path based on the problem type.

    :param problem_type: Type of the problem (e.g., classification, semantic_segmentation, object_detection).
    :type problem_type: str

    :return: Path to the JSON file containing metadata.
    :rtype: str
    """
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__)
                )
            )
        )
    )
    return os.path.join(
        project_root,
        'tests',
        '_data',
        problem_type,
        'test_metadata_new.json'
    )

@metric_manager.register("object_detection.dataset_population_distribution_gender")
@metric_manager.register("semantic_segmentation.dataset_population_distribution_gender")
def calculate_dataset_population_distribution_metric(data: dict, problem_type: str):
    """
    Calculates dataset population distribution metrics based on the problem type.

    :param data: Dictionary containing 'ground_truth' and 'prediction' data.
    :type data: dict
    :param problem_type: Type of the problem (e.g., classification, semantic_segmentation, object_detection).
    :type problem_type: str

    :return: Dictionary with dataset population distribution gender metrics.
    :rtype: dict
    """
    metric_calculator = problem_type_map[problem_type]()
    result = metric_calculator.calculate(data)
    return {
        "dataset_population_distribution_gender": result
    }

@plot_manager.register("semantic_segmentation.dataset_population_distribution_gender")
def plot_dataset_population_distribution_gender(
    results: dict,
    save_plot: bool,
    file_name: str = "dataset_population_distribution_gender.png",
    cohort_id: Optional[int] = None,
    suffix: str = '',
    problem_type: str = 'semantic_segmentation'
) -> Union[dict, None]:
    """
    Plots dataset population distribution for semantic segmentation, optionally saving the plot to disk.

    :param results: Dictionary containing metric results.
    :type results: dict

    :param save_plot: Flag indicating whether to save the plot.
    :type save_plot: bool

    :param file_name: Name of the file to save the plot as, defaults to "dataset_population_distribution_gender.png".
    :type file_name: str

    :param cohort_id: Optional identifier for the cohort, defaults to None.
    :type cohort_id: Optional[int], optional

    :param suffix: Optional suffix to add to the plot title, defaults to ''.
    :type suffix: str, optional

    :param problem_type: Type of the problem, defaults to 'semantic_segmentation'.
    :type problem_type: str, optional

    :return: File path of the saved plot if saved, otherwise None.
    :rtype: Union[str, None]

    :raises ValueError: If data validation fails.
    """
    plotter = PlotDatasetPopulationDistributionGender(cohort_id=cohort_id)
    json_file = _get_json_file_path(problem_type)
    plotter._validate_data(json_file)
    fig = plotter.plot_gender(json_file, suffix=suffix)
    if save_plot:
        saved_path = plotter.save(fig, filename=file_name)
        plt.close(fig)
        return saved_path
    else:
        plt.show()
        plt.close(fig)
        
@plot_manager.register("object_detection.dataset_population_distribution_gender")
def plot_dataset_population_distribution_gender(
    results: dict,
    save_plot: bool,
    file_name: str = "dataset_population_distribution_gender.png",
    cohort_id: Optional[int] = None,
    suffix: str = '',
    problem_type: str = 'object_detection'
) -> Union[dict, None]:
    """
    Plots dataset population distribution for object detection, optionally saving the plot to disk.

    :param results: Dictionary containing metric results.
    :type results: dict

    :param save_plot: Flag indicating whether to save the plot.
    :type save_plot: bool

    :param file_name: Name of the file to save the plot as, defaults to "dataset_population_distribution_gender.png".
    :type file_name: str

    :param cohort_id: Optional identifier for the cohort, defaults to None.
    :type cohort_id: Optional[int], optional

    :param suffix: Optional suffix to add to the plot title, defaults to ''.
    :type suffix: str, optional

    :param problem_type: Type of the problem, defaults to 'object_detection'.
    :type problem_type: str, optional

    :return: File path of the saved plot if saved, otherwise None.
    :rtype: Union[str, None]

    :raises ValueError: If data validation fails.
    """
    plotter = PlotDatasetPopulationDistributionGender(cohort_id=cohort_id)
    json_file = _get_json_file_path(problem_type)
    plotter._validate_data(json_file)
    fig = plotter.plot_gender(json_file, suffix=suffix)
    if save_plot:
        saved_path = plotter.save(fig, filename=file_name)
        plt.close(fig)
        return saved_path
    else:
        plt.show()
        plt.close(fig)