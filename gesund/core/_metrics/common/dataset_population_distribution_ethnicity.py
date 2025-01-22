from typing import Union, Optional
import os
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager

class Classification:
    pass

class SemanticSegmentation:
    def __init__(self):
        pass

    def _validate_data(self, data: dict) -> bool:
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        if "ground_truth" not in data:
            raise ValueError("Data must contain 'ground_truth' key.")
        return True

    def _preprocess(self, data: dict, get_class_only=False) -> pd.DataFrame:
        results = {"gt_class_label": []}  
        for image_id in data["ground_truth"]:
            for annotation in data["ground_truth"][image_id]["annotation"]:
                if annotation["type"] == "mask":  
                    results["gt_class_label"].append(annotation["label"])

        results = pd.DataFrame(results)
        return results

    def _calculate_metrics(self, data: dict) -> dict:
        results = {}
        dataset_pop = self._preprocess(data)
        results["dataset_population_distribution"] = dataset_pop
        return results

    def calculate(self, data: dict) -> dict:
        self._validate_data(data)
        return self._calculate_metrics(data)

class ObjectDetection(SemanticSegmentation):
    def __init__(self):
        super().__init__()

    def _preprocess(self, data: dict, get_class_only=False) -> pd.DataFrame:
        results = {"gt_class_label": []}  
        for image_id in data["ground_truth"]:
            for annotation in data["ground_truth"][image_id]["annotation"]:
                if annotation["type"] == "bbox": 
                    results["gt_class_label"].append(annotation["label"])

        results = pd.DataFrame(results)
        return results

class PlotDatasetPopulationDistributionEthnicity:
    def __init__(self, data=None, cohort_id=None):
        self.data = data
        self.cohort_id = cohort_id

    def _validate_data(self, json_file):
        try:
            with open(json_file, 'r') as f:
                self.data = json.load(f)
            if not isinstance(self.data, list) or not all('Age' in item for item in self.data):
                raise ValueError("Invalid data format: Must be a list of dictionaries with 'Age' field")
        except Exception as e:
            raise ValueError(f"Error reading JSON file: {str(e)}")

    def calculate(self):
        if self.data is None:
            raise ValueError("No data loaded. Call _validate_data first.")

        ages = [item['Age'] for item in self.data]
        return {
            'ages': ages,
            'mean_age': np.mean(ages),
            'median_age': np.median(ages)
        }

    def save(self, fig: Figure, filename: str) -> str:
        dir_path = "plots"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        filepath = f"{dir_path}/{self.cohort_id}_{filename}" if self.cohort_id else f"{dir_path}/{filename}"
        fig.savefig(filepath, format="png")
        return filepath

    def _setup_plot(self, figsize=(12, 7)):
        plt.style.use('dark_background')
        sns.set_theme(style="darkgrid", font_scale=1.2)

        fig = Figure(figsize=figsize, facecolor='black')
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        return fig, ax

    def _annotate_bars(self, ax):
        for p in ax.patches:
            count = int(p.get_height())
            x = p.get_x() + p.get_width()/2
            y = p.get_height()
            ax.annotate(f'{count}', (x, y),
                        ha='center', va='bottom',
                        fontsize=10, color='white')

    def _customize_plot(self, ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=16, pad=20, color='white')
        ax.set_xlabel(xlabel, fontsize=12, color='white')
        ax.set_ylabel(ylabel, fontsize=12, color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')



    def plot_ethnicity(self, json_file, suffix=''):
        self._validate_data(json_file)
        df = pd.DataFrame(self.data)

        fig, ax = self._setup_plot()

        sns.countplot(
            data=df,
            x='Ethnicity',
            hue='Ethnicity',
            ax=ax,
            palette="coolwarm",
            edgecolor='white',
            alpha=0.8,
            legend=False
        )

        self._annotate_bars(ax)
        self._customize_plot(ax, 'Ethnicity Distribution', 'Ethnicity', 'Count')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()

        return fig

problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}

def _get_json_file_path(problem_type: str) -> str:
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

@metric_manager.register("object_detection.dataset_population_distribution_ethnicity")
@metric_manager.register("semantic_segmentation.dataset_population_distribution_ethnicity")
def calculate_dataset_population_distribution_metric(data: dict, problem_type: str):
    metric_calculator = problem_type_map[problem_type]()
    result = metric_calculator.calculate(data)
    return {
        "dataset_population_distribution": result
    }

@plot_manager.register("semantic_segmentation.dataset_population_distribution_ethnicity")
def plot_dataset_population_distribution_ethnicity(
    results: dict,
    save_plot: bool,
    file_name: str = "dataset_population_distribution_ethnicity.png",
    cohort_id: Optional[int] = None,
    suffix: str = '',
    problem_type: str = 'semantic_segmentation'
) -> Union[dict, None]:
    """
    Plot dataset population distribution for semantic segmentation, optionally saving the plot to disk.
    """
    plotter = PlotDatasetPopulationDistributionEthnicity(cohort_id=cohort_id)
    json_file = _get_json_file_path(problem_type)
    plotter._validate_data(json_file)
    fig = plotter.plot_ethnicity(json_file, suffix=suffix)
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()


@plot_manager.register("object_detection.dataset_population_distribution_ethnicity")
def plot_dataset_population_distribution_ethnicity(
    results: dict,
    save_plot: bool,
    file_name: str = "dataset_population_distribution_ethnicity.png",
    cohort_id: Optional[int] = None,
    suffix: str = '',
    problem_type: str = 'object_detection'
) -> Union[dict, None]:
    """
    Plot dataset population distribution for object detection, optionally saving the plot to disk.
    """
    plotter = PlotDatasetPopulationDistributionEthnicity(cohort_id=cohort_id)
    json_file = _get_json_file_path(problem_type)
    plotter._validate_data(json_file)
    fig = plotter.plot_ethnicity(json_file, suffix=suffix)
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()