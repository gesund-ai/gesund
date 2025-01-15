from typing import Union, Optional
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager

class Classification:
    pass

class SemanticSegmentation:
    pass

class ObjectDetection:
    def _validate_data(self, data: dict) -> bool:
        pass

    def _preprocess(self, data: dict) -> tuple:
        pass

    def _calculate_metrics(self, data: dict) -> dict:
        pass

    def calculate(self, data: dict) -> dict:
        pass
    

class PlotModelStats:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        pass

    def _validate_data(self):
        pass

    def save(self, fig: Figure, filename: str) -> str:
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
        pass

problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}

@metric_manager.register("object_detection.model_stats")
def calculate_model_stats(data: dict, problem_type: str):
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result

@plot_manager.register("object_detection.model_stats")
def plot_model_stats(
    results: dict,
    save_plot: bool,
    file_name: str = "model_stats.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    
    plotter = PlotModelStats(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
