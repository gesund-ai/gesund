from typing import Union, Optional, List, Dict
import os
import itertools
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import sklearn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager
from gesund.core._metrics.common.average_precision import AveragePrecision

class Classification:
    pass

class SemanticSegmentation:
    pass

class ObjectDetection:
    def _validate_data(self):
        pass
    def __preprocess(self):
        pass
    def __calculate_metrics(self):
        pass
    def calculate(self, data: dict) -> dict:
        self._validate_data(data)
        result = self.__calculate_metrics(data, data.get("class_mappings"))
        return result

class PlotPredictedDistribution:
    def __init__(self, data: dict, cohort_id: Optional[int] = None):
        self.data = data
        #TODO: Continue from here
        self.cohort_id = cohort_id
    
    def _validate_data(self):
        #TODO: Continue init parameters in here.
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

@metric_manager.register("object_detection.predicted_distribution")
def calculate_predicted_distribution(
    data: dict,
    problem_type: str
):
    """
    A wrapper function to calculate the predicted_distribution metrics.
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result

@plot_manager.register("object_detection.predicted_distribution")
def plot_predicted_distribution_od(
    results: dict,
    save_plot: bool,
    file_name: str = "predicted_distribution.png"
) -> Union[str, None]:
    """
    A wrapper function to plot the predicted distribution metrics.
    """
    plotter = PlotPredictedDistribution(data=results)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
