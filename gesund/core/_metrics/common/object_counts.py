from typing import Union, Optional
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager

class Classification:
    pass

class ObjectDetection:
    def __init__(self):
        pass

    def _validate_data(self):
        pass

    def _preprocess(self):
        pass

    def _calculate_object_counts(self):
        pass

    def calculate(self):
        pass
    
class SemanticSegmentation:
    def __init__(self):
        pass

    def _validate_data(self):
        pass

    def _preprocess(self):
        pass

    def _calculate_object_counts(self):
        pass

    def calculate(self):
        pass


class PlotObjectCounts:
    def __init__(self):
        pass
    
    def _validate_data(self):
        pass

    def save(self):
        pass

    def plot(self):
        pass



problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}

@metric_manager.register("semantic_segmentation.object_counts")
@metric_manager.register("object_detection.object_counts")
def calculate_object_count_metric(data: dict, problem_type: str):
    metric_calculator = problem_type_map[problem_type]()
    result = metric_calculator.calculate(data)
    return result


@plot_manager.register("semantic_segmentation.object_counts")
@plot_manager.register("object_detection.object_counts")
def plot_object_counts(
    results: dict,
    save_plot: bool,
    file_name: str = "object_counts.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:

    plotter = PlotObjectCounts(data=results, cohort_id=cohort_id)
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
