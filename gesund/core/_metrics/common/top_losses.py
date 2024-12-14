from typing import Union
import os

import numpy as np
import pandas as pd
#TODO 1: check if the methods has specific imports

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core import metric_manager, plot_manager

# TODO 2: metadata handling like age we did in the previous classification metrics

class Classification:
    pass

class SemanticSegmentation(Classification):
    pass


class ObjectDetection(Classification):
    pass


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}

@metric_manager.register("object_detection.top_losses")
def calculate_top_losses(data: dict, problem_type: str):
    pass



@plot_manager.register("object_detection.top_losses")
def plot_top_losses(
        results: dict, save_plot: bool, file_name: str = "top_losses.png"
) -> Union[str, None]:
    pass
