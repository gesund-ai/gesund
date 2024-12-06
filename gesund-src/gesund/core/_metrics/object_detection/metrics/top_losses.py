import numpy as np
import pandas as pd

from ..metrics.average_precision import AveragePrecision


class TopLosses:
    def __init__(self, coco_, class_mappings, loss):
        self.coco_ = coco_
        self.class_mappings = class_mappings

    def calculate_top_losses(self, top_k=100):
        """
        Sorts top loss samples given the target class.
        :param self:
        :param target_class:
        :param top_k: top_k top loss sample to show
        :return: sorted top loss
        """
        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )

        top_losses_list = average_precision.plot_top_losses()

        return top_losses_list
