import numpy as np
from typing import Dict, List, Optional, Union, Any
from ..metrics.top_losses import TopLosses


class PlotTopLosses:
    def __init__(self, coco_, class_mappings, loss=None, meta_dict=None):
        self.loss = loss
        self.meta_dict = meta_dict
        self.coco_ = coco_
        self.class_mappings = class_mappings
        self.is_meta_exists = False
        if self.meta_dict is not None:
            self.is_meta_exists = True
        self.top_losses = TopLosses(
            loss=self.loss, coco_=self.coco_, class_mappings=self.class_mappings
        )

    def _plot_top_misses(
        self,
        top_k: int = 100
        ) -> Dict[str, Any]:
        """
        Calculates minimum loss samples with appropriate information.
        :param self:
        :param top_k: Number of most loss samples
        :return: payload_dict
        """
        top_losses_df = self.top_losses.calculate_top_losses(top_k=top_k)
        for i, item in enumerate(top_losses_df):
            idx = item["image_id"]

        payload_dict = {"type": "image", "data": top_losses_df}

        return payload_dict
