from typing import Optional


import numpy as np


class IoUCalc:
    def __init__(self):
        pass

    def _calculate_box(self, box1, box2):
        """
        A function to calculate the bounding box

        :param box1:
        :type box1:
        :param box2:
        :type box2:

        :return: the calculated Iou value
        :rtype: float
        """
        x1_i = np.maximum(box1[0], box2[0])
        y1_i = np.maximum(box1[1], box2[1])
        x2_i = np.maximum(box1[2], box2[2])
        y2_i = np.maximum(box1[3], box2[3])

        inter_area = np.maximum(0, x2_i - x1_i) * np.maximum(0, y2_i - y1_i)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        return self._calculate_box(y_true, y_pred)