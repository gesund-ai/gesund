from typing import Optional

import numpy as np


class DiceCalc:
    def __init__(self):
        pass

    def calculate(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate the dice coefficient between two masks

        :param mask1: the first mask
        :type mask1: np.ndarray
        :param mask2: the second mask
        :type mask2: np.ndarray

        :return: the dice coefficient
        :rtype: float
        """
        intersection = np.sum(mask1 * mask2)
        sum_masks = np.sum(mask1) + np.sum(mask2)
        dice = 2.0 * intersection / sum_masks
        return dice
