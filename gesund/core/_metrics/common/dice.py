from typing import Optional

import numpy as np


class DiceCalc:
    def __init__(self):
        pass

    @staticmethod
    def dice_loss(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6):
        """
        Function to calculate the dice loss

        :param y_true: The true labels as a 4D array (batch_size, height, width, num_classes)
        :type y_true: np.ndarray
        :param y_pred: The predicted labels as a 4d array (batch_size, height, width, num_classes)
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        intersection = np.sum(y_true_flat * y_pred_flat)
        union = np.sum(y_true_flat + y_pred_flat)
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        return round(1.0 - np.mean(dice_score), 4)

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