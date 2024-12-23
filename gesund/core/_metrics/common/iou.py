from typing import Optional


import numpy as np


class IoUCalc:
    def __init__(self):
        pass

    def calculate_iou_loss(self, gt_boxes: list, pred_boxes: list):
        """
        Calculates the intersection over union loss between ground truth and predicted box

        :param gt_boxes: list of ground truth boxes
        :type gt_boxes: list
        :param pred_boxes: list of predicted boxes
        :type pred_boxes: list

        :return: iou loss
        :rtype: float
        """
        num_gt_boxes = len(gt_boxes)
        num_pred_boxes = len(pred_boxes)

        iou_loss = 0.0

        for i in range(num_gt_boxes):
            max_iou = 0.0
            for j in range(num_pred_boxes):
                iou = self._calculate_box(gt_boxes[i], pred_boxes[j])
                max_iou = max(max_iou, iou)
            iou_loss += (1 - max_iou) ** 2

        iou_loss /= num_gt_boxes
        return iou_loss

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

    @staticmethod
    def calculate_iou_mask(mask1, mask2):
        """
        A function to calculate the intersection over union between two masks

        :param mask1: the first mask
        :type mask1: np.ndarray
        :param mask2: the second mask
        :type mask2: np.ndarray

        :return: the calculated iou value
        :rtype: float
        """
        if mask1.shape != mask2.shape:
            raise ValueError("Masks must have the same shape.")

        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)

        iou = np.sum(intersection) / np.sum(union)
        return round(iou, 4)
