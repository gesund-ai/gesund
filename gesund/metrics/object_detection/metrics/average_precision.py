import itertools
from datetime import datetime
import os
import pickle

import numpy as np
import pandas as pd
import sklearn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class AveragePrecision:
    def __init__(self, class_mappings, coco_):
        self.class_mappings = class_mappings
        self.class_idxs = [int(i) for i in list(class_mappings.keys())]
        self.coco_ = coco_

    def calculate_coco_metrics(
        self,
        pred_coco,
        gt_coco,
        return_class_metrics=False,
        return_points=False,
        return_conf_dist=False,
        threshold=None,
        top_losses=False,
        idxs=None,
    ):
        annType = ["segm", "bbox", "keypoints"]
        annType = annType[1]  # specify type here

        annFile = gt_coco
        cocoGt = COCO(annFile)

        resFile = pred_coco
        cocoDt = cocoGt.loadRes(resFile)

        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, annType)

        if top_losses:
            class_mean_list = []
            losses_list = []
            ids = cocoEval.params.catIds
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            for img in imgIds:
                for ids_ in ids:
                    a = cocoEval.ious[(img, ids_)]
                    if len(a) != 0:
                        class_mean = a.max(1).mean()
                        class_mean_list.append(class_mean)
                    else:
                        pass

                mean = sum(class_mean_list) / len(class_mean_list)
                losses_list.append({"image_id": img, "mIoU": mean})

            sorted_list = sorted(losses_list, key=lambda x: x["mIoU"], reverse=True)
            for i, item in enumerate(sorted_list):
                item["rank"] = i + 1
            return losses_list

        if return_conf_dist:
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            eval_imgs = [i for i in cocoEval.evalImgs if i is not None]

            return eval_imgs

        if return_points:
            xy_list = {}
            ids = cocoEval.params.catIds
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()

            for ids_ in ids:
                coordinates = cocoEval.plotPrecisionRecallGraph(threshold, ids_)
                xy_list[ids_] = coordinates

            return xy_list

        if return_class_metrics:

            metrics = []
            ids = cocoEval.params.catIds

            if idxs:
                cocoEval.params.imgIds = idxs

            for ids_ in ids:

                cocoEval.params.catIds = [ids_]  # Class-wise metrics outputted
                cocoEval.evaluate()  # [ 0_dict(), 1_dict(), 2_dict(), ...]
                cocoEval.accumulate()
                if threshold:
                    metric = cocoEval.summarize(threshold)
                else:
                    metric = cocoEval.summarize()

                metrics.append(metric)

            metrics_by_class = {
                metric: {i: metrics[i][metric] for i in range(len(ids))}
                for metric in metrics[0]
            }
            metrics_final = {k.replace("m", ""): v for k, v in metrics_by_class.items()}
            if threshold:
                metrics_final["APs"] = metrics_final.pop(f"ap{threshold}")
                metrics_final["mAP"] = np.mean(list(metrics_final["APs"].values()))

        else:
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            metrics_final = cocoEval.summarize()

        return metrics_final

    def calculate_highlighted_overall_metrics(self, threshold):

        pred_coco = self.coco_[0]
        gt_coco = self.coco_[1]

        return self.calculate_coco_metrics(pred_coco, gt_coco)

    def calculate_ap_metrics(
        self, threshold=None, idxs=None, return_map=False, return_class_metrics=True
    ):
        pred_coco = self.coco_[0]
        gt_coco = self.coco_[1]

        class_based_coco_metrics = self.calculate_coco_metrics(
            pred_coco,
            gt_coco,
            return_class_metrics=return_class_metrics,
            threshold=threshold,
            idxs=idxs,
        )

        return class_based_coco_metrics

    def calculate_iou_threshold_graph(self, threshold, return_points=True):

        pred_coco = self.coco_[0]
        gt_coco = self.coco_[1]

        return self.calculate_coco_metrics(
            pred_coco, gt_coco, return_points=return_points, threshold=threshold
        )

    def calculate_confidence_distribution(
        self, threshold, predicted_class=None, return_conf_dist=True
    ):
        pred_coco = self.coco_[0]
        gt_coco = self.coco_[1]
        eval_imgs = self.calculate_coco_metrics(
            pred_coco, gt_coco, return_conf_dist=return_conf_dist
        )
        existing_image_ids = [i["image_id"] for i in eval_imgs]
        image_id_scores = dict()
        for image_id in existing_image_ids:
            for cls_id in self.class_mappings:
                cls_id = int(cls_id)
                if predicted_class is not None and cls_id != predicted_class:
                    continue
                confidences = [
                    i["dtScores"]
                    for i in eval_imgs
                    if i["image_id"] == image_id and i["category_id"] == cls_id
                ]
                if len(confidences) != 0:
                    image_id_scores[image_id] = np.mean(np.array(confidences).flatten())
                    break
        return image_id_scores

    def plot_top_losses(self, top_losses=True):

        pred_coco = self.coco_[0]
        gt_coco = self.coco_[1]

        response = self.calculate_coco_metrics(
            pred_coco, gt_coco, top_losses=top_losses
        )

        return response
