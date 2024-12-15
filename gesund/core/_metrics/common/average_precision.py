from typing import Union, Optional, Callable, Dict, Any, List
import os

import numpy as np
import pandas as pd
import itertools
from datetime import datetime
import pickle
import sklearn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from gesund.core._utils import ValidationUtils
from gesund.core import metric_manager, plot_manager

COHORT_SIZE_LIMIT = 2
DEBUG = True


class Classification:
    pass


class SemanticSegmentation(Classification):
    pass


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

    def plot_top_losses(self, top_losses=True):

        pred_coco = self.coco_[0]
        gt_coco = self.coco_[1]

        response = self.calculate_coco_metrics(
            pred_coco, gt_coco, top_losses=top_losses
        )

        return response


class ObjectDetection:
    def _validate_data(self, data: dict) -> bool:
        """
        A function to validate the data that is required for metric calculation and plotting.

        :param data: The input data required for calculation, {"prediction":, "ground_truth": , "metadata":}
        :type data: dict

        :return: Status if the data is valid
        :rtype: bool
        """
        # Basic validation checks
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        required_keys = ["prediction", "ground_truth"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Data must contain '{key}'.")

        if len(data["prediction"]) != len(data["ground_truth"]):
            raise ValueError("Prediction and ground_truth must have the same length.")

        if (
            len(
                set(list(data["prediction"].keys())).difference(
                    set(list(data["ground_truth"].keys()))
                )
            )
            > 0
        ):
            raise ValueError("Prediction and ground_truth must have the same keys.")

        return True

    def __preprocess(self, data: dict, get_logits=False) -> tuple:
        """
        Preprocesses the data

        :param data: dictionary containing the data prediction, ground truth, metadata
        :type data: dict
        :param get_logits: in case of multi class classification set to True
        :type get_logits: boolean

        :return: data tuple
        :rtype: tuple
        """
        prediction, ground_truth = [], []
        for image_id in data["ground_truth"]:
            sample_gt = data["ground_truth"][image_id]
            sample_pred = data["prediction"][image_id]
            ground_truth.append(sample_gt["annotation"][0]["label"])

            if get_logits:
                prediction.append(sample_pred["logits"])
            else:
                prediction.append(sample_pred["prediction_class"])
        return (np.asarray(prediction), np.asarray(ground_truth))

    def __calculate_metrics(self, data: dict, class_mapping: dict) -> dict:
        """
        A function to calculate the metrics

        :param data: data dictionary containing data
        :type data: dict
        :param class_mapping: a dictionary with class mapping labels
        :type class_mapping: dict

        :return: results calculated
        :rtype: dict
        """
        class_order = [int(i) for i in list(class_mapping.keys())]

        if len(class_order) > 2:

            prediction, ground_truth = self.__preprocess(data, get_logits=True)

        else:
            prediction, ground_truth = self.__preprocess(data)

            # TODO:

        return data

    def calculate(self, data: dict) -> dict:
        """
        Calculates the Average Precision metric for the given dataset.

        :param data: The input data required for calculation and plotting
                     {"prediction":, "ground_truth": , "class_mappings":}
        :type data: dict

        :return: Calculated metric results
        :rtype: dict
        """
        result = {}

        # Validate the data
        self._validate_data(data)

        # calculate results
        result = self.__calculate_metrics(data, data.get("class_mapping"))

        return result


class PlotAveragePrecision:
    def __init__(
        self,
        class_mappings,
        coco_,
        meta_data_dict=None,
    ):
        self.class_mappings = class_mappings
        self.coco_ = coco_

        if bool(meta_data_dict):
            self.is_meta_exists = True
            meta_df = pd.DataFrame(meta_data_dict).T
            self.validation_utils = ValidationUtils(meta_df)

    def _validate_data(self):
        """
        Validates the data required for plotting.
        """
        if not self.coco_ or not self.class_mappings:
            raise ValueError("COCO data and class mappings must be provided.")

    def save(self, fig: Figure, filename: str) -> str:
        """
        Saves the plot to a file.

        :param filename: Path where the plot image will be saved
        :type filename: str

        :return: Path where the plot image is saved
        :rtype: str
        """
        dir_path = "plots"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if self.cohort_id:
            filepath = f"{dir_path}/{self.cohort_id}_{filename}"
        else:
            filepath = f"{dir_path}/{filename}"
        fig.savefig(filepath, format="png")
        return filepath

    def _plot_performance_by_iou_threshold(
        self, threshold: float, return_points: bool = False
    ) -> Dict[str, Any]:
        """
        Plot performance metrics at specific IoU threshold.

        :param threshold: IoU threshold value
        :type threshold: float
        :param return_points: Whether to return coordinate points
        :type return_points: bool
        :return: Dictionary containing performance metrics data
        :rtype: Dict[str, Any]
        """
        payload_dict = dict()
        payload_dict["type"] = "mixed"

        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )

        metrics = average_precision.calculate_ap_metrics(threshold=threshold)
        coordinates = average_precision.calculate_iou_threshold_graph(
            threshold=threshold
        )

        response = average_precision.calculate_highlighted_overall_metrics(threshold)

        if return_points:
            payload_dict["data"] = {
                "metrics": metrics,
                "coordinates": coordinates,
            }
        else:
            payload_dict["data"] = {"ap_results": response}
        return payload_dict

    def _plot_highlighted_overall_metrics(self, threshold: float) -> Dict[str, Any]:
        """
        Plot highlighted overall metrics at specified threshold.

        :param threshold: IoU threshold value
        :type threshold: float
        :return: Dictionary containing overall metrics data
        :rtype: Dict[str, Any]
        """
        rename_dict = {
            f"map{int(threshold*100)}": f"mAP@{int(threshold*100)}",
            f"map{int(threshold*100)}_11": f"mAP11@{int(threshold*100)}",
            "map50": "mAP@50",
            "map75": "mAP@75",
            "map5095": "mAP@[.50,.95]",
            "mar1": "mAR@max=1",
            "mar10": "mAR@max=10",
            "mar100": "mAR@max=100",
        }
        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )
        overall_metrics = average_precision.calculate_highlighted_overall_metrics(
            threshold
        )
        for metric in list(overall_metrics.keys()):
            overall_metrics[rename_dict[metric]] = overall_metrics.pop(metric)

        val_train_dict = {}
        for value in rename_dict.values():
            val_train_dict[value] = {"Validation": overall_metrics[value]}

        payload_dict = {"type": "overall", "data": val_train_dict}
        return payload_dict

    def _filter_ap_metrics(
        self, target_attribute_dict: Optional[Dict[str, Any]]
    ) -> List[int]:
        """
        Filter average precision metrics based on target attributes.

        :param target_attribute_dict: Dictionary of target attributes for filtering
        :type target_attribute_dict: Optional[Dict[str, Any]]
        :return: List of filtered indices
        :rtype: List[int]
        """
        if target_attribute_dict:
            idxs = self.validation_utils.filter_attribute_by_dict(
                target_attribute_dict
            ).index.tolist()

        return idxs

    def _plot_statistics_classbased_table(
        self,
        threshold: Optional[float] = None,
        target_attribute_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Plot class-based statistics table.

        :param threshold: IoU threshold value
        :type threshold: Optional[float]
        :param target_attribute_dict: Dictionary for filtering by attributes
        :type target_attribute_dict: Optional[Dict[str, Any]]
        :return: Dictionary containing class-based statistics
        :rtype: Dict[str, Any]
        """

        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )

        rename_dict = {
            f"ap10": f"AP@10",
            f"ap10_11": f"AP11@10",
            "ap50": "AP@50",
            "ap75": "AP@75",
            "ap5095": "AP@[.50,.95]",
            "ar1": "AR@max=1",
            "ar10": "AR@max=10",
            "ar100": "AR@max=100",
        }
        if target_attribute_dict:
            idxs = self._filter_ap_metrics(target_attribute_dict)
        else:
            idxs = None

        ap_metrics = average_precision.calculate_ap_metrics(idxs=idxs)

        class_ap_metrics = dict()

        for class_ in ap_metrics["ap50"].keys():
            class_dict = dict()
            for metric in ap_metrics:
                class_dict[rename_dict[metric]] = ap_metrics[metric][class_]
            class_ap_metrics[self.class_mappings[str(class_)]] = class_dict

        payload_dict = {"type": "table", "data": {"Validation": class_ap_metrics}}
        return payload_dict

    def _plot_training_validation_comparison_classbased_table(self):

        threshold = 0.1
        payload_dict = self._plot_highlighted_overall_metrics(threshold)

        keys_to_be_included = ["mAP@50", "mAP@75", "mAP@[.50,.95]"]
        all_keys = payload_dict["data"].keys()

        for key in list(set(all_keys) - set(keys_to_be_included)):
            payload_dict["data"].pop(key)

        payload_dict["type"] = "bar"
        return payload_dict

    def _main_metric(self, threshold):
        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )
        mean_map_given = average_precision.calculate_ap_metrics(threshold=threshold)[
            "mAP"
        ].round(4)
        payload_dict = {f"mAP@{int(threshold*100)}": mean_map_given}
        return payload_dict

    def blind_spot_metrics(
        self, target_attribute_dict: Optional[Dict[str, Any]], threshold: float
    ) -> Dict[str, Any]:
        """
        Plots ROC Curve for target_class.
        References:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        https://plotly.com/python/roc-and-pr-curves/
        :param target_class: target class to produce ROC plot
        :return: payload_dict
        """
        average_precision = AveragePrecision(
            class_mappings=self.class_mappings,
            coco_=self.coco_,
        )
        avg_rename_dict = {
            f"map{int(threshold*100)}": f"mAP@{int(threshold*100)}",
            f"map{int(threshold*100)}_11": f"mAP11@{int(threshold*100)}",
            "map50": "mAP@50",
            "map75": "mAP@75",
            "map5095": "mAP@[.50,.95]",
            "mar1": "mAR@max=1",
            "mar10": "mAR@max=10",
            "mar100": "mAR@max=100",
        }

        rename_dict = {
            "ap10": "AP@10",
            "ap10_11": "AP11@10",
            "ap50": "AP@50",
            "ap75": "AP@75",
            "ap5095": "AP@[.50,.95]",
            "ar1": "AR@max=1",
            "ar10": "AR@max=10",
            "ar100": "AR@max=100",
        }
        idxs = (
            self._filter_ap_metrics(target_attribute_dict)
            if bool(target_attribute_dict)
            else None
        )

        class_metrics = average_precision.calculate_ap_metrics(idxs=idxs)
        overall_metrics = average_precision.calculate_highlighted_overall_metrics(
            threshold
        )

        for key in list(class_metrics.keys()):
            class_metrics[rename_dict[key]] = class_metrics.pop(key)

        for metric in list(overall_metrics.keys()):
            overall_metrics[avg_rename_dict[metric]] = overall_metrics.pop(metric)

        blind_spot_metrics_dict = pd.DataFrame(class_metrics).T.to_dict()
        blind_spot_metrics_dict = {
            str(k): v for k, v in blind_spot_metrics_dict.items()
        }

        blind_spot_metrics_dict["Average"] = overall_metrics

        return blind_spot_metrics_dict

    def plot(self) -> Figure:
        """
        Plots the Top Losses.
        """
        self._validate_data()
        # TODO: Add the plot logic here


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("object_detection.average_precision")
def calculate_average_precision(data: dict, problem_type: str):
    """
    A wrapper function to calculate the Average Precision metric.

    :param data: Dictionary of data: {"prediction": , "ground_truth": }
    :type data: dict
    :param problem_type: Type of the problem
    :type problem_type: str

    :return: Dict of calculated results
    :rtype: dict
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result


@plot_manager.register("object_detection.average_precision")
def plot_average_precision(
    results: dict,
    save_plot: bool,
    file_name: str = "average_precision.png",
    cohort_id: Optional[int] = None,
) -> Union[str, None]:
    """
    A wrapper function to plot the Average Precision.
    """
    plotter = PlotAveragePrecision(
        coco_=results.get("coco_"),
        class_mappings=results.get("class_mappings"),
        loss=results.get("loss"),
        meta_dict=results.get("meta_dict"),
        cohort_id=cohort_id,
    )
    fig = plotter.plot()
    if save_plot:
        return plotter.save(fig, filename=file_name)
    else:
        plt.show()
