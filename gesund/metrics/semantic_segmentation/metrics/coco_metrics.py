import numpy as np
import miseval
from tqdm import tqdm
import pandas as pd 
import pickle 
import cv2
from scipy.spatial.distance import directed_hausdorff
from skimage import measure
#import seg_metrics.seg_metrics as sg
from scipy import ndimage
import math

from gesund.utils.validation_data_utils import ValidationUtils, Statistics
from .distance_metrics import *


class COCOMetrics:
    SUPPORTED_ANNOTATION_TYPES = ['class', 'polygons', 'boxes', 'rles', 'rle']
    UNSUPPORTED_MISEVAL_METRICS = ["calc_AUC_probability", "calc_Hinge", "calc_Boundary_Distance",
                                   "calc_ConfusionMatrix", "calc_CrossEntropy"]

    def __init__(self, class_mappings, study_list=None):
        """
        Initialize COCOMetrics with class mappings and an optional study list.

        :param class_mappings: A dictionary mapping class IDs to class names.
        :param study_list: Optional list of studies to include in the calculations.
        """
        self.class_mappings = class_mappings
        self.study_list = study_list

    def calculate_iou(self, gt_mask, pred_mask, threshold=0.5):
        """
        Calculate Intersection over Union (IoU) between ground truth and predicted masks.

        :param gt_mask: Ground truth mask (binary array).
        :param pred_mask: Predicted mask (binary array).
        :param threshold: Threshold to binarize the predicted mask.
        :return: IoU value as a float.
        """
        if threshold:
            pred_mask = (pred_mask > threshold) * 1

        gt_empty = int(1 in gt_mask)
        pred_empty = int(1 in pred_mask)

        if pred_empty + gt_empty <= 1:               # Checking if either gt or mask is empty. if both are empty, iou will be 1
            return 1 * (pred_empty + gt_empty != 1)  # if only one of them is empty, then iou will be 0.

        else:
            overlap = pred_mask * gt_mask  # Logical AND
            union = (pred_mask + gt_mask)>0  # Logical OR
            iou = overlap.sum() / float(union.sum())
            return iou

    def calculate_accuracy(self, gt_mask, pred_mask, threshold=0.5):
        """
        Calculate the accuracy of predicted mask against the ground truth mask.

        :param gt_mask: Ground truth mask (binary array).
        :param pred_mask: Predicted mask (binary array).
        :param threshold: Threshold to binarize the predicted mask.
        :return: Accuracy value as a float.
        """
        if threshold:
            pred_mask = (pred_mask > threshold) * 1
        return np.sum(pred_mask == gt_mask) / gt_mask.size

    def calculate_fwiou(self, gt_mask, pred_mask, frequency_rate, threshold=0.5):
        """
        Calculate Frequency Weighted IoU (FWIoU) for a given mask.

        :param gt_mask: Ground truth mask (binary array).
        :param pred_mask: Predicted mask (binary array).
        :param frequency_rate: Frequency of the class in the ground truth.
        :param threshold: Threshold to binarize the predicted mask.
        :return: FWIoU value as a float.
        """
        if threshold:
            pred_mask = (pred_mask > threshold) * 1
        return frequency_rate*np.sum(pred_mask == gt_mask) / gt_mask.size


    def calculate_dataset_pixel_accuracy(self, gt, pred):
        """
        Calculate pixel accuracy for a dataset of ground truth and predicted masks.

        :param gt: Dictionary of ground truth masks indexed by image ID.
        :param pred: Dictionary of predicted masks indexed by image ID.
        :return: Dictionary containing image-wise accuracy and overall pixel accuracy.
        """
        image_ids = list(gt.keys())
        accs = {}  
    
        for image_id in image_ids:
            acc = self.calculate_accuracy(gt[image_id]["mask"], pred[image_id]["mask"])
            accs[image_id] = acc 
        
        acc_results = {"imagewise_acc": accs, "pAcc":np.mean(list(accs.values()))}
        return acc_results


    def calculate_dataset_fwiou(self, gt, pred):
        """
        Calculate FWIoU for a dataset of ground truth and predicted masks.

        :param gt: Dictionary of ground truth masks indexed by image ID.
        :param pred: Dictionary of predicted masks indexed by image ID.
        :return: Dictionary containing image-wise FWIoU and overall FWIoU.
        """
        image_ids = list(gt.keys())
        fwious = {}  
    
        for image_id in image_ids:
            fwiou = self.calculate_fwiou(gt[image_id]["mask"], pred[image_id]["mask"], gt[image_id]["frequency"])
            fwious[image_id] = fwiou 
        
        acc_results = {"imagewise_fwiou": fwious, "fwiou":np.sum(list(fwious.values()))}
        return acc_results

    def calculate_miseval_metrics(self, gt, pred):
        """
        Calculate various metrics using the miseval library for ground truth and predicted masks.

        :param gt: Dictionary of ground truth masks indexed by image ID.
        :param pred: Dictionary of predicted masks indexed by image ID.
        :return: Dictionary containing image-wise and dataset-wise metrics.
        """
        miseval_metrics = dict()
        imagewise_metrics = dict()

        # Get image ids
        image_ids = list(gt.keys())
        # Get metrics from miseval
        metrics = [metric for metric in miseval.__dict__ if "calc" in metric]
        # Discard unwanted metrics
        for unw_metric in self.UNSUPPORTED_MISEVAL_METRICS:
            try:
                metrics.remove(unw_metric)
            except Exception as exc:
                print("UNWANTED_METRIC is not found in the miseval library :: {exc}")

        for image_id in tqdm(image_ids):
            gt_mask, pred_mask = gt[image_id]["mask"], pred[image_id]["mask"]

            metrics_dict = dict() 
            for metric in metrics:
                calculate_metric  = getattr(miseval, metric) 
                metrics_dict[metric.strip("calc_")] = calculate_metric(gt_mask, pred_mask)
            imagewise_metrics[image_id] = metrics_dict
        
        datasetwise_metrics =  pd.DataFrame(imagewise_metrics).mean(axis=1).to_dict()
        miseval_metrics["imagewise_metrics"] = imagewise_metrics
        miseval_metrics["datasetwise_metrics"] = datasetwise_metrics

        return miseval_metrics


    def calculate_dice(self, gt_mask, pred_mask, threshold=0.5):
        """
        Calculate the Dice coefficient between ground truth and predicted masks.

        :param gt_mask: Ground truth mask (binary array).
        :param pred_mask: Predicted mask (binary array).
        :param threshold: Threshold to binarize the predicted mask.
        :return: Dice coefficient value as a float.
        """
        pred_mask_threshold = (pred_mask > threshold) * 1
        dice = (
            np.sum(pred_mask_threshold[gt_mask == 1])
            * 2.0
            / (np.sum(pred_mask_threshold) + np.sum(gt_mask))
        )
        return dice


    def calculate_dataset_iou(self, gt, pred):
        """
        Calculate IoU for a dataset of ground truth and predicted masks.

        :param gt: Dictionary of ground truth masks indexed by image ID.
        :param pred: Dictionary of predicted masks indexed by image ID.
        :return: Dictionary containing image-wise IoU and overall mean IoU.
        """
        image_ids = list(gt.keys())
        ious = {}  
    
        for image_id in image_ids:
            iou = self.calculate_iou(gt[image_id]["mask"], pred[image_id]["mask"])
            ious[image_id] = iou
        
        iou_results = {"imagewise_iou": ious, "mean IoU":np.mean(list(ious.values()))}
        return iou_results

    def calculate_frequency(self, gt):
        """
        Calculate the frequency of the classes in the ground truth masks.

        :param gt: Dictionary of ground truth masks indexed by image ID.
        :return: Updated dictionary with frequencies for each image.
        """
        for image_id in gt:
            mask_ = gt[image_id]["mask"]
            gt[image_id]["frequency"] = mask_.sum()/mask_.size
        return gt

    def calculate_metrics(self, gt, pred):
        """
        Calculate various metrics for evaluating model performance based on ground truth and predicted data.

        This function processes the ground truth and prediction data by converting them into mask format. It then
        calculates metrics such as Intersection over Union (IoU), pixel accuracy, frequency-weighted IoU, and 
        other evaluation metrics relevant to the dataset.

        :param gt: A dictionary representing ground truth data, with image identifiers as keys and their corresponding
            mask data as values.
        :param pred: A dictionary representing predicted data, structured similarly to ground truth data.

        :return: A dictionary containing calculated metrics, including:
            - 'iou': The Intersection over Union metric.
            - 'pAccs': Pixel accuracy metric.
            - 'fwiou': Frequency-weighted Intersection over Union.
            - 'misevals': Additional evaluation metrics.
        """
        # Convert to mask from given annotation type
        gt_mask_dict = self._convert_dict_to_mask(gt)
        pred_mask_dict = self._convert_dict_to_mask(pred)
        
        gt_mask_dict = self.calculate_frequency(gt_mask_dict)

        iou = self.calculate_dataset_iou(gt_mask_dict,pred_mask_dict)
        pAccs = self.calculate_dataset_pixel_accuracy(gt_mask_dict,pred_mask_dict)
        fwiou = self.calculate_dataset_fwiou(gt_mask_dict,pred_mask_dict)
        misevals = self.calculate_miseval_metrics(gt_mask_dict,pred_mask_dict)

        artifacts = {"iou":iou, "pAccs":pAccs, "fwiou":fwiou, "misevals":misevals}
        

    def calculate_APD(self, gt, pred):
        """
        Calculate Average Perpendicular Distance (APD) metrics between ground truth and predicted contours.

        This function identifies contours from the ground truth and prediction data and computes the APD, which
        measures the average distance between the predicted contours and the ground truth contours. It also
        calculates the percentage of "Good Contours" based on a specified distance threshold.

        :param gt: A dictionary representing ground truth data, structured with image identifiers as keys.
        :param pred: A dictionary representing predicted data, structured similarly to ground truth data.

        :return: A dictionary containing the following metrics:
            - 'mean_APD': The mean Average Perpendicular Distance across all images.
            - 'std_APD': The standard deviation of the APD values.
            - 'perc_good_contours': The percentage of contours classified as "Good Contours".
        """

        num_good_contours = 0
        num_total_contours = 0
        image_apd_list = []

        gt_ids = list(gt.keys())
        pred_ids = list(pred.keys())
        matching_image_ids = list(set(gt_ids) & set(pred_ids))

        for image_id in matching_image_ids:
            data_gt = gt[image_id]
            data_pred = pred[image_id]

            contour_gt, _ = cv2.findContours(data_gt['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_pred, _ = cv2.findContours(data_pred['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            num_total_contours += 1
            gt_apds = []

            if len(contour_gt) > 0 and len(contour_pred) > 0:
                for contour_gt_i in contour_gt:
                    apd_list_for_all_pred_contours = []
                    for contour_pred_i in contour_pred:
                        gt_points_pd_contour_distance_list = []

                        # Calculate the distance between each point in gt contour and predicted contour
                        for point_gt in contour_gt_i:
                            distance = np.abs(cv2.pointPolygonTest(contour_pred_i, tuple(int(x) for x in point_gt[0]), True))
                            gt_points_pd_contour_distance_list.append(distance)

                        # Calculate the mean APD for the current pair of contours
                        apd = np.mean(gt_points_pd_contour_distance_list)
                        apd_list_for_all_pred_contours.append(apd)

                    # Find the minimum APD among all predicted contours for the current gt contour
                    apd_for_gt = min(apd_list_for_all_pred_contours)
                    gt_apds.append(apd_for_gt)

            # Check if there are any gt APD values (contours found). Skip this image if no contours are found in gt
            if len(gt_apds) == 0:
                continue
            else:
                # Calculate the mean APD for all the objects in the image (gt_apds)
                image_apd = np.mean(gt_apds)

                # If an image's APD is less than 5, it is considered a "Good Contour"
                # (This is for calculating the percentage of good contours)
                if image_apd < 5:
                    num_good_contours += 1
                image_apd_list.append(image_apd)

        # Calculate the mean and standard deviation of image APD values for the dataset
        apd_mean = np.mean(image_apd_list)
        apd_std = np.std(image_apd_list)

        # Calculate the percentage of good contours among all processed contours
        percentage_good_contours = (num_good_contours / num_total_contours) * 100

        return {
            "mean_APD": apd_mean,
            "std_APD": apd_std,
            "perc_good_contours": percentage_good_contours
        }
        
    # https://github.com/google-deepmind/surface-distance/tree/master
    def calculate_volumetric_metrics(self, gt_mask_series, pred_mask_series):
        """
        Compute volumetric metrics for evaluating segmentation quality between ground truth and predicted volumes.

        This function calculates metrics such as Mean Surface Distance (MSD), Symmetric Surface Overlap, 
        Volume Dice coefficient, and others for volumetric data. It requires ground truth and predicted mask 
        series, ensuring the evaluation is based on valid volumes.

        :param gt_mask_series: A dictionary representing ground truth mask series, with identifiers as keys.
        :param pred_mask_series: A dictionary representing predicted mask series, structured similarly to ground truth.

        :return: A dictionary containing various volumetric metrics, including:
            - 'mean_surface_distance': Mean Surface Distance between ground truth and predictions.
            - 'average_symmetric_surface_distance': Average symmetric surface distance.
            - 'maximum_surface_distance': Maximum surface distance observed.
            - 'symmetric_surface_overlap': Symmetric surface overlap metric.
            - 'surface_dice': Surface Dice coefficient.
            - 'volumetric_dice': Volumetric Dice coefficient.
            - 'volume_overlap_error': Volume overlap error metric.
            - 'relative_volume_difference': Relative difference in volume between ground truth and prediction.
            - 'robust_hausdorff_distance': Robust Hausdorff Distance at the 95th percentile.
        """
        
        image_ids = set(gt_mask_series.keys()).intersection(set(pred_mask_series.keys()))        
        spacing_mm = next(iter(gt_mask_series.values()))['spacing_mm']
        
        shape = next(iter(gt_mask_series.values()))['mask'].shape
        depth = len(image_ids)
        
        gt_volume = np.zeros((depth, shape[0], shape[1]), dtype=np.uint8)
        pred_volume = np.zeros((depth, shape[0], shape[1]), dtype=np.uint8)
        
        i = 0
        for image_id in image_ids:
            gt_masks = gt_mask_series[image_id]['mask']
            pred_masks = pred_mask_series[image_id]['mask']
            
            gt_volume[i, :, :] = gt_masks
            pred_volume[i, :, :] = pred_masks
            i += 1
            
        gt_volume = np.asarray(gt_volume, dtype=bool)
        pred_volume = np.asarray(pred_volume, dtype=bool)
        
        # Ensure the volumes are not empty
        if not gt_volume.any():
            raise ValueError("The ground truth volume is entirely empty (all zeros, background label). Cannot compute surface distances.")
        if not pred_volume.any():
            raise ValueError("The prediction volume is entirely empty (all zeros, background label). Cannot compute surface distances.")
        
        surface_distances = compute_surface_distances(gt_volume, pred_volume, spacing_mm)
        distances_gt_to_pred = surface_distances['distances_gt_to_pred']
        if distances_gt_to_pred.size == 0:
            raise ValueError("The distances_gt_to_pred array is empty. Indicating no overlap between the masks.")
        
        distances_pred_to_gt = surface_distances['distances_pred_to_gt']
        if distances_pred_to_gt.size == 0:
            raise ValueError("The distances_pred_to_gt array is empty. Indicating no overlap between the masks.")        
        
        # Mean Surface Distance (MSD)
        mean_surface_distance = (distances_gt_to_pred.mean() + distances_pred_to_gt.mean()) / 2
        
        # Maximum Surface Distance (MSD)
        max_surface_distance = max(distances_gt_to_pred.max(), distances_pred_to_gt.max())
        
        # Average Symmetric Surface Distance (ASSD)
        average_symmetric_surface_distance = (distances_gt_to_pred.mean() + distances_pred_to_gt.mean()) / 2
        
        # Surface Overlap at Tolerance
        tolerance_mm = np.full_like(distances_gt_to_pred, 2 * np.mean(spacing_mm))
        overlap_gt, overlap_pred = compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm)
        
        # Symmetric Surface Overlap
        symmetric_surface_overlap = (overlap_gt + overlap_pred) / 2
        
        # Surface Dice at Tolerance
        surface_dice = compute_surface_dice_at_tolerance(surface_distances, tolerance_mm)
        
        # Volumetric Dice coefficient (vDSc)
        volumetric_dice = compute_dice_coefficient(gt_volume, pred_volume)
        
        # Volume Overlap Error
        volume_overlap_error = 1 - volumetric_dice

        # Relative Volume Difference (RVD)
        volume_gt = gt_volume.sum()
        volume_pred = pred_volume.sum()
        relative_volume_difference = abs(volume_pred - volume_gt) / volume_gt if volume_gt != 0 else np.NaN
        
        # Robust Hausdorff Distance at 95th percentile
        robust_hausdorff_distance = compute_robust_hausdorff(surface_distances, 95)
        
        return {
            'mean_surface_distance': mean_surface_distance,
            'average_symmetric_surface_distance': average_symmetric_surface_distance,
            'maximum_surface_distance': max_surface_distance,
            'symmetric_surface_overlap': symmetric_surface_overlap,
            'surface_dice': surface_dice,
            'volumetric_dice': volumetric_dice,
            'volume_overlap_error': volume_overlap_error,
            'relative_volume_difference': relative_volume_difference,
            'robust_hausdorff_distance': robust_hausdorff_distance
        }

    def calculate_highlighted_overall_metrics(self, gt, pred):
        """
        Calculate overall highlighted metrics for segmented volumes based on ground truth and predicted data.

        This function aggregates various volumetric and pixel-based metrics across multiple studies. It evaluates
        the performance by utilizing individual study metrics calculated from the provided ground truth and 
        predicted data.

        :param gt: A dictionary representing ground truth data, with image identifiers as keys.
        :param pred: A dictionary representing predicted data, structured similarly to ground truth data.

        :return: A dictionary containing overall metrics across studies, including:
            - 'Vol.DSC': Volume Dice coefficient for the segmented volumes.
            - 'Vol.OE': Volume overlap error.
            - 'mSD': Mean Surface Distance.
            - 'Avg.SSD': Average Symmetric Surface Distance.
            - 'MSD': Maximum Surface Distance.
            - 'SSO': Symmetric Surface Overlap.
            - 'SDSC': Surface Dice coefficient.
            - 'RVD': Relative Volume Difference.
            - 'RHD': Robust Hausdorff Distance.
            - Other volumetric and evaluation metrics aggregated over studies.
        """
        # Convert to mask from given annotation type
        gt_mask_dict = self._convert_dict_to_mask(gt)
        pred_mask_dict = self._convert_dict_to_mask(pred)
                
        if self.study_list:
            gt_dict_of_dicts, pred_dict_of_dicts = self._convert_dict_to_study_dicts(gt_mask_dict, pred_mask_dict)

            overall_metrics = {
                "Vol.DSC": [],
                "Vol.OE": [],
                "mSD": [],
                "Avg.SSD": [],
                "MSD": [],
                "SSO": [],
                "SDSC": [],
                "RVD": [],
                "RHD": [],
                "Vol.mIoU": [],
                "Vol.Pixel Acc.": [],
                "Vol.fwIoU": [],
                "Vol.Avg.Perp.Dist.": [],
                "Vol.%oGoodContours": [],
                "Vol.mSensitivity": [],
                "Vol.mSpecificity": [],
                "Vol.mAUC": [],
                "Vol.mKappa": []
            }
            
            for series_id in gt_dict_of_dicts:
                gt_mask_series = gt_dict_of_dicts[series_id]
                pred_mask_series = pred_dict_of_dicts[series_id]
                
                volumetric_metrics = self.calculate_volumetric_metrics(gt_mask_series, pred_mask_series)
                
                gt_mask_series = self.calculate_frequency(gt_mask_series)
                iou = self.calculate_dataset_iou(gt_mask_series, pred_mask_series)
                pAccs = self.calculate_dataset_pixel_accuracy(gt_mask_series, pred_mask_series)
                fwiou = self.calculate_dataset_fwiou(gt_mask_series, pred_mask_series)
                misevals = self.calculate_miseval_metrics(gt_mask_series, pred_mask_series)
                APD = self.calculate_APD(gt_mask_series, pred_mask_series)
                
                overall_metrics["Vol.DSC"].append(volumetric_metrics["volumetric_dice"])
                overall_metrics["Vol.OE"].append(volumetric_metrics["volume_overlap_error"])
                overall_metrics["mSD"].append(volumetric_metrics["mean_surface_distance"])
                overall_metrics["Avg.SSD"].append(volumetric_metrics["average_symmetric_surface_distance"])
                overall_metrics["MSD"].append(volumetric_metrics["maximum_surface_distance"])
                overall_metrics["SSO"].append(volumetric_metrics["symmetric_surface_overlap"])
                overall_metrics["SDSC"].append(volumetric_metrics["surface_dice"])
                overall_metrics["RVD"].append(volumetric_metrics["relative_volume_difference"])
                overall_metrics["RHD"].append(volumetric_metrics["robust_hausdorff_distance"])
                overall_metrics["Vol.mIoU"].append(iou["mean IoU"])
                overall_metrics["Vol.Pixel Acc."].append(pAccs["pAcc"])
                overall_metrics["Vol.fwIoU"].append(fwiou["fwiou"])
                overall_metrics["Vol.Avg.Perp.Dist."].append(APD["mean_APD"])
                overall_metrics["Vol.%oGoodContours"].append(round(APD["perc_good_contours"], 3))
                overall_metrics["Vol.mSensitivity"].append(misevals["datasetwise_metrics"]["Sensitivity"])
                overall_metrics["Vol.mSpecificity"].append(misevals["datasetwise_metrics"]["Specificity"])
                overall_metrics["Vol.mAUC"].append(misevals["datasetwise_metrics"]["AUC"])
                overall_metrics["Vol.mKappa"].append(misevals["datasetwise_metrics"]["Kapp"])

            # Average the metrics across all series
            final_dict = {key: sum(values) / len(values) for key, values in overall_metrics.items()}
            
            return final_dict

        else:
            gt_mask_dict = self.calculate_frequency(gt_mask_dict)

            iou = self.calculate_dataset_iou(gt_mask_dict, pred_mask_dict)
            pAccs = self.calculate_dataset_pixel_accuracy(gt_mask_dict, pred_mask_dict)
            fwiou = self.calculate_dataset_fwiou(gt_mask_dict, pred_mask_dict)
            misevals = self.calculate_miseval_metrics(gt_mask_dict, pred_mask_dict)
            APD = self.calculate_APD(gt_mask_dict, pred_mask_dict)

            return {
                "mean IoU": iou["mean IoU"],
                "Pixel Accuracy": pAccs["pAcc"],
                "fwIoU": fwiou["fwiou"],
                "Dice Score": misevals["datasetwise_metrics"]["DSC"],
                "Avg. Perp. Dist.": APD["mean_APD"],
                "% of Good Contours": round(APD["perc_good_contours"], 3),
                "Avg. Hausdorff Dist.": misevals["datasetwise_metrics"]["AverageHausdorffDistance"],
                "mean Sensitivity": misevals["datasetwise_metrics"]["Sensitivity"],
                "mean Specificity": misevals["datasetwise_metrics"]["Specificity"],
                "mean AUC": misevals["datasetwise_metrics"]["AUC"],
                "mean Kappa": misevals["datasetwise_metrics"]["Kapp"],
            }

    def calculate_statistics_classbased_table(self, gt, pred, target_attribute_dict=None):
        """
        Calculate statistics based on ground truth and prediction data for each class.

        This function filters ground truth and predicted masks by class and computes various volumetric metrics 
        such as Dice Similarity Coefficient (DSC), Volume Overlap Error (Vol.OE), and others for each class in 
        the predictions. The results are aggregated and returned as a dictionary.

        :param gt: A dictionary containing the ground truth masks with image identifiers as keys.
        :param pred: A dictionary containing the predicted masks with image identifiers as keys.
        :param target_attribute_dict: (dict, optional) A dictionary containing target attributes for filtering 
                                    the study ID. If provided, only the masks corresponding to this study ID 
                                    will be processed.
        
        :return: A dictionary containing the calculated metrics for each class. Each key is the class mapping 
                and the value is another dictionary of metrics.
        """
        metrics_dict = dict()
        if target_attribute_dict:
            if target_attribute_dict.get("study_id"):
                self.study_list = [target_attribute_dict["study_id"]]

        gt_mask_dict = self._convert_dict_to_mask(gt)
        pred_mask_dict = self._convert_dict_to_mask(pred)

        for class_key, class_mapping in self.class_mappings.items():
            gt_filtered = {}
            pred_filtered = {}

            # Filter gt_mask_dict and pred_mask_dict by class
            for key in gt_mask_dict:
                gt_masks = gt_mask_dict[key]
                if isinstance(gt_masks, list):
                    gt_filtered[key] = [mask for mask in gt_masks if mask['class'] == int(class_key)]
                else:
                    if gt_masks['class'] == int(class_key):
                        gt_filtered[key] = [gt_masks]

            for key in pred_mask_dict:
                pred_masks = pred_mask_dict[key]
                if isinstance(pred_masks, list):
                    pred_filtered[key] = [mask for mask in pred_masks if mask['class'] == int(class_key)]
                else:
                    if pred_masks['class'] == int(class_key):
                        pred_filtered[key] = [pred_masks]

            # Ensure gt_filtered and pred_filtered have no empty lists
            gt_filtered = {k: v for k, v in gt_filtered.items() if v}
            pred_filtered = {k: v for k, v in pred_filtered.items() if v}

            if gt_filtered and pred_filtered:
                
                if self.study_list:
                    
                    gt_dict_of_dicts, pred_dict_of_dicts = self._convert_dict_to_study_dicts(gt_filtered, pred_filtered)
                    
                    overall_metrics = {
                        "Vol.DSC": [],
                        "Vol.OE": [],
                        "mSD": [],
                        "Avg.SSD": [],
                        "MSD": [],
                        "SSO": [],
                        "SDSC": [],
                        "RVD": [],
                        "RHD": [],
                        "Vol.mIoU": [],
                        "Vol.Pixel Acc.": [],
                        "Vol.fwIoU": [],
                        "Vol.Avg.Perp.Dist.": [],
                        "Vol.%oGoodContours": [],
                        "Vol.mSensitivity": [],
                        "Vol.mSpecificity": [],
                        "Vol.mAUC": [],
                        "Vol.mKappa": []
                    }
                    
                    for series_id in gt_dict_of_dicts:
                        gt_mask_series = gt_dict_of_dicts[series_id]
                        pred_mask_series = pred_dict_of_dicts[series_id]
                        
                        volumetric_metrics = self.calculate_volumetric_metrics(gt_mask_series, pred_mask_series)
                        
                        gt_mask_series = self.calculate_frequency(gt_mask_series)
                        iou = self.calculate_dataset_iou(gt_mask_series, pred_mask_series)
                        pAccs = self.calculate_dataset_pixel_accuracy(gt_mask_series, pred_mask_series)
                        fwiou = self.calculate_dataset_fwiou(gt_mask_series, pred_mask_series)
                        misevals = self.calculate_miseval_metrics(gt_mask_series, pred_mask_series)
                        APD = self.calculate_APD(gt_mask_series, pred_mask_series)
                        
                        overall_metrics["Vol.DSC"].append(volumetric_metrics["volumetric_dice"])
                        overall_metrics["Vol.OE"].append(volumetric_metrics["volume_overlap_error"])
                        overall_metrics["mSD"].append(volumetric_metrics["mean_surface_distance"])
                        overall_metrics["Avg.SSD"].append(volumetric_metrics["average_symmetric_surface_distance"])
                        overall_metrics["MSD"].append(volumetric_metrics["maximum_surface_distance"])
                        overall_metrics["SSO"].append(volumetric_metrics["symmetric_surface_overlap"])
                        overall_metrics["SDSC"].append(volumetric_metrics["surface_dice"])
                        overall_metrics["RVD"].append(volumetric_metrics["relative_volume_difference"])
                        overall_metrics["RHD"].append(volumetric_metrics["robust_hausdorff_distance"])
                        overall_metrics["Vol.mIoU"].append(iou["mean IoU"])
                        overall_metrics["Vol.Pixel Acc."].append(pAccs["pAcc"])
                        overall_metrics["Vol.fwIoU"].append(fwiou["fwiou"])
                        overall_metrics["Vol.Avg.Perp.Dist."].append(APD["mean_APD"])
                        overall_metrics["Vol.%oGoodContours"].append(round(APD["perc_good_contours"], 3))
                        overall_metrics["Vol.mSensitivity"].append(misevals["datasetwise_metrics"]["Sensitivity"])
                        overall_metrics["Vol.mSpecificity"].append(misevals["datasetwise_metrics"]["Specificity"])
                        overall_metrics["Vol.mAUC"].append(misevals["datasetwise_metrics"]["AUC"])
                        overall_metrics["Vol.mKappa"].append(misevals["datasetwise_metrics"]["Kapp"])
                    
                    # Average the metrics across all series
                    final_dict = {key: sum(values) / len(values) for key, values in overall_metrics.items()}
                        
                    metrics_dict[class_mapping] = final_dict

                else:
                    gt_mask_dict = self.calculate_frequency(gt_mask_dict)
                    
                    iou = self.calculate_dataset_iou(gt_mask_dict, pred_mask_dict)
                    pAccs = self.calculate_dataset_pixel_accuracy(gt_mask_dict, pred_mask_dict)
                    fwiou = self.calculate_dataset_fwiou(gt_mask_dict, pred_mask_dict)
                    misevals = self.calculate_miseval_metrics(gt_mask_dict, pred_mask_dict)
                    APD = self.calculate_APD(gt_mask_dict, pred_mask_dict)
                    
                    metrics_dict[class_mapping] = {
                        "mean IoU": iou["mean IoU"],
                        "pAccs": pAccs["pAcc"],
                        "fwIoU": fwiou["fwiou"],
                        "Dice": misevals["datasetwise_metrics"]["DSC"],
                        "Avg. Perp. Dist.": APD["mean_APD"],
                        "% of Good Contours": round(APD["perc_good_contours"], 3),
                        "Avg. Hausdorff Dist.": misevals["datasetwise_metrics"]["AverageHausdorffDistance"],
                        "mean Sensitivity": misevals["datasetwise_metrics"]["Sensitivity"],
                        "mean Specificity": misevals["datasetwise_metrics"]["Specificity"],
                        "mean AUC": misevals["datasetwise_metrics"]["AUC"],
                        "mean Kappa": misevals["datasetwise_metrics"]["Kapp"]
                    }
                
        return metrics_dict


    def _detect_annotation_type(self, dict_):
        """
        Detect the type of annotations present in the given dictionary.

        This function identifies the annotation type from a dictionary containing various annotation formats. 
        It returns the first found type that matches the supported annotation types.

        :param dict_: A dictionary containing annotations where keys are image identifiers.

        :return: The detected annotation type as a string if found; otherwise, None.
        """
        random_image_id = list(dict_.keys())[0]
        dict_annotation_types = list(dict_[random_image_id].keys())
        intersection_annotation_types = list(set(dict_annotation_types) & set(self.SUPPORTED_ANNOTATION_TYPES))

        for annotation_type in intersection_annotation_types:
            if bool(dict_[random_image_id][annotation_type]):
                return annotation_type

    def _convert_dict_to_mask(self, dict_):
        """
        Convert a dictionary of annotations to a mask representation.

        This function processes the input dictionary, detecting the annotation type and converting relevant 
        data into a mask format. The output is a new dictionary with masks.

        :param dict_: A dictionary containing annotations to be converted.

        :return: A dictionary where keys are image identifiers and values are dictionaries containing masks and 
                corresponding class labels.
        """
        new_dict = dict()
        annotation_type =self._detect_annotation_type(dict_)

        for key in dict_:
            non_mask = dict_[key][annotation_type]
            if annotation_type=="rles":
                for layer in non_mask:
                    new_dict[key] = {"mask": ValidationUtils.rle_to_mask(layer["rle"], layer["shape"]), "class": layer["class"]}
        
        
        return new_dict
    
    def _convert_dict_to_study_dicts(self, gt, pred):
        """
        Convert ground truth and prediction dictionaries into study-based dictionaries.

        This function organizes the ground truth and prediction data into dictionaries keyed by study ID. 
        Each study ID contains a further dictionary mapping image identifiers to their corresponding mask data.

        :param gt: A dictionary containing ground truth data with image identifiers as keys.
        :param pred: A dictionary containing predicted data with image identifiers as keys.

        :return: Two dictionaries: one for ground truth data and one for predicted data, both organized by 
                study ID.
        """
        gt_dict_of_dicts = {}
        pred_dict_of_dicts = {}

        # Process ground truth
        for image_id, ground_truth_items in gt.items():
            if isinstance(ground_truth_items, list):
                for ground_truth_item in ground_truth_items:
                    series_id = ground_truth_item.get("series_id", None)
                    if series_id not in gt_dict_of_dicts:
                        gt_dict_of_dicts[series_id] = {}
                    gt_dict_of_dicts[series_id][image_id] = ground_truth_item
            else:
                # Handle case where ground_truth_items is a single dictionary
                series_id = ground_truth_items.get("series_id", None)
                if series_id not in gt_dict_of_dicts:
                    gt_dict_of_dicts[series_id] = {}
                gt_dict_of_dicts[series_id][image_id] = ground_truth_items

        # Process predictions
        for image_id, pred_items in pred.items():
            if isinstance(pred_items, list):
                for pred_item in pred_items:
                    series_id = pred_item.get("series_id", None)
                    if series_id not in pred_dict_of_dicts:
                        pred_dict_of_dicts[series_id] = {}
                    pred_dict_of_dicts[series_id][image_id] = pred_item
            else:
                # Handle case where pred_items is a single dictionary
                series_id = pred_items.get("series_id", None)
                if series_id not in pred_dict_of_dicts:
                    pred_dict_of_dicts[series_id] = {}
                pred_dict_of_dicts[series_id][image_id] = pred_items

        return gt_dict_of_dicts, pred_dict_of_dicts

    def create_artifacts(self, gt, pred, artifacts_path=None):
        """
        Create artifacts from the ground truth and prediction masks.

        This function generates various metrics such as IoU, pixel accuracy, and others by processing ground 
        truth and prediction data. The results are aggregated and can be saved to a specified artifacts path.

        :param gt: A dictionary containing the ground truth masks.
        :param pred: A dictionary containing the predicted masks.
        :param artifacts_path: (str, optional) Path where artifacts should be saved. If None, artifacts are 
                            not saved.
        
        :return: A dictionary containing combined metrics and information about the generated artifacts.
        """
        
        if self.study_list:
            gt_mask_dict = self._convert_dict_to_mask(gt)
            pred_mask_dict = self._convert_dict_to_mask(pred)
            
            gt_dict_of_dicts, pred_dict_of_dicts = self._convert_dict_to_study_dicts(gt_mask_dict, pred_mask_dict)
            
            combined_metrics = {
                "iou": {"imagewise_iou": {}, "mean IoU": 0},
                "pAccs": {"imagewise_acc": {}, "pAcc": 0},
                "fwiou": {"imagewise_fwiou": {}, "fwiou": 0},
                "misevals": {"imagewise_metrics": {}, "datasetwise_metrics": {}}
            }
            
            num_series = len(gt_dict_of_dicts)
            
            for series_id in gt_dict_of_dicts:
                gt_mask_series = gt_dict_of_dicts[series_id]
                pred_mask_series = pred_dict_of_dicts[series_id]
                
                gt_mask_dict = self.calculate_frequency(gt_mask_series)

                iou = self.calculate_dataset_iou(gt_mask_series, pred_mask_series)
                pAccs = self.calculate_dataset_pixel_accuracy(gt_mask_series, pred_mask_series)
                fwiou = self.calculate_dataset_fwiou(gt_mask_series, pred_mask_series)
                misevals = self.calculate_miseval_metrics(gt_mask_series, pred_mask_series)
                
                # Aggregate the metrics into combined_metrics
                combined_metrics["iou"]["imagewise_iou"].update(iou["imagewise_iou"])
                combined_metrics["iou"]["mean IoU"] += iou["mean IoU"]

                combined_metrics["pAccs"]["imagewise_acc"].update(pAccs["imagewise_acc"])
                combined_metrics["pAccs"]["pAcc"] += pAccs["pAcc"]

                combined_metrics["fwiou"]["imagewise_fwiou"].update(fwiou["imagewise_fwiou"])
                combined_metrics["fwiou"]["fwiou"] += fwiou["fwiou"]

                combined_metrics["misevals"]["imagewise_metrics"].update(misevals["imagewise_metrics"])
                
                # Aggregate datasetwise_metrics
                for key, value in misevals["datasetwise_metrics"].items():
                    if key not in combined_metrics["misevals"]["datasetwise_metrics"]:
                        combined_metrics["misevals"]["datasetwise_metrics"][key] = 0
                    combined_metrics["misevals"]["datasetwise_metrics"][key] += value

            # Compute mean values by dividing by the number of series
            combined_metrics["iou"]["mean IoU"] /= num_series
            combined_metrics["pAccs"]["pAcc"] /= num_series
            combined_metrics["fwiou"]["fwiou"] /= num_series
            for key in combined_metrics["misevals"]["datasetwise_metrics"]:
                combined_metrics["misevals"]["datasetwise_metrics"][key] /= num_series

            artifacts_dict = combined_metrics
            
        else:
            gt_mask_dict = self._convert_dict_to_mask(gt)
            pred_mask_dict = self._convert_dict_to_mask(pred)
            
            gt_mask_dict = self.calculate_frequency(gt_mask_dict)

            iou = self.calculate_dataset_iou(gt_mask_dict,pred_mask_dict)
            pAccs = self.calculate_dataset_pixel_accuracy(gt_mask_dict,pred_mask_dict)
            fwiou = self.calculate_dataset_fwiou(gt_mask_dict,pred_mask_dict)
            misevals = self.calculate_miseval_metrics(gt_mask_dict,pred_mask_dict)
            artifacts_dict = {"iou":iou, "pAccs":pAccs, "fwiou":fwiou, "misevals":misevals}

        if artifacts_path is None:
            return artifacts_dict
            
        with open(artifacts_path, 'wb') as f: 
            pickle.dump(artifacts_dict, f)