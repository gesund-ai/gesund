import argparse
import json
import os
import bson
from pathlib import Path


def read_json(file_path):
    """
    Read a JSON file and return its contents as a Python dictionary.

    This function opens the specified file, reads the JSON content, and
    returns it as a Python dictionary. The file should contain valid JSON data.

    :param file_path: (str) The path to the JSON file to read.

    :return: (dict) The JSON data loaded into a Python dictionary.

    :raises FileNotFoundError: If the specified file does not exist.
    :raises json.JSONDecodeError: If the file contains invalid JSON data.
    """

    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_plot_metrics_as_json(overall_metrics, output_dir):
    """
    Save each plot's metrics as individual JSON files in the specified directory.

    This function iterates over the overall metrics dictionary and saves
    each plot's metrics in separate JSON files, named according to the plot names.
    If the directory does not exist, it will be created.

    :param overall_metrics: (dict) A dictionary containing the metrics for multiple plots.
    :param output_dir: (str) Path to the directory where the JSON files should be saved.

    :return: None

    :raises OSError: If the directory cannot be created or if the files cannot be written.
    """

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Iterate over the overall metrics and save each as a separate JSON
    for plot_name, metrics in overall_metrics.items():
        output_file = os.path.join(output_dir, f"{plot_name}.json")
        try:
            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            print(f"Could not save metrics for {plot_name} because: {e}")


import numpy as np
import pandas as pd


class ValidationUtils:
    def __init__(self, meta_pred_true):
        """
        Initialize the ValidationUtils class.

        :param meta_pred_true: (dict or pd.DataFrame) A dictionary or DataFrame containing metadata and predictions.

        :return: None
        """
        if isinstance(meta_pred_true, dict):
            self.meta_pred_true = pd.DataFrame(meta_pred_true).T
        else:
            self.meta_pred_true = meta_pred_true

    def filter_attribute_by_dict(self, target_attribute_dict=None):
        """
        Filters data by more than one attribute.

        :param target_attribute_dict: (dict, optional) A dictionary where keys are attribute names
                                      and values are the desired filter criteria.

        :return: (pd.DataFrame) A DataFrame containing the filtered metadata based on the specified attributes.
        """
        if bool(target_attribute_dict):
            all_params = target_attribute_dict.keys()
            filtered_meta_pred_true = self.meta_pred_true.copy()
            for target_attribute in all_params:
                if self.is_list_numeric(self.meta_pred_true[target_attribute].values):
                    slider_min, slider_max = target_attribute_dict[target_attribute]
                    filtered_meta_pred_true = filtered_meta_pred_true[
                        self.meta_pred_true[target_attribute].between(
                            slider_min, slider_max
                        )
                    ]
                else:
                    target_value = target_attribute_dict[target_attribute]
                    filtered_meta_pred_true = filtered_meta_pred_true[
                        self.meta_pred_true[target_attribute] == target_value
                    ]
            return filtered_meta_pred_true
        else:
            return self.meta_pred_true

    @staticmethod
    def filter_validation_collection_by_meta(validation_collection_data, meta_filter):
        """
        Filters validation collection data based on specified metadata.

        :param validation_collection_data: (list) A list of validation data documents.
        :param meta_filter: (dict) A dictionary of metadata filters.

        :return: (list) A list of filtered validation collection data documents.
        """
        filtered_validation_collection_data = []
        allowed_meta_keys = list(meta_filter.keys())
        for doc in validation_collection_data:
            add = ValidationUtils.is_meta_in_range(
                meta_data=doc.meta_data, meta_filter=meta_filter
            )
            if add:
                for meta_name in list(
                    set(doc.meta_data.keys()) - set(allowed_meta_keys)
                ):
                    doc.meta_data.pop(meta_name)
                filtered_validation_collection_data.append(doc.dict())
        return filtered_validation_collection_data

    @staticmethod
    def is_meta_in_range(meta_data, meta_filter):
        """
        Determines if the metadata values fall within the specified filter ranges.

        :param meta_data: (dict) A dictionary containing metadata.
        :param meta_filter: (dict) A dictionary containing filter criteria.

        :return: (bool) True if metadata values are within specified ranges, False otherwise.
        """
        add = True
        for meta_key in meta_filter:
            if isinstance(meta_filter[meta_key], str):
                if meta_data[meta_key] != meta_filter[meta_key]:
                    add = False
            elif isinstance(meta_filter[meta_key], list):
                if meta_filter[meta_key][0] == "str":
                    add_multi = []
                    if isinstance(meta_filter[meta_key], str):
                        if meta_data[meta_key] != meta_filter[meta_key]:
                            add_multi.append(False)
                    add = any(add_multi)
                elif not (
                    min(meta_filter[meta_key])
                    <= meta_data[meta_key]
                    <= max(meta_filter[meta_key])
                ):
                    add = False
        return add

    def filter_attribute(self, target_attribute_dict):
        """
        Filters data by a single attribute.

        :param target_attribute_dict: (dict) A dictionary where the key is the attribute name
                                       and the value is the desired filter criteria.

        :return: (pd.DataFrame) A DataFrame containing the filtered metadata based on the specified attribute.
        """
        target_attribute = list(target_attribute_dict.keys())[0]
        target_value = target_attribute_dict[target_attribute]
        if self.is_list_numeric(self.meta_pred_true[target_attribute].values):
            slider_min, slider_max = target_attribute_dict[target_attribute]
            filtered_meta_pred_true = self.meta_pred_true[
                self.meta_pred_true[target_attribute].between(slider_min, slider_max)
            ]
        else:
            filtered_meta_pred_true = self.meta_pred_true[
                self.meta_pred_true[target_attribute] == target_value
            ]
        return filtered_meta_pred_true

    def multifilter_attribute(self, target_attributes_dict):
        """
        Filters data by multiple attributes.

        :param target_attributes_dict: (dict) A dictionary where keys are attribute names
                                        and values are the desired filter criteria.

        :return: (pd.DataFrame) A DataFrame containing the filtered metadata based on the specified attributes.
        """
        all_params = target_attributes_dict.keys()
        filtered_meta_pred_true = self.meta_pred_true.copy()
        for target_attribute in all_params:
            if self.is_list_numeric(self.meta_pred_true[target_attribute].values):
                slider_min, slider_max = target_attributes_dict[target_attribute]
                filtered_meta_pred_true = filtered_meta_pred_true[
                    self.meta_pred_true[target_attribute].between(
                        slider_min, slider_max
                    )
                ]
            else:
                target_value = target_attributes_dict[target_attribute]
                filtered_meta_pred_true = filtered_meta_pred_true[
                    self.meta_pred_true[target_attribute] == target_value
                ]
        return filtered_meta_pred_true

    @staticmethod
    def is_list_numeric(x_list):
        """
        Checks if all elements in the list are numeric.

        :param x_list: (list) A list of elements to check.

        :return: (bool) True if all elements are numeric, False otherwise.
        """
        return all(
            [
                isinstance(
                    i,
                    (
                        int,
                        float,
                        np.int,
                        np.int8,
                        np.int16,
                        np.int32,
                        np.float,
                        np.float16,
                        np.float32,
                        np.float64,
                    ),
                )
                for i in x_list
            ]
        )

    @staticmethod
    def polygon_to_mask(poly, shape, max_value=1):
        """
        Converts a polygon defined by a list of points into a binary mask.

        :param poly: (list) A list of points defining the polygon.
        :param shape: (tuple) The shape of the output mask (height, width).
        :param max_value: (int, optional) The value to fill inside the polygon. Default is 1.

        :return: (np.ndarray) A binary mask with the polygon filled in.
        """
        import skimage

        if isinstance(shape, int):
            img = np.zeros((shape, shape, 1), "uint8")
        else:
            shape = int(shape[0]), int(shape[1]), 1
            img = np.zeros(shape)
        xs = [xy["x"] for xy in poly]
        ys = [xy["y"] for xy in poly]
        # fill polygon

        rr, cc = skimage.draw.polygon(xs, ys, img.shape)
        img[rr, cc] = max_value
        return img

    @staticmethod
    def rle_to_mask(mask_rle: str, shape, label=1):
        """
        Converts run-length encoded mask to a binary mask.

        :param mask_rle: (str) Run-length encoded mask as a string (start length).
        :param shape: (tuple) The shape (height, width) of the output mask.
        :param label: (int, optional) The label to assign to the mask pixels. Default is 1.

        :return: (np.ndarray) A binary mask as a numpy array (1 - mask, 0 - background).
        """
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = label
        return img.reshape(shape)  # Needed to align to RLE direction

    @staticmethod
    def mask_to_rle(image):
        """
        Converts a binary mask to run-length encoded format.

        :param image: (np.ndarray) A binary mask as a numpy array (1 - mask, 0 - background).

        :return: (tuple) A tuple containing the run-length encoded string and the image shape.
        """
        image_shape = image.shape
        pixels = image.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        rle_string = " ".join(str(x) for x in runs)
        return rle_string, image_shape

    @staticmethod
    def polygon_to_rle(polygon: list, shape):
        import cv2
        from pycocotools.mask import encode

        """
        Convert a polygon to a Run-Length Encoding (RLE) format mask.

        This function fills a polygon in a mask of the given shape and encodes
        it into Run-Length Encoding format for efficient storage and transmission.

        :param polygon: (list) A list of points representing the vertices of the polygon in the format
                        [[x1, y1], [x2, y2], ...].
        :param shape: (tuple) The shape of the mask to be created, typically in the form
                      (height, width).

        :return: (dict) A dictionary representing the Run-Length Encoding of the mask.

        :raises ValueError: If the polygon points do not form a valid shape or if the shape is invalid.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        polygon = np.asarray(polygon)
        polygon = polygon.reshape(-1, 2)
        cv2.fillPoly(mask, [polygon], 1)

        rle = encode(np.asfortranarray(mask))  # Encoding the mask into RLE
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    @staticmethod
    def calculate_iou(gt_mask, pred_mask, threshold=0.5):
        """
        Calculate the Intersection over Union (IoU) between two masks.

        This function computes the IoU score, which is a measure of overlap
        between two masks. The IoU is calculated as the area of intersection
        divided by the area of union.

        :param gt_mask: (list) A list of points representing the ground truth mask vertices
                         in the format [[x1, y1], [x2, y2], ...].
        :param pred_mask: (list) A list of points representing the predicted mask vertices
                          in the format [[x1, y1], [x2, y2], ...].
        :param threshold: (float, optional) The threshold value for IoU, default is 0.5.

        :return: (float) The IoU score in the range [0, 1]. A score of 0 indicates no overlap,
                 while a score of 1 indicates perfect overlap.

        :raises ValueError: If the input masks are invalid or not in the expected format.
        """
        # Don't even convert polygons to mask if there's no intersection
        gt_xs, gt_ys = [i["x"] for i in gt_mask], [i["y"] for i in gt_mask]
        min_gt_x, max_gt_x = min(gt_xs), max(gt_xs)
        min_gt_y, max_gt_y = min(gt_ys), max(gt_ys)

        pred_xs, pred_ys = [i["x"] for i in pred_mask], [i["y"] for i in pred_mask]
        min_pred_x, max_pred_x = min(pred_xs), max(pred_xs)
        min_pred_y, max_pred_y = min(pred_ys), max(pred_ys)

        shape = max(max_pred_x, max_gt_x, max_gt_y, max_pred_y)

        is_intersect = (max(min_gt_x, min_pred_x) < min(max_gt_x, max_pred_x)) and (
            max(min_gt_y, min_pred_y) < min(max_gt_y, max_pred_y)
        )
        if not is_intersect:
            iou = 0
            return iou

        mask1 = ValidationUtils.polygon_to_mask(gt_mask, shape=shape)
        mask2 = ValidationUtils.polygon_to_mask(pred_mask, shape=shape)

        mask1_area = np.count_nonzero(mask1 == 1)
        mask2_area = np.count_nonzero(mask2 == 1)
        intersection = np.count_nonzero(np.logical_and(mask1 == 1, mask2 == 1))
        iou = intersection / (mask1_area + mask2_area - intersection)
        return iou


class Statistics:
    @staticmethod
    def calculate_confidence_interval(metric, len_, z_value=1.96):
        """
        Calculate the confidence interval for a given metric.

        This function computes the confidence interval for a given metric based
        on the provided sample size and z-score. The confidence interval is
        calculated using the formula: CI = metric ± z * sqrt((metric * (1 - metric)) / len_).

        :param metric: (float) The observed metric for which to calculate the confidence interval.
        :param len_: (int) The sample size used to estimate the confidence interval.
        :param z_value: (float, optional) The z-score corresponding to the desired confidence level,
                        default is 1.96 (for 95% confidence level).

        :return: (tuple) A tuple containing the lower and upper bounds of the confidence interval.

        :raises ValueError: If the sample size is less than or equal to zero.
        """
        metric = np.abs(metric)
        ci_length = z_value * np.sqrt((metric * (1 - metric)) / len_)
        ci_lower = metric - ci_length
        ci_upper = metric + ci_length
        return (ci_lower, ci_upper)

    @staticmethod
    def calculate_histogram(array_, min_, max_, n_bins):
        """
        Calculate a histogram for a given array.

        This function computes a histogram for the specified array, creating bins
        based on the provided minimum and maximum values. The histogram is returned
        as a list of dictionaries with bin categories and counts.

        :param array_: (list or np.ndarray) The array of values for which to calculate the histogram.
        :param min_: (float) The minimum value for the histogram bins.
        :param max_: (float) The maximum value for the histogram bins.
        :param n_bins: (int) The number of bins to create in the histogram.

        :return: (list) A list of dictionaries where each dictionary contains a category
                 and the corresponding value count for that bin.

        :raises ValueError: If n_bins is less than or equal to zero.
        """
        array = np.array(array_)
        bin_spaces = np.linspace(min_, max_, n_bins + 1)
        histogram_list = list()

        for i in range(len(bin_spaces) - 1):
            bin_min = bin_spaces[i]
            bin_max = bin_spaces[i + 1]
            histogram_list.append(
                {
                    "category": f"{bin_min.round(2)}",
                    "value": np.sum([(bin_min < array_) & (array_ <= bin_max)]).item(),
                }
            )

        return histogram_list
