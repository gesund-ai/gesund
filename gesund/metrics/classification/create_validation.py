import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from .plots.plot_driver import ClassificationPlotDriver
from gesund.metrics.classification.classification_metric_plot import Classification_Plot

class ValidationCreation:
    """
    Class responsible for creating validation data and generating classification metrics and plots.

    :param batch_job_id: Unique identifier for the batch job.
    :param filter_field: The field to filter the data, default is "image_url".
    :param generate_metrics: Whether to generate metrics during validation creation, default is True.
    """
    def __init__(self, batch_job_id, filter_field="image_url", generate_metrics=True):
        """
        Initialize the ValidationCreation class.

        :param batch_job_id: Identifier for the batch job.
        :param filter_field: Field used for filtering the data. Defaults to "image_url".
        :param generate_metrics: Flag to control whether metrics should be generated. Defaults to True.
        """

        self.batch_job_id = batch_job_id
        self.filter_field = filter_field
        self.generate_metrics = generate_metrics

    def create_validation_collection_data(self, successful_batch_data, annotation_data, format=None, meta_data=None):
        """
        Create validation collection data from successful batch predictions and annotations.

        :param successful_batch_data: Dictionary containing prediction data for each item ID.
        :param annotation_data: Dictionary containing annotation (ground truth) data for each item ID.
        :param format: Format for the data (currently unused). Defaults to None.
        :param meta_data: Additional metadata for each item. Defaults to None.

        :raises ValueError: If annotation data is missing for any item ID.

        :return: A list of dictionaries with validation collection data including predictions, confidence, and ground truth.
        """

        validation_collection_data = []

        for item_id in successful_batch_data:
            if item_id not in annotation_data:
                raise ValueError(f"Annotation data for item ID {item_id} is missing")

            batch_item = successful_batch_data[item_id]
            annotation_item = annotation_data[item_id]

            information_dict = {
                "batch_job_id": self.batch_job_id,
                "image_id": batch_item["image_id"],
                "confidence": batch_item["confidence"],
                "logits": batch_item["logits"],
                "prediction_class": batch_item["prediction_class"],
                "loss": batch_item["loss"],
                "ground_truth": annotation_item["annotation"][0]["label"],
                "meta_data": meta_data[item_id]["metadata"] if meta_data else {},
                "created_timestamp": time.time()
            }

            validation_collection_data.append(information_dict)            

        return validation_collection_data
    
    def load(self, validation_collection_data, class_mappings, filtering_meta=None):
        """
        Load the validation data, calculate metrics, and return the overall metrics.

        :param validation_collection_data: A list of validation collection data dictionaries.
        :param class_mappings: Mapping of class IDs to class names.
        :param filtering_meta: Metadata used for filtering. Defaults to None.

        :return: Overall metrics computed from the validation data.
        """

        generate_metrics = True
        plotting_data = self._load_plotting_data(validation_collection_data)

        # Create per image variables
        ground_truth_dict = plotting_data["per_image"]["ground_truth"]
        meta_data_dict = plotting_data["per_image"]["meta_data"]
        logits_dict = plotting_data["per_image"]["logits"]
        if generate_metrics:
            loss_dict = plotting_data["per_image"]["loss"]

        # Create validation variables
        true = pd.DataFrame(ground_truth_dict, index=[0]).loc[0].astype(int)
        class_mappings = class_mappings
        pred = pd.DataFrame(logits_dict)
        meta = pd.DataFrame(meta_data_dict).T

        loss = None
        if generate_metrics:
            loss = pd.DataFrame(loss_dict, index=[0])
        sample_size = len(true)
        class_order = list(range(len(class_mappings.keys())))
        true.name = "true"
        pred.name = "pred"
        if np.shape(pred.shape)[0] == 1:
            pred_categorical = pred
            pred_categorical.name = "pred_categorical"
            pred_categorical = pred_categorical.astype(int)
        else:
            pred_logits = pred
            pred_categorical = pred.idxmax()
            pred_categorical.name = "pred_categorical"
        meta_pred_true = pd.concat([meta, pred_categorical, true], axis=1)

        self.plot_driver = ClassificationPlotDriver(
            true=true,
            pred=pred,
            meta=meta,
            pred_categorical=pred_categorical,
            pred_logits=pred_logits,
            meta_pred_true=meta_pred_true,
            class_mappings=class_mappings,
            loss=loss,
            filtering_meta=filtering_meta,
        )
        
        overall_metrics = self.plot_driver._calling_all_plots()
        
        return overall_metrics
    
    def plot_metrics(self, metrics, jsons_dir, plot_dir):
        """
        Plot the specified metrics and save them to a given directory.

        :param metrics: Dictionary containing the metrics to be plotted.
        :param jsons_dir: Path to the directory containing JSON files with plot data.
        :param plot_dir: Path to the directory where plots will be saved.

        :return: None
        """

        draw_type = 'class_distributions'
        plot = Classification_Plot(class_distributions_path=os.path.join(jsons_dir, f'plot_{draw_type}.json'))
        plot.draw('class_distributions', metrics=['normal'], threshold=10, save_path=os.path.join(plot_dir, f'{draw_type}.png'))

        draw_type = 'blind_spot'
        plot = Classification_Plot(blind_spot_path=os.path.join(jsons_dir, f'plot_{draw_type}_metrics.json'))
        plot.draw('blind_spot', class_type='Average', save_path=os.path.join(plot_dir, f'{draw_type}.png'))

        draw_type = 'performance_by_threshold'
        plot = Classification_Plot(performance_threshold_path=os.path.join(jsons_dir, f'plot_class_{draw_type}.json'))
        plot.draw('performance_by_threshold', graph_type='graph_1', metrics=['F1', 'Sensitivity', 'Specificity', 'Precision'], threshold=0.2, save_path=os.path.join(plot_dir, f'{draw_type}.png'))
        
        draw_type = 'roc'
        plot = Classification_Plot(roc_statistics_path=os.path.join(jsons_dir, f'plot_{draw_type}_multiclass_statistics.json'))
        plot.draw('roc', roc_class='normal', save_path=os.path.join(plot_dir, f'{draw_type}.png'))

        draw_type = 'precision_recall'
        plot = Classification_Plot(precision_recall_statistics_path=os.path.join(jsons_dir, f'plot_{draw_type}_multiclass_statistics.json'))
        plot.draw('precision_recall', pr_class='normal', save_path=os.path.join(plot_dir, f'{draw_type}.png'))

        draw_type = 'confidence_histogram'
        plot = Classification_Plot(confidence_histogram_path=os.path.join(jsons_dir, f'plot_{draw_type}_scatter_distribution.json'))
        plot.draw('confidence_histogram', metrics=['TP','FP'], threshold=0.5, save_path=os.path.join(plot_dir, f'{draw_type}.png'))  

        draw_type = 'overall_metrics'
        plot = Classification_Plot(overall_json_path=os.path.join(jsons_dir, f'plot_highlighted_{draw_type}.json'))
        plot.draw('overall_metrics', metrics=['mean AUC', 'fwIoU'],  save_path=os.path.join(plot_dir, f'{draw_type}.png'))



    def _load_plotting_data(self, validation_collection_data):
        """
        Load and organize the data needed for plotting.

        :param validation_collection_data: A list of validation collection data dictionaries.

        :return: A dictionary containing per-image plotting data including ground truth, logits, metadata, and loss.
        """

        plotting_data = dict()
        plotting_data["per_image"] = self._craft_per_image_plotting_data(
            validation_collection_data
        )
        return plotting_data

    def _craft_per_image_plotting_data(self, validation_collection_data):
        """
        Craft the per-image plotting data from the validation collection.

        :param validation_collection_data: A list of validation collection data dictionaries.

        :return: A dictionary containing per-image data such as ground truth, logits, metadata, and loss for each image.
        """

        data = dict()
        validation_df = pd.DataFrame(validation_collection_data)
        # Ground truth dict
        ground_truth_dict = validation_df[["image_id", "ground_truth"]].values
        ground_truth_dict = dict(zip(ground_truth_dict[:, 0], ground_truth_dict[:, 1]))
        # Loss dict
        loss_dict = (
            validation_df[["image_id", "loss"]]
            .set_index("image_id")
            .to_dict()["loss"]
        )

        # Meta Data dict
        meta_data_dict = validation_df[["image_id", "meta_data"]].values
        meta_data_dict = dict(zip(meta_data_dict[:, 0], meta_data_dict[:, 1]))
        # Logits dict
        logits_dict = validation_df[["image_id", "logits"]].values
        logits_dict = dict(zip(logits_dict[:, 0], logits_dict[:, 1]))

        data["ground_truth"] = ground_truth_dict
        data["meta_data"] = meta_data_dict
        data["logits"] = logits_dict
        data["loss"] = loss_dict

        return data
    