import json
import bson
import os
from pathlib import Path
from gesund.utils.io_utils import read_json, save_plot_metrics_as_json, format_metrics
from gesund.utils.yolo_converter import YoloToGesund
from gesund.utils.coco_converter import COCOConverter
from gesund.problem_type_factory import get_validation_creation

def run_metrics(args):
    """
    Run validation metrics based on the passed arguments.

    This function processes prediction data, annotation data, and class mappings to calculate validation metrics.
    It supports multiple formats such as 'coco', 'yolo', and 'gesund_custom_format'. The results are optionally 
    saved as JSON files if specified in the arguments.

    :param args: A dictionary of arguments containing the following keys:
        - 'annotations_json_path' (str): Path to the JSON file containing the annotations.
        - 'predictions' (str): Path to the JSON file containing the predictions.
        - 'class_mappings' (str): Path to the JSON file containing class mappings.
        - 'problem_type' (str): Type of problem (e.g., 'classification', 'object_detection').
        - 'format' (str): Data format for the validation (e.g., 'coco', 'yolo', 'gesund_custom_format').
        - 'metadata' (str, optional): Path to the metadata file (if available).
        - 'write_results_to_json' (bool, optional): Whether to save the resulting metrics as JSON files.
        
    :return: A dictionary containing the validation metrics and additional information such as:
        - 'problem_type' (str): The type of problem.
        - 'batch_job_id' (str): A unique identifier for the batch job.
        - 'successful_batch_data' (dict): The processed predictions data.
        - 'annotation_data' (dict): The processed annotations data.
        - 'meta_data' (dict or None): Metadata if provided.
        - 'class_mappings' (dict): Class mappings used in the validation.
        - 'format' (str): Data format used in the validation.
        - 'output_dir' (str): Directory where output files are saved.
    """
    
    try:
        successful_batch_data = read_json(args['predictions'])
        annotation_data = read_json(args['annotations_json_path'])
        class_mappings = read_json(args['class_mappings'])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading input files: {e}")
        return None
    
    try:
        meta_data = read_json(args['metadata'])
    except:
        print("Metadata file not provided!")
        meta_data = None

    batch_job_id = str(bson.ObjectId())
    output_dir = os.path.join("outputs", batch_job_id)
    json_outputs_dir = os.path.join(output_dir, "plot_jsons")

    if args['format'] == 'coco':
        converter_annot = COCOConverter(annotations=annotation_data, problem_type=args['problem_type'])
        converter_pred = COCOConverter(successful_batch_data=successful_batch_data, problem_type=args['problem_type'])
        annotation_data = converter_annot.convert_annot_if_needed()
        successful_batch_data = converter_pred.convert_pred_if_needed()

    elif args['format'] == 'yolo':
        # problem_type = "semantic_segmentation" | "object_detection" | "instance_segmentation"
        yolo_converter = YoloToGesund(annotations=annotation_data, predictions=successful_batch_data)
        annotation_data = yolo_converter.run(problem_type=args["problem_type"], input_type="annotation")
        successful_batch_data = yolo_converter.run(problem_type=args["problem_type"], input_type="prediction")

    ValidationCreationClass = get_validation_creation(args['problem_type'])
    validation = ValidationCreationClass(batch_job_id)

    try:
        validation_data = validation.create_validation_collection_data(successful_batch_data, annotation_data, args['format'])
        metrics = validation.load(validation_data, class_mappings)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

    if args.get('write_results_to_json', False):
        save_plot_metrics_as_json(metrics, json_outputs_dir)

    metrics.update({
        "problem_type": args['problem_type'],
        "batch_job_id": batch_job_id,
        "successful_batch_data": successful_batch_data,
        "annotation_data": annotation_data,
        "meta_data": meta_data,
        "class_mappings": class_mappings,
        "format": args['format'],
        "output_dir": output_dir
    })

    return metrics

def plotting_metrics(metrics, args, filtering_meta=None):
    """
    Plot the validation metrics using the stored validation instance.

    This function generates plots for the validation metrics based on the provided metrics data. 
    It can optionally apply metadata filtering to create filtered plots. The results are saved to 
    output directories based on the batch job ID.

    :param metrics: A dictionary containing the validation metrics and related data, typically generated
                    by the `run_metrics` function. It should include keys like 'problem_type', 'batch_job_id',
                    'successful_batch_data', and 'output_dir'.
    :param filtering_meta: (optional) A dictionary of filtering criteria for applying metadata filtering
                           to the validation plots. It may contain:
        - 'metadata_file' (str): Path to the metadata file used for filtering.
        - 'filter_meta' (dict): Specific metadata filters to apply.
    
    :return: None. The function generates and saves plot images and JSON files with plot data.
    """

    ValidationCreationClass = get_validation_creation(metrics['problem_type'])
    validation = ValidationCreationClass(metrics["batch_job_id"])

    output_dir = metrics["output_dir"]
    json_outputs_dir = os.path.join(output_dir, "plot_jsons")

    if filtering_meta:
        meta_data = read_json(filtering_meta['metadata_file'])
        meta_filtered_validation_data = validation.create_validation_collection_data(
            metrics['successful_batch_data'], 
            metrics['annotation_data'], 
            metrics['format'], 
            meta_data
        )
        meta_filtered_metrics = validation.load(meta_filtered_validation_data, metrics['class_mappings'], filtering_meta['filter_meta'])
        filtered_jsons_outputs_dir = os.path.join(output_dir, "filtered_plot_jsons")
        Path(filtered_jsons_outputs_dir).mkdir(parents=True, exist_ok=True)
        save_plot_metrics_as_json(meta_filtered_metrics, filtered_jsons_outputs_dir)
        
        filtered_plot_outputs_dir = os.path.join(output_dir, "filtered_plots")
        Path(filtered_plot_outputs_dir).mkdir(parents=True, exist_ok=True)
        validation.plot_metrics(meta_filtered_metrics, filtered_jsons_outputs_dir, filtered_plot_outputs_dir, args['plot_configs'])

    else:
        plot_outputs_dir = os.path.join(output_dir, "plots")
        Path(plot_outputs_dir).mkdir(parents=True, exist_ok=True)
        validation.plot_metrics(metrics, json_outputs_dir, plot_outputs_dir, args['plot_configs'])