import argparse
import json
import bson
import os
from gesund.utils.io_utils import read_json, save_plot_metrics_as_json, format_metrics
from gesund.utils.yolo_converter import YOLOConverter
from gesund.utils.coco_converter import COCOConverter
from gesund.scripts.problem_type_factory import get_validation_creation

def main():
    parser = argparse.ArgumentParser(description='Run validation metrics calculation.')
    parser.add_argument('--annotations', type=str, required=True, help='Path to the JSON file with annotations.')
    parser.add_argument('--predictions', type=str, required=True, help='Path to the JSON file with predictions.')
    parser.add_argument('--class_mappings', type=str, required=True, help='Path to the JSON file with class mappings.')
    parser.add_argument('--problem_type', type=str, required=True,
                        choices=['classification', 'semantic_segmentation', 'instance_segmentation', 'object_detection'],
                        help='The type of problem.')
    parser.add_argument('--format', type=str, required=True, choices=['coco', 'yolo', 'gesund_custom_format'], 
                        help='Format of the input data (COCO, YOLO, Gesund_Custom_Format).')

    args = parser.parse_args()

    try:
        successful_batch_data = read_json(args.predictions)
        annotation_data = read_json(args.annotations)
        class_mappings = read_json(args.class_mappings)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading input files: {e}")
        return

    batch_job_id = str(bson.ObjectId())
    output_dir = os.path.join("outputs", batch_job_id)

    if args.format == 'coco':
        converter_annot = COCOConverter(annotations=annotation_data, problem_type=args.problem_type)
        converter_pred = COCOConverter(successful_batch_data=successful_batch_data, problem_type=args.problem_type)
        annotation_data = converter_annot.convert_annot_if_needed()
        successful_batch_data = converter_pred.convert_pred_if_needed()

    elif args.format == 'yolo':
        yolo_converter = YOLOConverter(annotations=annotation_data, successful_batch_data=successful_batch_data, problem_type=args.problem_type)
        annotation_data = yolo_converter.convert_annot_if_needed()
        successful_batch_data = yolo_converter.convert_pred_if_needed()

    elif args.format == 'gesund_custom_format':
        pass

    ValidationCreationClass = get_validation_creation(args.problem_type)
    validation = ValidationCreationClass(batch_job_id)

    try:
        validation_data = validation.create_validation_collection_data(successful_batch_data, annotation_data)
        metrics = validation.load(validation_data, class_mappings)    
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return

    format_metrics(metrics)
    save_plot_metrics_as_json(metrics, output_dir)

if __name__ == "__main__":
    main()