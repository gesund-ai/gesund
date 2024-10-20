import argparse
import json
from gesund.validation import plotting_metrics, run_metrics
from gesund.utils.io_utils import read_json


def main():
    """
    Main function to plot validation metrics with optional filtering.

    This function sets up the command line argument parser, collects the required 
    parameters for validation, and invokes the run_metrics function to compute 
    metrics. It also plots the metrics using the plotting_metrics function, 
    with an option to filter based on metadata.

    :return: None

    :raises SystemExit: If the required arguments are not provided or if an error occurs 
                        during argument parsing.
    """
    parser = argparse.ArgumentParser(description="Plot validation metrics with optional filtering")
    
    parser.add_argument('--annotations_json_path', type=str, required=True, 
                        help='Path to annotations JSON file')
    parser.add_argument('--predictions', type=str, required=True, 
                        help='Path to predictions JSON file')
    parser.add_argument('--class_mappings', type=str, required=True, 
                        help='Path to class mappings JSON file')
    parser.add_argument('--problem_type', type=str, required=True, 
                        help='Type of problem (e.g., object_detection)')
    parser.add_argument('--format', type=str, required=True, 
                        help='Format of input data (e.g., gesund_custom_format)')
    parser.add_argument('--write_results_to_json', type=bool, default=False, 
                        help='Whether to write results to a JSON file')
    parser.add_argument('--filtering_meta', type=str, required=False, 
                        help='JSON string with metadata filtering (e.g., \'{"Age": [20, 40], "Gender": ["Female"]}\')')

    args = parser.parse_args()
    
    args_dict = {
        'annotations_json_path': args.annotations_json_path,
        'predictions': args.predictions,
        'class_mappings': args.class_mappings,
        'problem_type': args.problem_type,
        'format': args.format,
        'write_results_to_json': args.write_results_to_json
    }

    # Converting the filter_meta JSON string to a Python dictionary
    filtering_meta = json.loads(args.filtering_meta)
    
    metrics = run_metrics(args_dict)
    plotting_metrics(metrics)
    plotting_metrics(metrics, filtering_meta)

if __name__ == "__main__":
    main()
