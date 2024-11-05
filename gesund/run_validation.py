import argparse
from gesund.core.validation import run_metrics

def main():
    """
    Main function to run validation metrics.

    This function sets up the command line argument parser, collects the required 
    parameters for validation, and invokes the run_metrics function from the 
    gesund to perform validation on the predictions against the 
    annotations.

    :return: None

    :raises SystemExit: If the required arguments are not provided or if an error occurs 
                        during argument parsing.
    """
    parser = argparse.ArgumentParser(description="Run validation metrics")
    
    parser.add_argument('--annotations_json_path', type=str, required=True, 
                        help='Path to annotations JSON file')
    parser.add_argument('--predictions', type=str, required=True, 
                        help='Path to predictions JSON file')
    parser.add_argument('--class_mappings', type=str, required=True, 
                        help='Path to class mappings JSON file')
    parser.add_argument('--meta_data', type=str, required=False, 
                        help='Path to metadata JSON file')
    parser.add_argument('--problem_type', type=str, required=True, 
                        help='Type of problem (e.g., object_detection)')
    parser.add_argument('--format', type=str, required=True, 
                        help='Format of input data (e.g., gesund_custom_format)')
    parser.add_argument('--write_results_to_json', type=bool, default=False, 
                        help='Whether to write results to a JSON file')

    args = parser.parse_args()
    
    args_dict = {
        'annotations_json_path': args.annotations_json_path,
        'predictions': args.predictions,
        'class_mappings': args.class_mappings,
        'problem_type': args.problem_type,
        'format': args.format,
        'write_results_to_json': args.write_results_to_json
    }

    result = run_metrics(args_dict)

if __name__ == "__main__":
    main()
