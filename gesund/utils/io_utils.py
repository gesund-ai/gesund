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

    with open(file_path, 'r') as file:
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
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            print(f"Could not save metrics for {plot_name} because: {e}")
            
            
def format_metrics(metrics):
    """
    Format and print the overall metrics in a readable format.
    
    This function takes in the metrics data, formats it, and prints out the 
    highlighted overall metrics, including confidence intervals when applicable.
    It also prints a message indicating that all graphs and plot metrics have been saved.

    :param metrics: (dict) A dictionary containing the metrics, expected to have 
                    a 'plot_highlighted_overall_metrics' key with the metrics data.

    :return: None
    """

    print("\nOverall Highlighted Metrics:\n" + "-"*40)
    for metric, values in metrics['plot_highlighted_overall_metrics']['data'].items():
        print(f"{metric}:")
        for key, value in values.items():
            if isinstance(value, list):  # If it's a confidence interval
                value_str = f"{value[0]:.4f} to {value[1]:.4f}"
            else:
                value_str = f"{value:.4f}"
            print(f"    {key}: {value_str}")
        print("-"*40)
    print("All Graphs and Plots Metrics saved in JSONs.\n" + "-"*40)
