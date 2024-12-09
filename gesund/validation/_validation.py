import os
import json
import bson
from typing import Union, Optional

from gesund.core.schema import UserInputParams, UserInputData
from gesund.core._exceptions import MetricCalculationError
from gesund.core._data_loaders import DataLoader
from gesund.core._converters import ConverterFactory
from ._result import ValidationResult
from gesund.core._managers.metric_manager import metric_manager
from gesund.core._managers.plot_manager import plot_manager


class Validation:
    def __init__(
        self,
        annotations_path: str,
        predictions_path: str,
        class_mapping: Union[str, dict],
        problem_type: str,
        data_format: str,
        json_structure_type: str,
        plot_config: str,
        metadata_path: Optional[str] = None,
    ):
        """
        Initialization function to handle the validation pipeline

        :param annotations_path: Path to the JSON file containing the annotations data.
        :type annotations_path: str

        :param predictions_path: Path to the JSON file containing the predictions data.
        :type predictions_path: str

        :param class_mappings: Path to the JSON file containing class mappings or a dictionary file.
        :type class_mappings: Union[str, dict]

        :param problem_type: Type of problem (e.g., 'classification', 'object_detection').
        :type problem_type: str

        :param json_structure_type: Data format for the validation (e.g., 'coco', 'yolo', 'gesund').
        :type json_structure_type: str

        :param plot_config: Config for the plotting
        :type plot_config: dict

        :param metadata_path: Path to the metadata file (if available).
        :type metadata_path: str
        :optional metadata_path: true

        :return: None
        """
        # set up user parameters
        params = {
            "annotations_path": annotations_path,
            "predictions_path": predictions_path,
            "metadata_path": metadata_path,
            "problem_type": problem_type,
            "json_structure_type": json_structure_type,
            "data_format": data_format,
            "class_mapping": class_mapping,
            "plot_config": plot_config,
        }
        self.user_params = UserInputParams(**params)

        # set up and load all the required data
        self._load_data()

        # set up batch job id
        self.batch_job_id = str(bson.ObjectId())
        self.output_dir = f"outputs/{self.batch_job_id}"

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)

    def _load_data(self) -> dict:
        """
        A Function to load the JSON files

        :return: None
        """
        # Load data
        # set up source data for processing
        data_loader = DataLoader(self.user_params.data_format)
        data = {
            "prediction": data_loader.load(self.user_params.predictions_path),
            "annotation": data_loader.load(self.user_params.annotations_path),
        }
        if isinstance(self.user_params.class_mapping, str):
            data["class_mapping"] = data_loader.load(self.user_params.class_mapping)
        else:
            data["class_mapping"] = self.user_params.class_mapping

        if self.user_params.metadata_path:
            data["metadata"] = data_loader.load(self.user_params.metadata_path)

        # run conversion
        if self.user_params.json_structure_type != "gesund":
            self._convert_data(data)

        self.data = UserInputData(**data)

    def _convert_data(self, data):
        """
        A function to convert the data from the respective structure to the gesund format

        :param data: dictionary containing the data
        :type data: dict

        :return: data dictionary
        :rtype: dict
        """
        # setup data converter
        data_converter = ConverterFactory().get_converter(
            self.user_params.json_structure_type
        )

        # run the converters
        (
            data["converted_annotation"],
            data["converted_prediction"],
        ) = data_converter.convert(
            annotation=data["annotation"],
            prediction=data["prediction"],
            problem_type=self.user_params.problem_type,
        )
        data["was_converted"] = True
        return data

    def _run_validation(self) -> dict:
        """
        A function to run the validation

        :param: None

        :return: result dictionary
        :rtype: dict
        """
        if self.data.was_converted:
            prediction = self.data.converted_prediction
            annotation = self.data.converted_annotation
        else:
            prediction = self.data.prediction
            annotation = self.data.annotation
        metadata = None
        if self.user_params.metadata_path:
            metadata = self.data.metadata

        results = {}
        try:
            for metric_name in metric_manager.get_names(
                problem_type=self.user_params.problem_type
            ):
                _metric_executor = metric_manager[
                    f"{self.user_params.problem_type}.{metric_name}"
                ]
                _result = _metric_executor(
                    data={
                        "prediction": prediction,
                        "ground_truth": annotation,
                        "metadata": metadata,
                        "class_mapping": self.data.class_mapping,
                    },
                    problem_type=self.user_params.problem_type,
                )
                results[metric_name] = _result
        except Exception as e:
            print(e)
            raise MetricCalculationError("Error in calculating metrics!")

        return results

    def _save_json(self, results) -> None:
        """
        A function to save the validation results

        :param results: Dictionary containing the results
        :type results: dict

        :return: None
        """
        for plot_name, metrics in results.items():
            output_file = os.path.join(self.output_dir, f"{plot_name}.json")
            try:
                with open(output_file, "w") as f:
                    json.dump(metrics, f, indent=4)
            except Exception as e:
                print(f"Could not save metrics for {plot_name} because: {e}")

    @staticmethod
    def format_metrics(metrics: dict) -> dict:
        """
        Format and print the overall metrics in a readable format.

        This function takes in the metrics data, formats it, and prints out the
        highlighted overall metrics, including confidence intervals when applicable.
        It also prints a message indicating that all graphs and plot metrics have been saved.

        :param metrics: (dict) A dictionary containing the metrics, expected to have
                        a 'plot_highlighted_overall_metrics' key with the metrics data.

        :return: None
        """

        print("\nOverall Highlighted Metrics:\n" + "-" * 40)
        for metric, values in metrics["plot_highlighted_overall_metrics"][
            "data"
        ].items():
            print(f"{metric}:")
            for key, value in values.items():
                if isinstance(value, list):  # If it's a confidence interval
                    value_str = f"{value[0]:.4f} to {value[1]:.4f}"
                else:
                    value_str = f"{value:.4f}"
                print(f"    {key}: {value_str}")
            print("-" * 40)
        print("All Graphs and Plots Metrics saved in JSONs.\n" + "-" * 40)

    def run(self) -> ValidationResult:
        """
        A function to run the validation pipeline

        :param: None
        :type:

        :return: None
        :rtype:
        """
        results = {}

        # run the validation
        results = self._run_validation()

        # return the results
        results = ValidationResult(
            data=self.data, input_params=self.user_params, result=results
        )

        return results
