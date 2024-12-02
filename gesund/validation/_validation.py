from typing import Optional, Union
import bson
import os
import json

from gesund.core._converters import ConverterFactory
from gesund.core._data_loaders import DataLoader
from gesund.core._schema import (
    UserInputParams,
    UserInputData,
    ResultDataClassification,
    ResultDataObjectDetection,
    ResultDataSegmentation,
)
from gesund.core._plot import PlotData
from gesund.core._exceptions import MetricCalculationError


class ValidationProblemTypeFactory:
    @classmethod
    def get_problem_type_factory(cls, problem_type: str):
        """
        Return the validation creation class based on the problem_type.

        :param problem_type: Type of problem (e.g., 'classification', 'object_detection').
        :type problem_type: str

        :return: Validation creation class corresponding to the problem type.
        :rtype: class
        """
        if problem_type == "classification":
            from gesund.core._metrics.classification.create_validation import (
                ValidationCreation,
            )

            return ValidationCreation
        elif problem_type == "semantic_segmentation":
            from gesund.core._metrics.semantic_segmentation.create_validation import (
                ValidationCreation,
            )

            return ValidationCreation
        elif problem_type == "object_detection":
            from gesund.core._metrics.object_detection.create_validation import (
                ValidationCreation,
            )

            return ValidationCreation
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")


class Validation:
    def __init__(
        self,
        annotations_path: str,
        predictions_path: str,
        class_mapping: Union[str, dict],
        problem_type: str,
        data_format: str,
        json_structure_type: str,
        plot_config: dict,
        metadata_path: Optional[str] = None,
        return_dict: Optional[bool] = False,
        display_plots: Optional[bool] = False,
        store_plots: Optional[bool] = False,
        run_validation_only: Optional[bool] = False,
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

        :param plot_config: The configuration in dictionary format required for plotting
        :type plot_config: dict

        :param metadata_path: Path to the metadata file (if available).
        :type metadata_path: str
        :optional metadata_path: true

        :param return_dict: If true then the return the result as dict
        :type return_dict: bool
        :optional return_dict: true

        :param display_plots: if true then the plots are displayed
        :type display_plots: bool
        :optional display_plots: true

        :param store_plots: if true then the plots are saved in local as png files
        :type store_plots: bool
        :optional store_plots: true

        :param run_validation_only: by default true the plots are run
        :type run_validation_only: bool
        :optional run_validation_only: true


        :return: None
        """
        # set up user parameters
        params = {
            "annotations_path": annotations_path,
            "predictions_path": predictions_path,
            "metadata_path": metadata_path,
            "problem_type": problem_type,
            "json_structure_type": json_structure_type,
            "return_dict": return_dict,
            "display_plots": display_plots,
            "store_plots": store_plots,
            "data_format": data_format,
            "class_mapping": class_mapping,
            "plot_config": plot_config,
            "run_validation_only": run_validation_only,
        }
        self.user_params = UserInputParams(**params)

        # set up and load all the required data
        self._load_data()

        # set up batch job id
        self.batch_job_id = str(bson.ObjectId())
        self.output_dir = f"outputs/{self.batch_job_id}"

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        self.problem_type_result_map = {
            "classification": ResultDataClassification,
            "object_detection": ResultDataObjectDetection,
            "semantic_segmentation": ResultDataSegmentation,
        }

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
        _validation_class = ValidationProblemTypeFactory().get_problem_type_factory(
            self.user_params.problem_type
        )

        _metric_validation_executor = _validation_class(self.batch_job_id)
        try:
            if self.data.was_converted:
                prediction = self.data.converted_prediction
                annotation = self.data.converted_annotation
            else:
                prediction = self.data.prediction
                annotation = self.data.annotation
            metadata = None
            if self.user_params.metadata_path:
                metadata = self.data.metadata
            validation_data = (
                _metric_validation_executor.create_validation_collection_data(
                    prediction, annotation, metadata
                )
            )

            metrics = _metric_validation_executor.load(
                validation_data, self.data.class_mapping
            )
            return metrics
        except Exception as e:
            print(e)
            raise MetricCalculationError("Error in calculating metrics!")

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

    def _plot_metrics(self, results: dict) -> None:
        """
        A function to plot the metrics

        :param results: a dictionary containing the validation metrics
        :type results: dict

        :return: None
        """
        plot_data_executor = PlotData(
            metrics_result=results,
            user_params=self.user_params,
            user_data=self.data,
            batch_job_id=self.batch_job_id,
            validation_problem_type_factory=ValidationProblemTypeFactory(),
        )
        plot_data_executor.plot()

    def _round_nested_dict(self, data: Union[dict, list, float, int], round_digits: int) -> Union[dict, list, float, int]:
        """
        Recursively round numbers in nested dictionaries and lists.
        
        :param data: Input data structure (dict, list, or numeric value)
        :param round_digits: Number of decimal places to round to
        :return: Data structure with rounded numbers
        """
        if isinstance(data, dict):
            return {k: self._round_nested_dict(v, round_digits) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._round_nested_dict(item, round_digits) for item in data]
        elif isinstance(data, float):
            return round(data, round_digits)
        return data

    def run(self, round_digits: Optional[int] = None) -> Union[None, ResultDataClassification]:
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

        # Round numbers if specified
        if round_digits is not None:
            results = self._round_nested_dict(results, round_digits)
            
        # format the results
        # results = self._format_results(results)

        # store the results
        self._save_json(results)

        # plot the metrics only if user requires
        if self.user_params.run_validation_only is False:
            self._plot_metrics(results)

        # return the results
        if self.user_params.return_dict:
            results = self.problem_type_result_map[self.user_params.problem_type](
                **results
            )
            return results
