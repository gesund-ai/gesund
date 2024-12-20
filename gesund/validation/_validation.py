import os
import json
import bson
from typing import Union, Optional

import pandas as pd

from gesund.core.schema import UserInputParams, UserInputData
from gesund.core._exceptions import MetricCalculationError
from gesund.core._data_loaders import DataLoader
from gesund.core._converters import ConverterFactory
from ._result import ValidationResult
from gesund.core._managers.metric_manager import metric_manager


def categorize_age(age):
    if age < 18:
        return "Child"
    elif 18 <= age < 30:
        return "Young Adult"
    elif 30 <= age < 60:
        return "Adult"
    else:
        return "Senior"


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
        cohort_args: Optional[dict] = {},
        plot_args: Optional[dict] = {},
        metric_args: Optional[dict] = {},
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
        :param cohort_args: arguments supplied for cohort analysis
        :type cohort_args: dict
        :param plot_args: arguments supplied for cohort analysis
        :type plot_args: dict
        :param metric_args: arguments supplied for validation metrics
        :type metric_args: dict

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

        # cohort parameters
        self.cohort_params = {
            "max_num_cohort": 10,
            "min_cohort_size": 0,
            "cohort_map": {},
        }
        if cohort_args:
            self.cohort_params.update(cohort_args)

        self.plot_args = plot_args
        self.metric_args = metric_args

        self.debug_mode = False

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

    def apply_metadata(self, prediction: dict, annotation: dict) -> dict:
        """
        A function to create cohorts as per the metadata

        :param prediction: structure containing the prediction data
        :type prediction: dict
        :param annotation: structure containing the annotation data
        :type annotation: dict

        :return: Filtered dataset
        :rtype: dict
        """
        # convert the metadata into pandas dataframe for better access
        df: pd.DataFrame = pd.DataFrame.from_records(self.data.metadata)
        cohorts_data = {}
        cohort_id_map = {}

        # collect only the metadata columns
        metadata_columns = df.columns.tolist()
        metadata_columns.remove("image_id")
        lower_case = {i: i.lower() for i in metadata_columns}
        df = df.rename(columns=lower_case)

        # categorize age in metadata
        if "age" in list(lower_case.values()):
            df["age"] = df["age"].apply(categorize_age)

        # loop over group by to form cohorts with unique metadata
        cohort_id = 0
        for grp, subset_data in df.groupby(list(lower_case.values())):
            grp_str = ",".join([str(i) for i in grp])
            cohort_id = cohort_id + 1
            cohort_id_map[cohort_id] = {"group": grp_str, "size": subset_data.shape[0]}

            # it seems that
            if subset_data.shape[0] < self.cohort_params["min_cohort_size"]:
                print(
                    f"Warning - grp excluded - {grp_str} cohort size < {self.cohort_params['min_cohort_size']}"
                )
            else:
                image_ids = set(subset_data["image_id"].to_list())
                filtered_data = {
                    "prediction": {
                        i: prediction[i] for i in prediction if i in image_ids
                    },
                    "ground_truth": {
                        i: annotation[i] for i in annotation if i in image_ids
                    },
                    "metadata": subset_data,
                    "class_mapping": self.data.class_mapping,
                }
                cohorts_data[cohort_id] = filtered_data

        self.cohort_params["cohort_map"] = cohort_id_map
        return cohorts_data

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

    def _run_validation(self, data: dict) -> dict:
        """
        A function to run the validation

        :param: None

        :return: result dictionary
        :rtype: dict
        """
        results = {}
        try:
            for metric_name in metric_manager.get_names(
                problem_type=self.user_params.problem_type
            ):
                key_name = f"{self.user_params.problem_type}.{metric_name}"
                print("Running ", key_name, "." * 5)
                _metric_executor = metric_manager[key_name]
                _result = _metric_executor(
                    data=data, problem_type=self.user_params.problem_type
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
        cohorts = None

        # the data is assigned to local veriables to later pass on to functions as parameters
        # eventhough the data is stored in class attribute, for better code clarity
        if self.data.was_converted:
            prediction = self.data.converted_prediction
            annotation = self.data.converted_annotation
        else:
            prediction = self.data.prediction
            annotation = self.data.annotation

        # if the metadata is made available then the data is divided into cohorts as per the metadata
        if self.user_params.metadata_path:
            cohorts = self.apply_metadata(prediction, annotation)

        # run the validation
        if cohorts:
            for _cohort_id in cohorts:
                data = cohorts[_cohort_id]
                data["metric_args"] = self.metric_args
                results[_cohort_id] = self._run_validation(data)
        else:
            results = self._run_validation(
                data={
                    "prediction": prediction,
                    "ground_truth": annotation,
                    "metadata": self.data.metadata,
                    "class_mapping": self.data.class_mapping,
                    "metric_args": self.metric_args,
                }
            )

        # return the results
        return ValidationResult(
            data=self.data,
            input_params=self.user_params,
            result=results,
            plot_args=self.plot_args,
            cohort_args=self.cohort_params,
        )
