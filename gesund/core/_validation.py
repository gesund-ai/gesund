from typing import Optional, Union
import bson

from ._converters import ConverterFactory
from ._data_loaders import DataLoader
from ._schema import InputParams, Data, ResultData
from ._plot import PlotData
from ._exceptions import MetricCalculationError


class ValidationProblemTypeFactory:
    
    def get_problem_type_factory(self, problem_type:str):
        """
        A function to return the class as per the problem_type

        :param problem_type: problem type identifying the validation
        :type problem_type: str

        :return: class as per the problem type
        :rtype: class
        """
        if problem_type == "classification":
            from ._metrics.classification.create_validation import ValidationCreation
            return ValidationCreation
        elif problem_type == "semantic_segmentation":
            from ._metrics.semantic_segmentation.create_validation import ValidationCreation
            return ValidationCreation
        elif problem_type == "object_detection":
            from ._metrics.object_detection.create_validation import ValidationCreation
            return ValidationCreation
        else:
            raise ValueError(f"Unknow problem type : {problem_type}")


class Validation:

    ALLOWED_VALUES = {
        "problem_type": ["classification", "object_detection", "semantic_segmentation"],
        "json_structure_type": ["gesund", "coco", "yolo"],
        "data_format": ["json"]
    }

    def __init__(
            self,
            annotations_path: str,
            predictions_path: str,
            problem_type: str,
            data_format: str,
            json_structure_type: str,
            metadata_path: Optional[str] = None,
            return_dict: Optional[bool] = False,
            store_json: Optional[bool] = False,
            display_plots: Optional[bool] = False,
            store_plots: Optional[bool] = False
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
        
        :param metadata_path: Path to the metadata file (if available).
        :type metadata_path: str
        :optional metadata_path: true
        
        :param return_dict: If true then the return the result as dict 
        :type return_dict: bool
        :optional return_dict: true
        
        :param store_json: if true then the result is written in local as JSON file
        :type store_json: bool
        :optional store_json: true

        :param display_plots: if true then the plots are displayed
        :type display_plots: bool
        :optional display_plots: true

        :param store_plots: if true then the plots are saved in local as png files
        :type store_plots: bool
        :optional store_plots: true
        

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
            "store_json": store_json,
            "display_plots": display_plots,
            "store_plots": store_plots,
            "allowed_values": self.ALLOWED_VALUES
        }
        self.user_params = InputParams(**params)

        # set up source data for processing 
        self.data_loader = DataLoader(data_format)
        data = self._load_data(self.params, self.data_loader)
        self.data = Data(**data)

        # set up batch job id
        self.batch_job_id = str(bson.ObjectId())
        self.output_dir = f"outputs/{self.batch_job_id}"

    
    @staticmethod
    def _load_data(
        user_params: InputParams, data_loader: DataLoader) -> dict:
        """
        A Function to load the JSON files

        :param user_params: Object data containing the input parameters from users
        :type user_params: pydantic.BaseModel

        :return: dictionary containing the loaded JSON files
        :rtype: dict
        """
        data = {
            "prediction": data_loader.load(user_params.predictions_path),
            "annotation": data_loader.load(user_params.annotations_path)
        }
        if user_params.metadata_path:
            data["metadata"] = data_loader.load(user_params.metadata_path)
        
        return data
    
    def _convert_data(self):
        """
        A function to convert the data from the respective structure to the gesund format

        :param: None

        :return: None
        """
        # setup data converter
        data_converter = ConverterFactory().get_converter(self.user_params.json_structure_type)

        # set up validation handlers
        self.data.converted_annotation, self.data.converted_prediction = data_converter.convert(
            annotation=self.data.annotation,
            prediction=self.data.prediction,
            problem_type=self.user_params.problem_type
        )

        self.data.was_converted = True
    
    def _run_validation(self) -> dict:
        """
        A function to run the validation 

        :param: None

        :return: result dictionary
        :rtype: dict
        """
        _validation_class = ValidationProblemTypeFactory().get_problem_type_factory(
            self.user_params.problem_type)
        
        _metric_validation_executor = _validation_class(self.batch_job_id)

        try:
            if self.data.was_converted:
                prediction = self.data.converted_prediction
                annotation = self.data.converted_annotation
            else:
                prediction = self.data.prediction
                annotation = self.data.annotation
            validation_data = _metric_validation_executor.create_validation_collection_data(
                prediction, annotation, self.user_params.json_structure_type)
            
            metrics  = _metric_validation_executor.load(
                validation_data, self.data.class_mapping
            )
            return metrics
        except Exception as e:
            print(e)
            raise MetricCalculationError("Error in calculating metrics!")
        
    
    def _plot_metrics(self, results: dict) -> None:
        """
        A function to plot the metrics

        :param results: a dictionary containing the validation metrics
        :type results: dict

        :return: None
        """
        plot_data_executor = PlotData(
            metrics_result=results,
            save_plots=self.user_params.store_plots,
            user_params=self.user_params
        )
        plot_data_executor.plot()


    def run(self) -> Union[None, ResultData]:
        """
        A function to run the validation pipeline

        :param: 
        :type:

        :return:
        :rtype: 
        """
        results = {}

        # run the converters
        self._convert_data()

        # run the validation
        results = self._run_validation()

        # store the results
        if self.user_params.store_json:
            self._save_json(results)
        

        # plot the metrics
        self._plot_metrics(results)
        
        # return the results
        if self.user_params.return_dict:
            results = ResultData(**results)
            return results


