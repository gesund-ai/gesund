from typing import Optional

from ._converters import ConverterFactory
from ._data_loaders import DataLoader
from ._schema import InputParams, Data


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
        data = self.load_data(self.params, self.data_loader)
        self.data = Data(**data)

        # setup data converter
        self.data_converter = ConverterFactory().get_converter(json_structure_type)

        # set up validation handlers
        
    
    @staticmethod
    def load_data(
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
    
    def convert_data(self):
        """
        A function to convert the data from the respective structure to the gesund format

        :param: None

        :return: None
        """
        # convert annotation data
        self.data.converted_annotation = self.data_converter.convert_annotation(
            self.data.annotation)

        # convert prediction data
        self.data.converted_prediction = self.data_converter.convert_prediction(
            self.data.prediction)

    
    def run(self):
        """
        A function to run the validation pipeline

        :param: 
        :type:

        :return:
        :rtype: 
        """
        results = {}

        # run the converters
        self.convert_data()

        # run the validation

        # store the results
        if self.user_params.store_json:
            self.save_json()
        
        if self.user_params.store_plots:
            self.save_plots()

        # return the results
        if self.user_params.return_dict:
            return results


