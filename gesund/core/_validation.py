from typing import Optional

from ._data_loaders import DataLoader
from ._schema import InputParams, Data


class Validation:
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
        params = {
            "annotations_path": annotations_path,
            "predictions_path": predictions_path,
            "metadata_path": metadata_path,
            "problem_type": problem_type,
            "json_structure_type": json_structure_type, 
            "return_dict": return_dict,
            "store_json": store_json,
            "display_plots": display_plots,
            "store_plots": store_plots
        }

        self.params = InputParams(**params)
        self.data = Data(
            **self.load_data(self.params))
        self.data_loader = DataLoader(data_format)
    
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



