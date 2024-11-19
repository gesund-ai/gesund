from typing import Union

from .coco_converter import COCOToGesund
from .yolo_converter import YoloToGesund


class ConverterFactory:
    def get_converter(
        self, json_structure_type: str
    ) -> Union[COCOToGesund, YoloToGesund]:
        """
        A function to get the converter to transform the data to gesund format given the current json
        structure type

        :param json_structure_type: current type of the data
        :type json_structure_type: str

        :return: object of the converter
        :rtype: Union[CocoToGesund, YoloToGesund]
        """
        if json_structure_type == "coco":
            return COCOToGesund()
        elif json_structure_type == "yolo":
            return YoloToGesund()
