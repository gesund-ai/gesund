import json
import pandas as pd

from ._exceptions import DataLoadError


class DataLoader:
    def __init__(self, data_format: str):
        """
        A function to initialize the data

        :param data_format: string value indicating the data
        :type data_format: str

        :return: None
        """
        data_loaders = {"csv": self._csv_loader, "json": self._json_loader}
        self._loader_fxn = data_loaders[data_format]

    @staticmethod
    def _json_loader(src_path: str) -> list:
        """
        A function to load JSON file

        :param src_path: source path of the file
        :type src_path: str

        :return: loaded data
        :rtype: list
        """
        try:
            with open(src_path, "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(e)
            raise DataLoadError("Could not load JSON file !")

    @staticmethod
    def _csv_loader(src_path: str) -> pd.DataFrame:
        """
        A function to load csv file

        :param src_path: source path of the file
        :type src_path: str

        :return: loaded data frame
        :rtype: pd.DataFrame
        """
        return pd.DataFrame(src_path)

    def load(self, src_path: str) -> dict:
        """
        A function to load the data

        :param src_path: source path of the file
        :type src_path: str

        :return: None
        """
        return self._loader_fxn(src_path)
