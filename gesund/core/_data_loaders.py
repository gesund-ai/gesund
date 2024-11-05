import pandas as pd


class DataLoader:
    def __init__(self):
        """
        A initialisation function

        :return: None
        """
        pass

    @staticmethod
    def _json_loader(src_path: str) -> list:
        """
        A function to load JSON file

        :param src_path: source path of the file
        :type src_path: str

        :return: loaded data
        :rtype: list
        """
        pass

    @staticmethod
    def _csv_loader(src_path: str) -> pd.DataFrame:
        """
        A function to load csv file

        :param src_path: source path of the file
        :type src_path: str

        :return: loaded data frame
        :rtype: pd.DataFrame
        """
        pass

    def load(self, src_path: str, data_format: str) -> dict:
        """
        A function to load the data

        :param src_path: source path of the file
        :type src_path: str
        :param data_format: type of the data format
        :type data_format: str

        :return: None
        """
        data_loaders = {
            "csv_loader": self._csv_loader,
            "json_loader": self._json_loader
        }
        return data_loaders[data_format](src_path)
