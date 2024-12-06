from typing import Union, Callable

from gesund.core._managers.metric_manager import metric_manager
from gesund.core._managers.plot_manager import plot_manager


class Classification:
    def _validate_data(self, data: dict) -> bool:
        """
        A function to validate the data that is required for metric calculation and plotting.

        :param data: The input data required for calculation,  {"prediction":, "ground_truth": , "metadata":}
        :type data: dict

        :return: status if the data is valid
        :rtype: bool
        """
        pass

    def apply_metadata(self, data: dict, metadata: dict) -> dict:
        """
        A function to apply the metadata on the data for metric calculation and plotting.

        :param data: the input data required for calculation, {"prediction":, "ground_truth": , "metadata":}
        :type data: dict

        :return: filtered dataset
        :rtype: dict
        """
        pass

    def calculate(self, data: dict) -> dict:
        """
        A function to calculate the AUC metric for given dataset

        :param data: The input data required for calculation and plotting  {"prediction":, "ground_truth": , "metadata":}
        :type data: dict

        :return: calculated metric
        :rtype: dict
        """
        # validate the data

        # format the data

        # apply metadata if given

        # run the calculation logic
        pass


class SemanticSegmentation(Classification):
    pass


class ObjectDetection:
    pass


class PlotAuc:
    def __init__(self, data: Union[dict, list]):
        self.data = data

    def _validate_data(self):
        """
        A function to validate the data required for plotting the AUC.
        """
        pass

    def save(self) -> str:
        """
        A function to save the plot
        """
        pass

    def plot(self):
        """
        Logic to plot the AUC
        """

        # validate the data

        # run the plotting logic
        pass


problem_type_map = {
    "classification": Classification,
    "semantic_segmentation": SemanticSegmentation,
    "object_detection": ObjectDetection,
}


@metric_manager.register("classification.auc")
def calculate_auc(data: dict, problem_type: str):
    """
    A wrapper function

    :param data: dictionary of data: {"prediction": , "ground_truth": }
    :type data: Union[str, list]
    :param problem_type: type of the problem
    :type problem_type: str

    :return : dict or list of calculated results
    :rtype: Union[dict, list]
    """
    _metric_calculator = problem_type_map[problem_type]()
    result = _metric_calculator.calculate(data)
    return result


@plot_manager.register("classification.auc")
def plot_auc(results: Union[dict, list], save_plot: bool) -> Union[str, None]:
    """
    A wrapper function

    :param results: list or dictionary of the results
    :type results: Union[dict, list]
    :param save_plot: boolean value if save plot
    :type save_plot: bool

    :return: None or string of the path
    :rtype: Union[str, None]
    """
    plotter = PlotAuc(data=results)
    plotter.plot()
    if save_plot:
        return plotter.save()
