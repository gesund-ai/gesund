from typing import Union, Optional

from gesund.core.schema import UserInputParams, UserInputData
from gesund.core._managers.metric_manager import metric_manager
from gesund.core._managers.plot_manager import plot_manager


class ValidationResult:
    def __init__(
        self,
        data: UserInputData,
        input_params: UserInputParams,
        result: Union[dict, list],
    ) -> None:
        """
        A function to initialize the resultant data

        :param data: the data loaded by the validation class
        :type data:
        :param input_params:
        :type input_params:
        :param result:
        :type result:

        :return: None
        """
        self.data = data
        self.user_params = input_params
        self.result = result

    def save(self, metric_name: str = "all", format: str = "json") -> str:
        """
        A function to save the metric in json format

        :param metric_name: name of the metric to save the results
        :type metric_name: str
        :param format: data format for the result to save in
        :type format: str

        :return: location of the json file stored
        :rtype: str
        """
        pass

    def plot(
        self, metric_name: str = "all", save_plot: bool = False
    ) -> Union[str, None]:
        """
        A functon to plot the given metric

        :param metric_name: name of the metric to be plotted
        :type metric_name: str
        :param save_plot: True if the plot is to be saved
        :type save_plot: bool
        :param file_name: Plot file name
        :type file_name: str

        :return: path of the plot if saved
        :rtype: str
        """
        if metric_name == "all":
            for _metric in plot_manager.get_names(
                problem_type=self.user_params.problem_type
            ):
                _plot_executor = plot_manager[
                    f"{self.user_params.problem_type}.{_metric}"
                ]
                _plot_executor(results=self.result[_metric], save_plot=save_plot)
        else:
            _plot_executor = plot_manager[
                f"{self.user_params.problem_type}.{metric_name}"
            ]
            _plot_executor(results=self.result[metric_name], save_plot=save_plot)
