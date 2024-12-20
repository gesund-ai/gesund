from typing import Union, Optional, Callable

from gesund.core.schema import UserInputParams, UserInputData
from gesund.core._managers.plot_manager import plot_manager


class ValidationResult:
    def __init__(
        self,
        data: UserInputData,
        input_params: UserInputParams,
        result: Union[dict, list],
        cohort_args: Optional[dict] = {},
        plot_args: Optional[dict] = {},
    ) -> None:
        """
        A function to initialize the resultant data

        :param data: the data loaded by the validation class
        :type data: UserInputData
        :param input_params: the user input provided by the user inputs
        :type input_params: UserInputParams
        :param result: result from validation execution
        :type result: Union[dict, list]
        :param cohort_args: user arguments for cohort creation
        :type cohort_args: dict
        :param  plot_args: user arguments for plotting
        :type plot_args: dict

        :return: None
        """
        self.data = data
        self.user_params = input_params
        self.result = result
        self.cohort_args = cohort_args
        self.plot_args = plot_args

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

    def _run_plot_executor(
        self,
        plot_executor: Callable,
        metric_name: str,
        save_plot: bool,
        cohort_id: Optional[int] = None,
        file_name: Optional[str] = None,
    ) -> None:
        """
        A function to run the plot executor

        :param plot_executor: The registered function required for executing the function
        :type plot_executor: Callable
        :param metric_name: name of the metric to be plotted
        :type metric_name: str
        :param save_plot: boolean value of the metric to be saved
        :type save_plot: bool
        :param cohort_id: the id of the cohort if applicable
        :type cohort_id: int
        :param file_name: the name of the file to use for saving
        :type file_name: str

        """
        # cohort wise plotting is only possible when the cohort map is populated
        # if it is not populated then it is considered as a whole data
        if self.cohort_args["cohort_map"]:
            if cohort_id:
                result = self.result[cohort_id][metric_name]
                plot_executor(
                    results=result,
                    save_plot=save_plot,
                    cohort_id=cohort_id,
                    file_name=file_name,
                )
            else:
                for _cohort_id in self.cohort_args["cohort_map"]:
                    result = self.result[_cohort_id][metric_name]
                    plot_executor(
                        results=result,
                        save_plot=save_plot,
                        cohort_id=_cohort_id,
                        file_name=file_name,
                    )
        else:
            result = self.result[metric_name]
            plot_executor(results=result, save_plot=save_plot, file_name=file_name)

    def plot(
        self,
        metric_name: str = "all",
        save_plot: bool = False,
        file_name: Optional[str] = None,
        cohort_id: Optional[int] = None,
    ) -> Union[str, None]:
        """
        A functon to plot the given metric

        :param metric_name: name of the metric to be plotted
        :type metric_name: str
        :param save_plot: True if the plot is to be saved
        :type save_plot: bool
        :param file_name: Plot file name
        :type file_name: str
        :param cohort_id: id of the data cohort if applicable
        :type cohort_id: int

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
                self._run_plot_executor(
                    _plot_executor, _metric, save_plot, cohort_id, file_name
                )
        else:
            _plot_executor = plot_manager[
                f"{self.user_params.problem_type}.{metric_name}"
            ]
            self._run_plot_executor(
                _plot_executor, metric_name, save_plot, cohort_id, file_name
            )
