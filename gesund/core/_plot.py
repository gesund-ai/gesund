
from ._exceptions import PlotError, MetricCalculationError
from gesund.validation import ValidationProblemTypeFactory
from ._schema import InputParams



class PlotData:

    def __init__(
            self, 
            metrics_result: dict, 
            user_params: InputParams
    ):
        """
        An intialization function for the plot data driver

        :param metrics_results: dictionary containing the result
        :type metrics_results: dict
        :param user_params: parameters received from the user
        :type user_params: object

        :return: None
        """

        self.metrics_results = metrics_result
        self.user_params = user_params
    
    def _get_metric_plotter(
            self, 
            metric_name: str, 
            metric_executor: str
        ):
        """
        A function to return plotting function specific to metric

        :param metric_name: name of the metric
        :type metric_name: str
        :param metric_executor: executor for  
        :type metric_executor: function

        :return: function to plot the metric
        :rtype: object
        """
        fxn_plot_map = {}
        if metric_name not in fxn_plot_map:
            raise ModuleNotFoundError(f"{metric_name} plotting function")
        
        return fxn_plot_map[metric_name](metric_executor)

    
    def _plot_all(self, metric_validation_executor):
        """
        A function to plot all the metrics

        :param metric_validation_executor: metric validation executor
        :type metric_validation_executor: object
        :return: None
        """
        try:
            metric_validation_executor.plot_metrics(self.metrics_results)
        except Exception as e:
            print(e)
            raise MetricCalculationError("Could not run the validation executor !")
        
        if self.user_params.store_plots:
            pass

    def write_plots(self):
        pass
    
    def apply_filters(self):
        pass

    def plot(self, metric_name: str = "all"):
        """
        A function to plot the data from the results

        :param metric_name: name of the metric to plot
        :type metric_name: str

        :return: None
        """
        _validation_class = ValidationProblemTypeFactory().get_problem_type_factory(
            self.user_params.problem_type)
        
        _metric_validation_executor = _validation_class(self.batch_job_id)

        if metric_name == "all":
            try:
                self._plot_all(_metric_validation_executor)
            except Exception as e:
                print(e)
                raise PlotError("Could not plot the metrics!")
        else:
            _metric_plotter = self._get_metric_plotter(
                metric_name, _metric_validation_executor)
            _metric_plotter.plot()
