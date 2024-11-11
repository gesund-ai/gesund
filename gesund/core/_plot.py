from typing import Union


from ._exceptions import PlotError, MetricCalculationError
from gesund.validation import ValidationProblemTypeFactory
from ._schema import InputParams
from gesund.core._metrics.classification.classification_metric_plot import Classification_Plot


class CommonPlots:
    def _class_distribution(
            self, 
            metrics: Union[dict, list], 
            threshold: float,
            save_path: Union[str, None]):
        """
        A function to plot class distribution

        :param metrics: The metrics to be plotted
        :type metrics: Union[dict, list]
        :param threshold: the value to be applied as threshold
        :type threshold: float
        :param save_path: The path to be saved the plot in
        :type save_path: The path to be saved

        :return: None 
        """
        self.cls_driver._plot_class_distributions(metrics, threshold, save_path)
        
    def _blind_spot(self,
                    metrics: Union[dict, list],
                    class_types: list,
                    save_path: Union[str, None]):
        """
        A function to plot blind spot metrics

        :param metrics: The metrics to be plotted
        :type metrics: Union[dict, list]
        :param class_types: List of class types to include
        :type class_types: list 
        :param save_path: The path to be saved the plot in
        :type save_path: Union[str, None]

        :return: None 
        """
        self.cls_driver._plot_blind_spot(metrics, class_types, save_path)
           


class ClassificationPlots(CommonPlots):
    def __init__(self):
        self.cls_driver = Classification_Plot()


class ObjectDetectionPlots(CommonPlots):
    pass

class SegmentationPlots(CommonPlots):
    pass


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
        self.plot_save_dir = "outputs/plots"

        # set up plotters
        self.classification_plotter = ClassificationPlots()
        self.object_detection_plotter = ObjectDetectionPlots()
        self.segmentation_plotter = SegmentationPlots()
    
    def _plot_single_metric(
            self, 
            metric_name: str, 
        ):
        """
        A function to return plotting function specific to metric

        :param metric_name: name of the metric
        :type metric_name: str

        :return: None
        """
        fxn_plot_map = {
            "classification": {
                "class_distribution": self.classification_plotter._class_distribution,
                "blind_spot": self.classification_plotter._blind_spot
            }
            
        }
        if metric_name not in fxn_plot_map:
            raise ModuleNotFoundError(f"{metric_name} plotting function")
        
        _plotter_fxn = fxn_plot_map[self.user_params.problem_type][metric_name]
        _plotter_fxn(
            metrics=self.metrics_results,
            threshold=self.user_params.threshold,
            save_path=self.plot_save_dir if self.user_params.save_plots else None
        )
        
    
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

    def plot(self, metric_name: str = "all", threshold: float = 0.0):
        """
        A function to plot the data from the results

        :param metric_name: name of the metric to plot
        :type metric_name: str
        :param threshold: float value indicating the threshold
        :type threshold: float

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
            self._plot_single_metric(metric_name, threshold)
