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
           
    def _class_performance_by_threshold(
            self,
            metrics: Union[dict, list],
            graph_type: str,
            threshold: float, 
            save_path: Union[str, None]):
        """
        A function to plot class performance by threshold

        :param metrics: The metrics to be plotted
        :type metrics: Union[dict, list]
        :param graph_type: The type of graph to plot (e.g., 'graph_1')
        :type graph_type: str
        :param threshold: the value to be applied as threshold
        :type threshold: float
        :param save_path: The path to be saved the plot in
        :type save_path: Union[str, None]

        :return: None
        """
        self.cls_driver._plot_class_performance_by_threshold(graph_type, metrics, threshold, save_path)

    def _roc_statistics(
            self,
            roc_class: list,
            save_path: Union[str, None]):
        """
        A function to plot ROC statistics

        :param metrics: The metrics to be plotted
        :type metrics: Union[dict, list]
        :param roc_class: List of classes to plot ROC curves for
        :type roc_class: list
        :param save_path: The path to be saved the plot in
        :type save_path: Union[str, None]

        :return: None
        """
        self.cls_driver._plot_roc_statistics(roc_class, save_path)

    def _precision_recall_statistics(
            self,
            pr_class: list,
            save_path: Union[str, None]):
        """
        A function to plot precision recall statistics

        :param metrics: The metrics to be plotted
        :type metrics: Union[dict, list]
        :param pr_class: List of classes to plot precision recall curves for
        :type pr_class: list
        :param save_path: The path to be saved the plot in
        :type save_path: Union[str, None]

        :return: None
        """
        self.cls_driver._plot_precision_recall_statistics(pr_class, save_path)

    def _confidence_histogram(
            self,
            confidence_histogram_args: list,
            save_path: Union[str, None]):
        """
        A function to plot confidence histogram

        :param confidence_histogram_args: List of arguments for confidence histogram
        :type confidence_histogram_args: list
        :param save_path: The path to be saved the plot in
        :type save_path: Union[str, None]

        :return: None
        """
        self.cls_driver._plot_confidence_histogram(confidence_histogram_args, save_path)
        
    def _overall_metrics(
            self,
            overall_metrics_args: list,
            save_path: Union[str, None]):
        """
        A function to plot overall metrics

        :param overall_metrics_args: List of arguments for overall metrics
        :type overall_metrics_args: list
        :param save_path: The path to be saved the plot in
        :type save_path: Union[str, None]

        :return: None
        """
        self.cls_driver._plot_overall_metrics(overall_metrics_args, save_path)

    def _confusion_matrix(
            self,
            save_path: Union[str, None]):
        """
        A function to plot confusion matrix

        :param save_path: The path to be saved the plot in
        :type save_path: Union[str, None]

        :return: None
        """
        self.cls_driver._plot_confusion_matrix(save_path)

    def _prediction_dataset_distribution(
            self,
            save_path: Union[str, None]):
        """
        A function to plot prediction dataset distribution

        :param save_path: The path to be saved the plot in
        :type save_path: Union[str, None]

        :return: None
        """
        self.cls_driver._plot_prediction_dataset_distribution(save_path)

    def _most_confused_bar(
            self,
            save_path: Union[str, None]):
        """
        A function to plot most confused bar

        :param save_path: The path to be saved the plot in
        :type save_path: Union[str, None]

        :return: None
        """
        self.cls_driver._plot_most_confused_bar(save_path)

    def _confidence_histogram_scatter_distribution(
            self,
            save_path: Union[str, None]):
        """
        A function to plot confidence histogram scatter distribution

        :param save_path: The path to be saved the plot in
        :type save_path: Union[str, None]

        :return: None
        """
        self.cls_driver._plot_confidence_histogram_scatter_distribution(save_path)

    def _lift_chart(
            self,
            save_path: Union[str, None]):
        """
        A function to plot lift chart

        :param save_path: The path to be saved the plot in
        :type save_path: Union[str, None]

        :return: None
        """
        self.cls_driver._plot_lift_chart(save_path)


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
            graph_type: str = 'graph_1'
        ):
        """
        A function to return plotting function specific to metric

        :param metric_name: name of the metric
        :type metric_name: str
        :param graph_type: type of graph for performance threshold plot
        :type graph_type: str

        :return: None
        """
        fxn_plot_map = {
            "classification": {
                "class_distribution": self.classification_plotter._class_distribution,
                "blind_spot": self.classification_plotter._blind_spot,
                "class_performance_by_threshold": self.classification_plotter._class_performance_by_threshold,
                "roc_statistics": self.classification_plotter._roc_statistics,
                "precision_recall_statistics": self.classification_plotter._precision_recall_statistics,
                "confidence_histogram": self.classification_plotter._confidence_histogram,
                "overall_metrics": self.classification_plotter._overall_metrics,
                "confusion_matrix": self.classification_plotter._confusion_matrix,
                "prediction_dataset_distribution": self.classification_plotter._prediction_dataset_distribution,
                "most_confused_bar": self.classification_plotter._most_confused_bar,
                "confidence_histogram_scatter_distribution": self.classification_plotter._confidence_histogram_scatter_distribution,
                "lift_chart": self.classification_plotter._lift_chart
            },
            "object_detection": {
                "mixed_plot": self.object_detection_plotter._mixed_plot,
                "top_misses": self.object_detection_plotter._top_misses,
                "confidence_histogram": self.object_detection_plotter._confidence_histogram,
                "classbased_table": self.object_detection_plotter._classbased_table,
                "overall_metrics": self.object_detection_plotter._overall_metrics,
                "blind_spot": self.object_detection_plotter._blind_spot
            },
            "segmentation": {
                "violin_graph": self.segmentation_plotter._violin_graph,
                "plot_by_meta_data": self.segmentation_plotter._plot_by_meta_data,
                "overall_metrics": self.segmentation_plotter._overall_metrics,
                "classbased_table": self.segmentation_plotter._classbased_table,
                "blind_spot": self.segmentation_plotter._blind_spot
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
