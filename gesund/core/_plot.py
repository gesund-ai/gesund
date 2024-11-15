import bson
from typing import Union, Optional, List, Dict, Any, Callable


from gesund.core._exceptions import PlotError, MetricCalculationError
from gesund.core._schema import UserInputParams, UserInputData
from gesund.core._metrics.classification.classification_metric_plot import Classification_Plot
from gesund.core._metrics.object_detection.object_detection_metric_plot import Object_Detection_Plot
from gesund.core._metrics.semantic_segmentation.segmentation_metric_plot import Semantic_Segmentation_Plot
from gesund.core._data_loaders import DataLoader


class CommonPlots:
    def __init__(self):
        self.cls_driver = None
        self.obj_driver = None
        self.seg_driver = None

    def _class_distribution(
            self, 
            metrics: Union[Dict, List],
            threshold: float,
            save_path: Optional[str]):
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
        
    def _classification_blind_spot(
            self,
            metrics: Union[Dict, List],
            class_types: List,
            save_path: Optional[str]):
        """
        A function to plot classification blind spot metrics

        :param metrics: The metrics to be plotted
        :type metrics: Union[dict, list]
        :param class_types: List of class types to include
        :type class_types: list 
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):


        :return: None 
        """
        self.cls_driver._plot_blind_spot(metrics, class_types, save_path)
           
    def _class_performance_by_threshold(
            self,
            metrics: Union[Dict, List],
            graph_type: str,
            threshold: float, 
            save_path: Optional[str]):
        """
        A function to plot class performance by threshold

        :param metrics: The metrics to be plotted
        :type metrics: Union[dict, list]
        :param graph_type: The type of graph to plot (e.g., 'graph_1')
        :type graph_type: str
        :param threshold: the value to be applied as threshold
        :type threshold: float
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):
        :return: None
        """
        self.cls_driver._plot_class_performance_by_threshold(graph_type, metrics, threshold, save_path)

    def _roc_statistics(
            self,
            roc_class: List,
            save_path: Optional[str]):
        """
        A function to plot ROC statistics

        :param metrics: The metrics to be plotted
        :type metrics: Union[dict, list]
        :param roc_class: List of classes to plot ROC curves for
        :type roc_class: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.cls_driver._plot_roc_statistics(roc_class, save_path)

    def _precision_recall_statistics(
            self,
            pr_class: List,
            save_path: Optional[str]):
        """
        A function to plot precision recall statistics

        :param pr_class: List of classes to plot precision recall curves for
        :type pr_class: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.cls_driver._plot_precision_recall_statistics(pr_class, save_path)

    def _classification_confidence_histogram(
            self,
            confidence_histogram_args: List,
            save_path: Optional[str]):
        """
        A function to plot classification confidence histogram

        :param confidence_histogram_args: List of arguments for confidence histogram
        :type confidence_histogram_args: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.cls_driver._plot_confidence_histogram(confidence_histogram_args, save_path)
        
    def _classification_overall_metrics(
            self,
            overall_metrics_args: List,
            save_path: Optional[str]):
        """
        A function to plot classification overall metrics

        :param overall_metrics_args: List of arguments for overall metrics
        :type overall_metrics_args: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.cls_driver._plot_overall_metrics(overall_metrics_args, save_path)

    def _confusion_matrix(
            self,
            save_path: Optional[str]):
        """
        A function to plot confusion matrix

        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.cls_driver._plot_confusion_matrix(save_path)

    def _prediction_dataset_distribution(
            self,
            save_path: Optional[str]):
        """
        A function to plot prediction dataset distribution

        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.cls_driver._plot_prediction_dataset_distribution(save_path)

    def _most_confused_bar(
            self,
            save_path: Optional[str]):
        """
        A function to plot most confused bar

        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.cls_driver._plot_most_confused_bar(save_path)

    def _confidence_histogram_scatter_distribution(
            self,
            save_path: Optional[str]):
        """
        A function to plot confidence histogram scatter distribution

        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.cls_driver._plot_confidence_histogram_scatter_distribution(save_path)

    def _lift_chart(
            self,
            save_path: Optional[str]):
        """
        A function to plot lift chart

        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.cls_driver._plot_lift_chart(save_path)

    def _object_detection_blind_spot(self,
            blind_spot_args: List,
            save_path: Optional[str]):
        """
        A function to plot object detection blind spot

        :param blind_spot_args: List of arguments for blind spot
        :type blind_spot_args: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.obj_driver._plot_blind_spot(blind_spot_args, save_path)

    
    def _object_detection_overall_metrics(
            self,
            overall_metrics_args: List,
            save_path: Optional[str]):
        """
        A function to plot object detection overall metrics

        :param overall_metrics_args: List of arguments for overall metrics
        :type overall_metrics_args: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.obj_driver._plot_overall_metrics(overall_metrics_args, save_path)

    def _top_misses(
            self,
            top_misses_args: List,
            save_path: Optional[str]):
        """
        A function to plot top misses

        :param top_misses_args: List of arguments for top misses
        :type top_misses_args: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.obj_driver._plot_top_misses(top_misses_args, save_path)

    def _classbased_table_metrics(
            self,
            classbased_table_args: List,
            save_path: Optional[str]):
        """
        A function to plot class based table metrics

        :param classbased_table_args: List of arguments for class based table
        :type classbased_table_args: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.obj_driver._plot_classbased_table_metrics(classbased_table_args, save_path)

    def _mixed_metrics(
            self,
            mixed_metrics_args: List,
            save_path: Optional[str]):
        """
        A function to plot mixed metrics

        :param mixed_metrics_args: List of arguments for mixed metrics
        :type mixed_metrics_args: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.obj_driver._plot_mixed_metrics(mixed_metrics_args, save_path)

    def _object_detection_confidence_histogram(
            self,
            confidence_histogram_args: List,
            save_path: Optional[str]):
        """
        A function to plot object detection confidence histogram

        :param confidence_histogram_args: List of arguments for confidence histogram
        :type confidence_histogram_args: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.obj_driver._plot_confidence_histogram(confidence_histogram_args, save_path)


    def _classbased_table(
            self,
            classbased_table_args: List,
            save_path: Optional[str]):
        """
        A function to plot class based table

        :param classbased_table_args: List of arguments for class based table
        :type classbased_table_args: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.seg_driver._plot_classbased_table(classbased_table_args, save_path)

    def _segmentation_overall_metrics(
            self,
            overall_args: List,
            save_path: Optional[str]):
        """
        A function to plot overall data

        :param overall_args: List of arguments for overall data
        :type overall_args: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.seg_driver._plot_overall_data(overall_args, save_path)

    def _segmentation_blind_spot(self,
            blind_spot_args: List,
            save_path: Optional[str]):
        """
        A function to plot segmentation blind spot

        :param blind_spot_args: List of arguments for blind spot
        :type blind_spot_args: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.seg_driver._plot_blind_spot(blind_spot_args, save_path)

    def _by_meta_data(
            self,
            meta_data_args: List,
            save_path: Optional[str]):
        """
        A function to plot by meta data

        :param meta_data_args: List of arguments for meta data
        :type meta_data_args: List
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.seg_driver._plot_by_meta_data(meta_data_args, save_path)


    def _violin_graph(
            self,
            metrics: Union[Dict, List], 
            threshold: float,
            save_path: Optional[str]):
        """
        A function to plot violin graph

        :param metrics: The metrics to be plotted
        :type metrics: Union[Dict, List]
        :param threshold: the value to be applied as threshold
        :type threshold: float
        :param save_path: The path to be saved the plot in
        :type save_path: Optional[str]):

        :return: None
        """
        self.seg_driver._plot_violin_graph(metrics, threshold, save_path)

class ClassificationPlots(CommonPlots):
    def __init__(self):
        super().__init__()
        self.cls_driver = Classification_Plot()


class ObjectDetectionPlots(CommonPlots):
    def __init__(self):
        super().__init__()
        self.obj_driver = Object_Detection_Plot()

class SegmentationPlots(CommonPlots):
    def __init__(self):
        super().__init__()
        self.seg_driver = Semantic_Segmentation_Plot()


class PlotData:
    FXN_PLOT_MAP = {
        "classification": {
            "class_distribution": lambda self: self.classification_plotter._class_distribution,
            "blind_spot": lambda self: self.classification_plotter._classification_blind_spot,
            "class_performance_by_threshold": lambda self: self.classification_plotter._class_performance_by_threshold,
            "roc_statistics": lambda self: self.classification_plotter._roc_statistics,
            "precision_recall_statistics": lambda self: self.classification_plotter._precision_recall_statistics,
            "confidence_histogram": lambda self: self.classification_plotter._classification_confidence_histogram,
            "overall_metrics": lambda self: self.classification_plotter._classification_overall_metrics,
            "confusion_matrix": lambda self: self.classification_plotter._confusion_matrix,
            "prediction_dataset_distribution": lambda self: self.classification_plotter._prediction_dataset_distribution,
            "most_confused_bar": lambda self: self.classification_plotter._most_confused_bar,
            "confidence_histogram_scatter_distribution": lambda self: self.classification_plotter._confidence_histogram_scatter_distribution,
            "lift_chart": lambda self: self.classification_plotter._lift_chart
        },
        "object_detection": {
            "mixed_metrics": lambda self: self.object_detection_plotter._mixed_metrics,
            "top_misses": lambda self: self.object_detection_plotter._top_misses, 
            "confidence_histogram": lambda self: self.object_detection_plotter._object_detection_confidence_histogram,
            "classbased_table_metrics": lambda self: self.object_detection_plotter._classbased_table_metrics,
            "overall_metrics": lambda self: self.object_detection_plotter._object_detection_overall_metrics,
            "blind_spot": lambda self: self.object_detection_plotter._object_detection_blind_spot
        },
        "segmentation": {
            "violin_graph": lambda self: self.segmentation_plotter._violin_graph,
            "by_meta_data": lambda self: self.segmentation_plotter._by_meta_data,
            "overall_metrics": lambda self: self.segmentation_plotter._segmentation_overall_metrics,
            "classbased_table": lambda self: self.segmentation_plotter._classbased_table,
            "blind_spot": lambda self: self.segmentation_plotter._segmentation_blind_spot
        }
    }

    def __init__(
            self,
            metrics_result: Dict[str, Any],
            user_params: UserInputParams,
            plot_configs: dict,
            user_data: Optional[UserInputData] = None,
            batch_job_id: Optional[str] = None,
            validation_problem_type_factory: Optional[Callable] = None):
        self.metrics_results = metrics_result
        self.user_params = user_params
        
        # set up data loading for plot if the plotData class is not part of the validation
        # execution workflow
        if not user_data:
            self.user_data = self._load_data()
        else:
            self.user_data = user_data
        
        # save the plots
        self.plot_save_dir = "outputs/plots"
        self.batch_job_id = batch_job_id
        if not batch_job_id:
            self.batch_job_id = str(bson.ObjectId())
        self.output_dir = f"outputs/{self.batch_job_id}"

        # instantiate the plotters
        self.classification_plotter = ClassificationPlots()
        self.object_detection_plotter = ObjectDetectionPlots()
        self.segmentation_plotter = SegmentationPlots()
        
        # import the validation problem type factory if plotData is not part of the validation
        # execution flow
        # if this class is part of the  validation execution flow, then importing the
        # validation problem type factory leads to circular imports
        if not validation_problem_type_factory:
            from gesund.validation import ValidationProblemTypeFactory
            self.validation_problem_type_factory = ValidationProblemTypeFactory()
        else:
            self.validation_problem_type_factory = validation_problem_type_factory
        

        # getter methods
        self.get_supported_plots = self.__get_supported_plots(user_params)

        
    
    def _load_data(self):
        """
        A function to load the JSON files

        :return: None
        """
        # Load data
        # set up source data for processing 
        data_loader = DataLoader(self.user_params.data_format)
        data = {
            "prediction": data_loader.load(self.user_params.predictions_path),
            "annotation": data_loader.load(self.user_params.annotations_path)
        }
        if isinstance(self.user_params.class_mapping, str):
            data["class_mapping"] = data_loader.load(self.user_params.class_mapping)
        else:
            data["class_mapping"] = self.user_params.class_mapping
        
        # loading metadata
        if self.user_params.metadata_path:
            data["metadata"] = data_loader.load(self.user_params.metadata_path)

        # run conversion
        if self.user_params.json_structure_type != "gesund":
            self._convert_data(data)
        
        self.data = UserInputData(**data)
    
    @classmethod
    def __get_supported_plots(cls, user_params) -> List[str]:
        """
        Returns list of supported plots for the current problem type

        :param user_params: The parameters provided by the user
        :type user_params: UserInputParams
        
        :return: List of supported plot names
        :rtype: list
        """
        return list(cls.FXN_PLOT_MAP.get(user_params.problem_type, {}).keys())

    def _plot_single_metric(
            self, 
            metric_name: str, 
            threshold: float = 0.0,
            graph_type: str = 'graph_1'
        ) -> None:
        """
        A function to return plotting function specific to metric

        :param metric_name: name of the metric
        :type metric_name: str
        :param threshold: threshold value for performance threshold plot
        :type threshold: float
        :param graph_type: type of graph for performance threshold plot
        :type graph_type: str

        :return: None
        """

        if self.user_params.problem_type not in self.FXN_PLOT_MAP:
            raise ModuleNotFoundError(f"Problem type {self.user_params.problem_type} not supported")

        if metric_name not in self.FXN_PLOT_MAP[self.user_params.problem_type]:
            raise ModuleNotFoundError(f"{metric_name} plotting function not found")


        _plotter_fxn = self.FXN_PLOT_MAP[self.user_params.problem_type][metric_name](self)
        _plotter_fxn(
            metrics=self.metrics_results,
            threshold=threshold,
            save_path=self.plot_save_dir if self.user_params.save_plots else None
        )

    def _plot_all(self, metric_validation_executor) -> None:
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

    def plot(self, metric_name: str = "all", threshold: float = 0.0) -> None:
        """
        Plot the data from the results

        :param metric_name: name of the metric to plot
        :type metric_name: str
        :param threshold: float value indicating the threshold
        :type threshold: float
        :return: None
        """
        if not self.batch_job_id:
            raise ValueError("batch_job_id is required for plotting")

        _validation_class = self.validation_problem_type_factory.get_problem_type_factory(
            self.user_params.problem_type)
        
        _metric_validation_executor = _validation_class(self.batch_job_id)

        # if the metadata path is provided then the metric result needs to be recalculated as per the
        # metadata filters of the interest
        if self.user_params.metadata_path:
            if self.user_data.was_converted:
                prediction = self.user_data.converted_prediction
                annotation = self.user_data.converted_annotation
            else:
                prediction = self.user_data.prediction
                annotation = self.user_data.annotation

            validation_data = _metric_validation_executor.create_validation_collection_data(
                prediction,
                annotation,
                self.user_params.json_structure_type,
                self.user_data.metadata
            )
            metric_results = _metric_validation_executor.load(
                validation_data,
                self.user_data.class_mapping,
                self.user_data.metadata
            )
        else:
            metric_results = self.metrics_results
    
        if metric_name == "all":
            try:
                _metric_validation_executor.plot_metrics(metric_results)
            except Exception as e:
                print(e)
                raise PlotError("Could not plot the metrics!")
        else:
            self._plot_single_metric(metric_name, metric_results, threshold)
