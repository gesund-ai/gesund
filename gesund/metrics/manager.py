from typing import Callable


class MetricsManager:
    def __init__(self):
        """
        An initialization function

        """
        self.metrics_store = {
            "classification": {},
            "instance_segmentation": {},
            "semantic_segmentation": {},
            "object_detection": {},
        }

    def register_metrics(self, problem_type: str, metric_name: str) -> None:
        """
        A function to register the metric

        :param problem_type: value of the problem type
        :type problem_type: str
        :param metric_name: name of the metric
        :type metric_name: str

        :return: The decorator function
        """

        def decorator(metrics_fxn: Callable) -> Callable:
            self.metrics_store[problem_type][metric_name] = metrics_fxn
            return metrics_fxn

        return decorator
