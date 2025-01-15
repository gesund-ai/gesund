import datetime
from typing import Callable, Dict, TypeVar, Generic, List, Optional

from gesund.core._exceptions import FunctionNotFoundError, RegistrationNotAllowed

T = TypeVar("T")


class GenericPMManager(Generic[T]):
    """
    A generic class manager for plot and metric manager

    """

    def __init__(self):
        self._store: Dict[str, T] = {}
        self._allowed_registerations = [
            "classification",
            "semantic_segmentation",
            "instance_segmentation",
            "object_detection",
        ]
        self.__history: List[Dict] = []

    def record_usage(self, name: str):
        """
        Record the usage of a metric or plot by appending its name and the current time to the history.

        :param name: The name of the metric or plot being used.
        :type name: str
        """
        self.__history.append(
            {
                "metric_name": name,
                "time": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
            }
        )

    def get_history(self) -> List[Dict]:
        """
        Get the history of recorded usages.

        :return: A list of dictionaries containing metric names and their usage times.
        """
        return self.__history

    def clear_history(self):
        """
        Clear the recorded usage history.
        """
        self.__history.clear()

    def register(self, name: str) -> Callable[[T], T]:
        """
        A decorator function to register a function under a given name:

        :param name: the name to register the function under
        :type name: str

        :return: The decorator function
        :rtype: Callable
        """

        def decorator(func: T) -> T:
            if "." in name:
                _k1, _k2 = name.split(".")
                if _k1 not in self._allowed_registerations:
                    raise RegistrationNotAllowed(
                        f"{_k1} not allowed, only allowed {','.join(self._allowed_registerations)}"
                    )

                if _k1 in self._store:
                    self._store[_k1][_k2] = func
                else:
                    self._store[_k1] = {}
                    self._store[_k1][_k2] = func
            else:
                self._store[name] = func
            return func

        return decorator

    def get_names(self, problem_type: Optional[str] = None) -> List[str]:
        """
        Get the names of the registered functions,

        :param problem_type: name of the problem type
        :type problem_type:str

        :return: A list of registered names.
        """
        if problem_type:
            return list(self._store[problem_type].keys())
        return list(self._store.keys())

    def __getitem__(self, key: str) -> Callable:
        """
        Get the metric function given the metric name

        :param key: The metric / plot name
        :type key: str

        :return: the metric function associated with the given name
        :raises FunctionNotFoundError: if the registration key is not found
        :rtype: Callable
        """
        try:
            if "." in key:
                _k1, _k2 = key.split(".")
                return self._store[_k1][_k2]
            else:
                return self._store[key]
        except Exception as e:
            print(e)
            raise FunctionNotFoundError(f"{key} not registered")
