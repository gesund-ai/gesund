import os
import json
import datetime
from typing import Dict, Any, List, Optional

from gesund.core._managers.base import GenericPMManager

class HistoryRecordManager(GenericPMManager):
    def __init__(
            self,
            history_dir: str = "validation_mechanism_history", 
            history_file: str = "val_history.json"
    ):
        """
        Initialize the HistoryRecordManager with specified history directory and file.

        :param history_dir: The directory where history files are stored. Defaults to "validation_mechanism_history".
        :type history_dir: str
        :param history_file: The name of the history JSON file. Defaults to "val_history.json".
        :type history_file: str
        """
        super().__init__()
        self.history_dir = history_dir
        self.history_file = os.path.join(history_dir, history_file)
        os.makedirs(self.history_dir, exist_ok=True)

    def load_history(self) -> Dict[str, Any]:
        """
        Load history from a JSON file.

        Attempts to read and parse the history JSON file. If the file does not exist or contains
        invalid JSON, it returns a default history structure.

        :return: The loaded history data.
        :rtype: Dict[str, Any]
        """
        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"test_runs": []}

    def save_history(self, history: Dict[str, Any]) -> None:
        """
        Save history to a JSON file.

        :param history: The history data to save.
        :type history: Dict[str, Any]
        
        :return: None
        :rtype: None
        """
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=4)

    def create_run_info(self, existing_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new run information dictionary.

        :param existing_history: The current history data.
        :type existing_history: Dict[str, Any]
        
        :return: A new run information dictionary.
        :rtype: Dict[str, Any]
        """
        return {
            "run_id": len(existing_history["test_runs"]) + 1,
            "start_date": datetime.datetime.now().date().isoformat(),
            "metrics": [],
            "plots": [],
            "status": "completed",
            "error": None
        }

    def update_run_info(
        self,
        run_info: Dict[str, Any],
        metrics: List[Any],
        plots: List[Any],
        error: Optional[str] = None,
    ) -> None:
        """
        Update the run information with metrics, plots, and error details.

        :param run_info: The run information to update.
        :type run_info: Dict[str, Any]
        :param metrics: A list of metrics associated with the run.
        :type metrics: List[Any]
        :param plots: A list of plots associated with the run.
        :type plots: List[Any]
        :param error: An error message if the run failed. Defaults to None.
        :type error: Optional[str]

        :return: None
        :rtype: None
        """
        run_info["metrics"] = metrics
        run_info["plots"] = plots
        if error:
            run_info["status"] = "failed"
            run_info["error"] = error

    def clear_and_save_history(
            self, request, metric_manager, plot_manager
    ):
        """
        Clear and save history before and after tests.

        :param request: The pytest request object used to determine the test context.
        :type request: Any
        :param metric_manager: The manager responsible for handling metrics.
        :type metric_manager: Any
        :param plot_manager: The manager responsible for handling plots.
        :type plot_manager: Any

        :yield: None
        :rtype: None
        """
        existing_history = self.load_history()
        run_info = self.create_run_info(existing_history)

        metric_manager.clear_history()
        plot_manager.clear_history()

        yield

        if request.node.name.startswith('test_plot_manager_single_metric'):
            try:
                metrics = self.get_history(metric_manager)
                plots = self.get_history(plot_manager)
                self.update_run_info(run_info, metrics, plots)
            except Exception as e:
                self.update_run_info(run_info, [], [], str(e))

            existing_history["test_runs"].append(run_info)
            self.save_history(existing_history)

    def get_history(self, manager) -> List[Dict]:
        """
        Get history from a manager.

        :param manager: The manager from which to retrieve history.
        :type manager: Any

        :return: The retrieved history data.
        :rtype: List[Dict]
        """
        return manager.get_history()

history_record_manager = HistoryRecordManager()