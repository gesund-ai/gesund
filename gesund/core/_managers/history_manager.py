import os
import json
import datetime
import time
from typing import Dict, Any, List, Optional

from gesund.core._managers.base import GenericPMManager

class HistoryRecordManager(GenericPMManager):
    def __init__(
            self,
            history_dir: str = "evaluation_history", 
            history_file: str = "val_history.json"
    ):
        """
        Initialize the HistoryRecordManager with specified history directory and file.

        :param history_dir: The directory where history files are stored. Defaults to "evaluation_history".
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
            "plots": [],
            "status": "completed",
            "error": None
        }

    def update_run_info(
        self,
        run_info: Dict[str, Any],
        plots: List[Any],
        metric_type: str = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Update the run information without including metrics.
        
        Args:
            run_info: Run information to update
            plots: List of plots
            metric_type: Type of metric to filter (e.g. 'classification', 'object_detection')
            error: Optional error message
        """
        if metric_type:
            plots = [p for p in plots if p['metric_name'].startswith(metric_type)]
        
        plot_names = {plot['metric_name'] for plot in plots}
                    
        run_info["plots"] = []
        for name in sorted(plot_names):
            #TODO: Cannot solved time issue added 0.001 to sleep time.
            time.sleep(0.001)
            run_info["plots"].append({
                "metric_name": name,
                "time": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            })
        if error:
            run_info["status"] = "failed"
            run_info["error"] = error


    def clear_history(self) -> None:
        """
        Clear all history data.

        :return: None
        :rtype: None
        """
        self.save_history({"test_runs": []})

    def get_next_run_id(self) -> int:
        """
        Get next available run ID.

        :return: The next run ID.
        :rtype: int
        """       
        existing_history = self.load_history()
        #TODO: Second problem is here, after restart we need to add increment 1 to run_id.
        if not existing_history["test_runs"]:
            return 1
        return max(run["run_id"] for run in existing_history["test_runs"]) + 1


    def clear_and_save_history(
        self, 
        request, 
        metric_manager, 
        plot_manager,
        metric_type: str = None,
        plot_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Clear and save history with filtered plots.

        :param request: The request object.
        :type request: Any
        :param metric_manager: The metric manager.
        :type metric_manager: Any
        :param plot_manager: The plot manager.
        :type plot_manager: Any
        :param metric_type: The type of metric to filter.
        :type metric_type: str
        :param plot_config: The plot configuration dictionary.
        :type plot_config: Dict[str, Any]
        :return: The updated run information dictionary.
        :rtype: Dict[str, Any]
        """
        try:
            self.clear_history()
            
            existing_history = self.load_history()
            run_info = self.create_run_info(existing_history)
    
            all_plots = plot_manager.get_history() or []

            if plot_config:
                problem_type = plot_config.get("problem_type")
                metric_name = plot_config.get("metric_name")
                
                if metric_name:
                    plots = [p for p in all_plots if p['metric_name'] == f"{problem_type}.{metric_name}"]
                elif problem_type:
                    plots = [p for p in all_plots if p['metric_name'].startswith(f"{problem_type}.")]
            else:
                plots = all_plots
            
            self.update_run_info(run_info, plots)
            
            new_history = {"test_runs": [run_info]}
            self.save_history(new_history)
            
            return run_info
                
        except Exception as e:
            error_run_info = {
                "run_id": self.get_next_run_id(),
                "start_date": datetime.datetime.now().date().isoformat(),
                "plots": [],
                "status": "failed",
                "error": str(e)
            }
            self.save_history({"test_runs": [error_run_info]})
            return error_run_info



    def get_history(self, manager, metric_type: str = None) -> List[Dict]:
        """
        Get filtered history from a manager.
        
        Args:
            manager: The manager from which to retrieve history
            metric_type: Type of metric to filter (e.g. 'classification.auc')
        
        Returns:
            List[Dict]: Filtered history data
        """
        history = manager.get_history()
        if metric_type and history:
            return [item for item in history if item['metric_name'] == metric_type]
        return history


history_record_manager = HistoryRecordManager()