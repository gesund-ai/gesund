import pandas as pd
import warnings
from tqdm import tqdm

from .top_losses import PlotTopLosses
from .auc import PlotAUC
from .confusion_matrix import PlotConfusionMatrix
from .threshold import PlotThreshold
from .parallel_plot import PlotParallel
from .prediction_data_analysis import PlotPredictionDataAnalysis
from .most_confused import PlotMostConfused
from .stats_tables import PlotStatsTables
from .blind_spot import PlotBlindSpot
from .lift_chart import PlotLiftGainChart


class ClassificationPlotDriver:
    def __init__(
        self,
        true,
        pred,
        meta,
        pred_categorical,
        pred_logits,
        meta_pred_true,
        class_mappings,
        loss,
        filtering_meta,
    ):
        self.filtering_meta = filtering_meta
        # Import Classes
        self.plot_loss = PlotTopLosses(
            pred=pred,
            loss=loss,
            meta_pred_true=meta_pred_true,
            meta=meta,
            class_mappings=class_mappings,
        )
        self.plot_auc = PlotAUC(
            pred_logits=pred_logits,
            meta_pred_true=meta_pred_true,
            class_mappings=class_mappings,
        )
        self.plot_confusion_matrix_ = PlotConfusionMatrix(
            meta_pred_true=meta_pred_true, class_mappings=class_mappings
        )
        self.plot_most_confused = PlotMostConfused(
            true=true,
            pred=pred,
            pred_categorical=pred_categorical,
            meta=meta,
            meta_pred_true=meta_pred_true,
            loss=loss,
            class_mappings=class_mappings,
        )
        self.plot_threshold = PlotThreshold(
            true=true, pred_logits=pred_logits, class_mappings=class_mappings
        )
        self.plot_parallel = PlotParallel(meta_pred_true=meta_pred_true)

        self.plot_prediction_data_analysis = PlotPredictionDataAnalysis(
            true=true,
            pred_logits=pred_logits,
            pred_categorical=pred_categorical,
            meta_pred_true=meta_pred_true,
            class_mappings=class_mappings,
            meta=meta,
        )

        self.plot_stats_tables = PlotStatsTables(
            true=true,
            pred_logits=pred_logits,
            pred_categorical=pred_categorical,
            meta_pred_true=meta_pred_true,
            class_mappings=class_mappings,
        )

        self.plot_lift_gain_chart = PlotLiftGainChart(
            true=true,
            pred_logits=pred_logits,
            class_mappings=class_mappings,
            meta=meta,
        )

        self.plot_blind_spot = PlotBlindSpot(
            true=true,
            pred_logits=pred_logits,
            pred_categorical=pred_categorical,
            meta_pred_true=meta_pred_true,
            class_mappings=class_mappings,
        )

    # Loss Graphs
    def plot_top_losses(self, predicted_class=None, top_k=100):
        return self.plot_loss.top_losses(predicted_class=predicted_class, top_k=top_k)

    # AUC Related Graphs
    def plot_precision_recall_multiclass_statistics(self, target_attribute_dict=None):
        return self.plot_auc.precision_recall_multiclass_statistics(
            target_attribute_dict=self.filtering_meta
        )

    def plot_roc_multiclass_statistics(self, target_attribute_dict=None):
        return self.plot_auc.roc_multiclass_statistics(
            target_attribute_dict=self.filtering_meta
        )

    def plot_confidence_histogram_scatter_distribution(
        self, predicted_class=0, n_samples=300, randomize_x=True, n_bins=25
    ):
        return self.plot_auc.confidence_histogram_scatter_distribution(
            predicted_class=predicted_class,
            n_samples=n_samples,
            randomize_x=randomize_x,
            n_bins=n_bins,
        )

    # Confusion Matrix Related Graphs
    def plot_confusion_matrix(self, target_attribute_dict=None):
        return self.plot_confusion_matrix_.confusion_matrix_(
            target_attribute_dict=self.filtering_meta
        )

    # Dataset Stats
    def plot_meta_distributions(self):
        return self.plot_prediction_data_analysis.meta_distributions()

    def plot_class_distributions(self):
        return self.plot_prediction_data_analysis.class_distributions()

    def plot_prediction_dataset_distribution(self, target_attribute_dict=None):
        return self.plot_prediction_data_analysis.prediction_dataset_distribution(
            target_attribute_dict=self.filtering_meta
        )

    # Confusion Related Graphs
    def plot_most_confused_bar(self, top_k=5):
        return self.plot_most_confused.most_confused_bar(top_k=top_k)

    def plot_most_confused_class_images(
        self,
        true_class=0,
        predicted_class=0,
        top_k=10,
        rank_mode="best_prediction_probability",
    ):
        return self.plot_most_confused.most_confused_class_images(
            true_class, predicted_class, top_k=top_k, rank_mode=rank_mode
        )

    # Threshold Graphs
    def plot_class_performance_by_threshold(self, predicted_class=0, threshold=0.5):
        return self.plot_threshold.class_performance_by_threshold(
            predicted_class=predicted_class, threshold=threshold
        )

    # Parallel Plots
    def plot_parallel_categorical_analysis(self, true_class=0):
        return self.plot_parallel.parallel_categorical_analysis(true_class=true_class)

    # Prediction Analysis
    def plot_prediction_distribution(self, target_attribute_dict=None):
        return self.plot_prediction_data_analysis.prediction_distribution(
            target_attribute_dict=self.filtering_meta
        )

    def plot_explore_predictions(self, predicted_class=0, top_k=3000):
        return self.plot_prediction_data_analysis.explore_predictions(
            predicted_class=predicted_class, top_k=top_k
        )

    def plot_gtless_confidence_histogram_scatter_distribution(
        self, predicted_class="overall", n_samples=300, randomize_x=True, n_bins=25
    ):
        return self.plot_prediction_data_analysis.gtless_confidence_histogram_scatter_distribution(
            predicted_class=predicted_class,
            n_samples=n_samples,
            randomize_x=randomize_x,
            n_bins=n_bins,
        )

    def plot_softmax_probabilities_distribution(self, predicted_class=0, n_bins=10):
        return self.plot_prediction_data_analysis.softmax_probabilities_distribution(
            predicted_class=predicted_class, n_bins=n_bins
        )

    # Stats Tables
    def plot_training_validation_comparison_classbased_table(
        self,
    ):
        return self.plot_stats_tables.training_validation_comparison_classbased_table()

    def plot_tp_tn_fp_fn(self, target_class=0, target_attribute_dict=None):
        return self.plot_stats_tables.tp_tn_fp_fn(
            target_class=target_class, target_attribute_dict=self.filtering_meta
        )

    def plot_lift_chart(self, predicted_class=None, target_attribute_dict=None):
        return self.plot_lift_gain_chart.lift_chart(
            target_attribute_dict=self.filtering_meta, predicted_class=predicted_class
        )

    def plot_statistics_classbased_table(self, target_attribute_dict=None):
        return self.plot_stats_tables.statistics_classbased_table(
            target_attribute_dict=self.filtering_meta
        )

    def plot_highlighted_overall_metrics(self):
        return self.plot_stats_tables.highlighted_overall_metrics()

    def plot_class_performances(self):
        return self.plot_stats_tables.class_performances()

    # Blind Spots
    def plot_blind_spot_metrics(self, target_attribute_dict=None):
        return self.plot_blind_spot.blind_spot_metrics(
            target_attribute_dict=self.filtering_meta
        )

    def _calling_all_plots(self):
        # Getting all methods that do not start with '_'
        methods = [
            method_name
            for method_name in dir(self)
            if callable(getattr(self, method_name)) and not method_name.startswith("_")
        ]
        results = {}
        for method_name in tqdm(methods, desc="Calling all plot functions"):
            method = getattr(self, method_name)
            try:
                # Suppress warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    print(f"Calling method: {method_name}...")
                    # Attempt to call the method, handle cases where no arguments are required
                    result = method()
                    results[method_name] = result
            except TypeError as e:
                results[method_name] = f"Could not call {method_name}: {str(e)}"
            except Exception as e:
                results[method_name] = f"Error in {method_name}: {str(e)}"

        return results
