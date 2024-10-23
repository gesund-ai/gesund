import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

class Classification_Plot:
    """
    A class for creating various plots and visualizations for classification metrics results.

    :param blind_spot_path: (str, optional) Path to the blind spot analysis JSON file.
    :param performance_threshold_path: (str, optional) Path to the performance by threshold JSON file.
    :param class_distributions_path: (str, optional) Path to the class distributions JSON file.
    :param roc_statistics_path: (str, optional) Path to the ROC statistics JSON file.
    :param precision_recall_statistics_path: (str, optional) Path to the precision-recall statistics JSON file.
    :param confidence_histogram_path: (str, optional) Path to the confidence histogram JSON file.
    :param overall_json_path: (str, optional) Path to the overall metrics JSON file.
    :param mixed_json_path: (str, optional) Path to the mixed metrics JSON file.
    :param output_dir: (str, optional) Directory path for saving output plots.
    """

    def __init__(self, blind_spot_path=None, performance_threshold_path=None, class_distributions_path=None, 
                roc_statistics_path=None, precision_recall_statistics_path=None,
                confidence_histogram_path=None, overall_json_path=None, mixed_json_path=None,

                output_dir=None):
        
        self.output_dir = output_dir
        if class_distributions_path:
            self.class_data = self._load_json(class_distributions_path)
        if blind_spot_path:
            self.metrics_data = self._load_json(blind_spot_path)
        if performance_threshold_path:
            self.performance_by_threshold = self._load_json(performance_threshold_path)
        if roc_statistics_path:
            self.roc_statistics = self._load_json(roc_statistics_path)
        if precision_recall_statistics_path:
            self.precision_recall_statistics = self._load_json(precision_recall_statistics_path)
        if confidence_histogram_path:
            self.confidence_histogram_data = self._load_json(confidence_histogram_path)
        if overall_json_path:
            self.overall_json_data = self._load_json(overall_json_path)
        if mixed_json_path:
            self.mixed_json_data = self._load_json(mixed_json_path)



    def _load_json(self, json_path):
        """
        Load data from a JSON file.

        :param json_path: (str) Path to the JSON file to be loaded.

        :return: (dict) Loaded JSON data as a Python dictionary.
        """
        with open(json_path, 'r') as file:
            data = json.load(file)
        return data
    
    def draw(self, plot_type, metrics=None, threshold=None, class_type='Average',
                graph_type='graph_1',roc_class='normal', pr_class='normal',
                confidence_histogram_args=None, overall_args=None, mixed_args=None,
                 save_path=None):
        """
        Draw various types of plots based on the specified plot type and parameters.

        :param plot_type: (str) Type of plot to generate ('class_distributions', 'blind_spot', 'performance_by_threshold', 
                          'roc', 'precision_recall', 'confidence_histogram', 'overall_metrics', 'mixed_plot').
        :param metrics: (list, optional) List of specific metrics or classes to include in the plot.
        :param threshold: (float, optional) Threshold value for filtering metrics.
        :param class_type: (str or list, optional) Class type(s) for 'blind_spot' plot.
        :param graph_type: (str, optional) Graph type for 'performance_by_threshold' plot.
        :param roc_class: (str or list, optional) Class(es) for 'roc' plot.
        :param pr_class: (str or list, optional) Class(es) for 'precision_recall' plot.
        :param confidence_histogram_args: (dict, optional) Arguments for 'confidence_histogram' plot.
        :param overall_args: (dict, optional) Arguments for 'overall_metrics' plot.
        :param mixed_args: (dict, optional) Arguments for 'mixed_plot'.
        :param save_path: (str, optional) Path to save the generated plot.

        :raises ValueError: If an unsupported plot type is specified.
        """
        if plot_type == 'class_distributions':
            self._plot_class_distributions(metrics, threshold, save_path)
        elif plot_type == 'blind_spot':
            self._plot_blind_spot(class_type, save_path)
        elif plot_type == 'performance_by_threshold':
            self._plot_class_performance_by_threshold(graph_type, metrics, threshold, save_path)
        elif plot_type == 'roc':
            self._plot_roc_statistics(roc_class, save_path)
        elif plot_type == 'precision_recall':
            self._plot_precision_recall_statistics(pr_class, save_path)
        elif plot_type == 'confidence_histogram':
            self._plot_confidence_histogram(confidence_histogram_args, save_path)
        elif plot_type == 'overall_metrics':
            self._plot_overall_metrics(overall_args, save_path)
        elif plot_type == 'mixed_plot':
            self._plot_mixed_metrics(mixed_args, save_path)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        



    def _plot_class_distributions(self, metrics=None, threshold=None, save_path=None):
        """
        Create a bar plot visualization for class distributions in validation data.

        :param metrics: (list, optional) List of specific classes to include in the plot.
        :param threshold: (float, optional) Minimum count threshold for filtering classes.
        :param save_path: (str, optional) Path where the plot should be saved. If None, saves as 'class_distributions.png'.

        :return: None
        :raises AttributeError: If no valid class distribution data is loaded.
        """
        if not hasattr(self, 'class_data') or self.class_data.get('type') != 'bar':
            print("No valid 'bar' data found in the JSON.")
            return
        
        validation_data = self.class_data.get('data', {}).get('Validation', {})        
        df = pd.DataFrame(list(validation_data.items()), columns=['Class', 'Count'])        
        if metrics:
            df = df[df['Class'].isin(metrics)]
        
        if threshold is not None:
            df = df[df['Count'] >= threshold]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Class', y='Count', data=df, palette='pastel', width=0.6)
        plt.title('Class Distribution in Validation Data', fontsize=18, fontweight='bold', pad=20)        
        plt.xlabel('Class Type', fontsize=14, labelpad=15)
        plt.ylabel('Number of Samples', fontsize=14, labelpad=15)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        for index, value in enumerate(df['Count']):
            plt.text(index, value + 0.5, f'{value}', ha='center', fontsize=12, color='black', fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.savefig('class_distributions.png', bbox_inches='tight', dpi=300)
        plt.show()        
        plt.close()



    def _plot_blind_spot(self, class_types, save_path=None):
        """
        Create a grouped bar plot visualization for comparing performance metrics across class types.

        :param class_types: (list) List of class types to include in the plot.
        :param save_path: (str, optional) Path where the plot should be saved. If None, saves as 'class_comparison_metrics.png'.

        :return: None
        :raises AttributeError: If no metrics data is loaded.
        """

        if not hasattr(self, 'metrics_data'):
            print("No metrics data found.")
            return
        
        all_metrics_df = pd.DataFrame()
        
        for class_type in class_types:
            class_metrics = self.metrics_data.get(class_type, {})
            df = pd.DataFrame(list(class_metrics.items()), columns=['Metric', 'Value'])
            df = df[~df['Metric'].isin(['Sample Size', 'Class Size','Matthews C C'])]
            df['Class Type'] = class_type            
            all_metrics_df = pd.concat([all_metrics_df, df])
        
        plt.figure(figsize=(14, 10))
        
        sns.catplot(
            data=all_metrics_df,
            x='Value', y='Metric', hue='Class Type', kind='bar', palette='pastel',
            height=8, aspect=1.5
        )
        
        plt.title('Comparison of Performance Metrics Across Class Types', fontsize=18, fontweight='bold', pad=20)        
        plt.xlabel('Metric Value', fontsize=14, labelpad=15)
        plt.ylabel('Metric', fontsize=14, labelpad=15)
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.savefig('class_comparison_metrics.png', bbox_inches='tight', dpi=300)
        
        plt.show()
        
        plt.close()


    def _plot_class_performance_by_threshold(self, graph_type, metrics, threshold, save_path=None):
        """
        Create a bar plot visualization for class performance metrics by threshold.

        :param graph_type: (str) The type of graph to plot (e.g., 'graph_1').
        :param metrics: (list, optional) List of specific metrics to include in the plot.
        :param threshold: (float, optional) Minimum threshold value for filtering metrics.
        :param save_path: (str, optional) Path where the plot should be saved. 
                          If None, saves as '{graph_type}_performance_metrics.png'.

        :return: None
        :raises AttributeError: If no performance threshold data is loaded.
        """

        if not self.performance_by_threshold:
            print("No performance threshold data found.")
            return

        performance_metrics = self.performance_by_threshold.get('data', {}).get(graph_type, {})
        
        df = pd.DataFrame(list(performance_metrics.items()), columns=['Metric', 'Value'])
        
        if metrics:
            df = df[df['Metric'].isin(metrics)]
        
        if threshold is not None:
            df = df[df['Value'] >= threshold]
        
        plt.figure(figsize=(12, 8))
        
        sns.barplot(x='Value', y='Metric', data=df, palette='pastel')

        plt.title(f'{graph_type} Performance Metrics (Threshold ≥ {threshold})', fontsize=18, fontweight='bold', pad=20)

        plt.xlabel('Metric Value', fontsize=14, labelpad=15)
        plt.ylabel('Metric Name', fontsize=14, labelpad=15)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.grid(True, axis='x', linestyle='--', alpha=0.7)

        for index, value in enumerate(df['Value']):
            plt.text(value + 0.01, index, f'{value:.4f}', va='center', fontsize=12, color='black', fontweight='bold')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'{graph_type}_performance_metrics.png', bbox_inches='tight', dpi=300)

        plt.show()        
        plt.close()



    def _plot_roc_statistics(self, roc_classes, save_path=None):
        """
        Create ROC curves for specified classes.

        :param roc_classes: (list) List of classes to plot ROC curves for.
        :param save_path: (str, optional) Path where the plot should be saved.

        :return: None
        :raises AttributeError: If no ROC statistics data is loaded.
        """
        if not hasattr(self, 'roc_statistics'):
            print("No ROC statistics data found.")
            return
        
        plt.figure(figsize=(10, 8))
        
        sns.set(style="whitegrid", rc={"axes.facecolor": "lightgrey", "grid.color": "white", "axes.edgecolor": "black"})
        
        for roc_class in roc_classes:
            roc_data = self.roc_statistics.get('data', {}).get('points', {}).get(roc_class, [])
            
            if not roc_data:
                print(f"No data found for class: {roc_class}")
                continue  
            
            df = pd.DataFrame(roc_data)
            
            plt.plot(df['fpr'], df['tpr'], marker='o', linestyle='-', lw=2, markersize=8, 
                    label=f'ROC curve for {roc_class} (AUC = {self.roc_statistics["data"]["aucs"][roc_class]:.2f})')
        
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Chance')
        
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curve for {", ".join(roc_classes)}', fontsize=16, weight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='lower right', fontsize=12)
        plt.tight_layout()  
        if save_path:
            plt.savefig(save_path, dpi=300)  
        plt.show()        
        plt.close()





    def _plot_precision_recall_statistics(self, pr_classes, save_path):
        """
        Create Precision-Recall curves for specified classes.

        :param pr_classes: (list) List of classes to plot Precision-Recall curves for.
        :param save_path: (str, optional) Path where the plot should be saved.

        :return: None
        :raises AttributeError: If no Precision-Recall statistics data is loaded.
        """

        if not hasattr(self, 'precision_recall_statistics'):
            print("No Precision-Recall statistics data found.")
            return


        plt.figure(figsize=(12, 8))
        sns.set(style="darkgrid", rc={"axes.facecolor": "lightgrey", "grid.color": "white", "axes.edgecolor": "black"})

        for pr_class in pr_classes:
            pr_data = self.precision_recall_statistics.get('data', {}).get('points', {}).get(pr_class, [])
            if not pr_data:
                print(f"No data found for class: {pr_class}")
                continue

            df = pd.DataFrame(pr_data)
            df = df.sort_values(by='x') 
            plt.plot(df['x'], df['y'], marker='o', linestyle='-', label=f'{pr_class} (AUC = {self.precision_recall_statistics["data"]["aucs"][pr_class]:.2f})', lw=2)
            plt.scatter(df['x'], df['y'], s=100, zorder=5)

            for i in range(len(df)):
                plt.annotate(f'{df["threshold"].iloc[i]:.2f}', 
                            (df['x'].iloc[i], df['y'].iloc[i]), 
                            textcoords="offset points", xytext=(5,5), ha='center', fontsize=10)

        plt.title('Precision-Recall Curves', fontsize=18, weight='bold')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.legend(loc='lower left', fontsize=12)
        plt.tight_layout()
        plt.show()

        if save_path:
            plt.savefig(save_path)
        plt.close()


    def _plot_confidence_histogram(self, confidence_histogram_args, save_path):
        """
        Create scatter and histogram plots for confidence histogram data.

        :param confidence_histogram_args: (dict, optional) Dictionary containing:
            - 'labels': (list) List of labels to include in the scatter plot.
        :param save_path: (str, optional) Path where the plots should be saved.
                          If None, saves as 'scatter_plot_points.png' and 'histogram.png'.

        :return: None
        :raises AttributeError: If no valid confidence histogram data is loaded.
        """
        if not self.confidence_histogram_data or self.confidence_histogram_data.get('type') != 'mixed':
            print("No valid 'confidence_histogram' data found in the new JSON.")
            return
        
        points_data = self.confidence_histogram_data.get('data', {}).get('points', [])
        histogram_data = self.confidence_histogram_data.get('data', {}).get('histogram', [])
        points_df = pd.DataFrame(points_data)
        histogram_df = pd.DataFrame(histogram_data)
        
        if confidence_histogram_args:
            if 'labels' in confidence_histogram_args:
                points_df = points_df[points_df['labels'].isin(confidence_histogram_args['labels'])]
        
        # Scatter Plot
        plt.figure(figsize=(12, 8))
        sns.set(style="whitegrid")
        sns.scatterplot(
            x='x', y='y', hue='labels', data=points_df, palette='pastel', 
            s=100, alpha=0.8, edgecolor="k"
        )
        plt.title('Scatter Plot of Points', fontsize=18, weight='bold')
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)        
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title='Labels', fontsize=12, title_fontsize=14, loc='upper right', frameon=True)
        plt.show()

        # Histogram Plot
        plt.figure(figsize=(12, 8))
        custom_palette = sns.color_palette('pastel', n_colors=len(histogram_df))        
        bars = sns.barplot(x='category', y='value', data=histogram_df, palette=custom_palette)
        plt.title('Confidence Histogram', fontsize=18, weight='bold')
        plt.xlabel('Category', fontsize=14)
        plt.ylabel('Value', fontsize=14)        
        plt.grid(True, linestyle='--', linewidth=0.5)
        handles = [plt.Rectangle((0,0),1,1, color=custom_palette[i]) for i in range(len(histogram_df))]
        labels = histogram_df['category'].tolist()
        plt.legend(handles, labels, title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14, frameon=True)        
        plt.show()

        if save_path:
            plt.savefig(save_path)
        plt.close()



    def _plot_overall_metrics(self, overall_args, save_path):
        """
        Create a bar plot visualization for overall validation metrics.

        :param overall_args: (dict, optional) Dictionary containing:
            - 'metrics': (list) List of specific metrics to include in the plot.
            - 'threshold': (float) Minimum threshold value for filtering metrics.
        :param save_path: (str, optional) Path where the plot should be saved.

        :return: None
        :raises AttributeError: If no valid overall data is loaded.
        """
        if not self.overall_json_data or self.overall_json_data.get('type') != 'overall':
            print("No valid 'overall' data found in the new JSON.")
            return
        
        data = self.overall_json_data.get('data', {})
        df = pd.DataFrame([(k, v['Validation']) for k, v in data.items() if k != 'Matthews C C'], columns=['Metric', 'Value'])
        
        if overall_args:
            if 'metrics' in overall_args:
                df = df[df['Metric'].isin(overall_args['metrics'])]
            if 'threshold' in overall_args:
                df = df[df['Value'] > overall_args['threshold']]
        
        df = df.sort_values('Value', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Value', y='Metric', data=df, palette='viridis')
        plt.title('Overall Metrics', fontsize=16)
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Metric', fontsize=12)
        for i, v in enumerate(df['Value']):
            plt.text(v, i, f' {v:.4f}', va='center')
        plt.tight_layout()
        plt.show()

        if save_path:
            plt.savefig(save_path)
        plt.close()