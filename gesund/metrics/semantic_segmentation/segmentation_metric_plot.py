import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 

class Semantic_Segmentation_Plot:
    """
    A class for creating various plots and visualizations for semantic segmentation results.

    :param violin_path: (str, optional) Path to the violin plot JSON data file.
    :param result_dict: (dict, optional) Dictionary containing result data.
    :param classed_table: (str, optional) Path to the class-based table JSON data file.
    :param overall_data: (str, optional) Path to the overall metrics JSON data file.
    :param blind_spot: (str, optional) Path to the blind spot analysis JSON data file.
    :param plot_by_meta_data: (str, optional) Path to the metadata plot JSON data file.
    :param output_dir: (str, optional) Directory path for saving output plots.
    """

    def __init__(self, violin_path=None, result_dict=None, classed_table=None, overall_data=None, blind_spot=None, 
                            plot_by_meta_data=None, output_dir=None):
        self.output_dir = output_dir
        if violin_path:
            self.violin_data = self._load_json(violin_path)
        if result_dict:
            self.result_dict = result_dict
        if classed_table:
            self.classbased_table = self._load_json(classed_table)
        if overall_data:
            self.overall_data = self._load_json(overall_data)
        if blind_spot:
            self.blind_spot_data = self._load_json(blind_spot)
        if plot_by_meta_data:
            self.plot_by_meta_data = self._load_json(plot_by_meta_data)
        
    def _load_json(self, json_path):
        """
        Load data from a JSON file.

        :param json_path: (str) Path to the JSON file to be loaded.
        :return: (dict) Loaded JSON data as a Python dictionary.
        """
        with open(json_path, 'r') as file:
            data = json.load(file)
        return data

    def draw(self, plot_type, metrics=None, threshold=None, classbased_table_args=None, overall_args=None, 
                        blind_spot_args=None, meta_data_args=None, save_path=None):
        """
        Draw various types of plots based on the specified plot type and parameters.

        :param plot_type: (str) Type of plot to generate ('violin_graph', 'classbased_table', 'overall_metrics', 'blind_spot', 'plot_by_meta_data').
        :param metrics: (list, optional) List of metrics to plot for violin graph.
        :param threshold: (float, optional) Threshold value for violin graph.
        :param classbased_table_args: (dict, optional) Arguments for class-based table plot.
        :param overall_args: (dict, optional) Arguments for overall metrics plot.
        :param blind_spot_args: (dict, optional) Arguments for blind spot analysis plot.
        :param meta_data_args: (dict, optional) Arguments for metadata-based plot.
        :param save_path: (str, optional) Path to save the generated plot.
        
        :raises ValueError: If an unsupported plot type is specified.
        """
        if plot_type == 'violin_graph':
            self._plot_violin_graph(metrics, threshold, save_path)
        elif plot_type == 'classbased_table':
            self._plot_classbased_table(classbased_table_args, save_path)
        elif plot_type == 'overall_metrics':
            self._plot_overall_data(overall_args, save_path)
        elif plot_type == 'blind_spot':
            self._plot_blind_spot(blind_spot_args, save_path)
        elif plot_type == 'plot_by_meta_data':
            self._plot_by_meta_data(meta_data_args, save_path)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
    def _plot_violin_graph(self, metrics=None, threshold=None, save_path=None):
        """
        Create a violin plot visualization for specified metrics data.

        :param metrics: (list, optional) List of metric names to include in the plot. If None, all metrics will be plotted.
        :param threshold: (float, optional) Minimum threshold value for filtering data points.
        :param save_path: (str, optional) Path where the plot should be saved. If None, saves as 'violin_plot.png'.
        
        :return: None
        :raises AttributeError: If no valid violin data is loaded.
        """
        if not hasattr(self, 'violin_data') or self.violin_data.get('type') != 'violin':
            print("No valid 'violin' data found in the JSON.")
            return
        
        data = self.violin_data.get('data', {})
        df = pd.DataFrame(data)
        
        if metrics:
            df = df[metrics]
        
        if threshold is not None:
            df = df[df.apply(lambda x: x > threshold)]
        
        plt.figure(figsize=(14, 8))
        sns.violinplot(data=df, palette='pastel', linewidth=1.5, inner='box')
        plt.title('Violin Plot of Metrics', fontsize=20, fontweight='bold', pad=20)
        
        plt.xlabel('Metrics', fontsize=14, labelpad=15)
        plt.ylabel('Values', fontsize=14, labelpad=15)
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.savefig('violin_plot.png', bbox_inches='tight', dpi=300)
        
        plt.show()
        plt.close()


    def _plot_classbased_table(self, classbased_table_args=None, save_path=None):
        """
        Create a heatmap visualization of class-based metrics data.

        :param classbased_table_args: (dict, optional) Dictionary containing plotting arguments.
                                    Expected format: {'classbased_table_args': threshold_value}
        :param save_path: (str, optional) Path where the plot should be saved. If None, saves as 'classbased_table_plot.png'.

        :return: None
        :raises AttributeError: If no valid table data is loaded.
        :note: The heatmap uses a coolwarm color scheme and displays values with 3 decimal places.
        """
        if not hasattr(self, 'classbased_table') or self.classbased_table.get('type') != 'table':
            print("No valid 'table' data found in the JSON.")
            return

        data = self.classbased_table.get('data', {})
        flattened_data = {}
        
        for category, metrics in data.items():
            for subcategory, values in metrics.items():
                flattened_data[f"{category}_{subcategory}"] = values

        df = pd.DataFrame(flattened_data).T

        threshold = 0.0  
        if isinstance(classbased_table_args, dict) and 'classbased_table_args' in classbased_table_args:
            threshold = classbased_table_args['classbased_table_args']

        df = df[df > threshold]  
        df = df.replace(0, np.nan)

        df = df.dropna(how='all').dropna(axis=1, how='all')

        plt.figure(figsize=(16, 10))
        
        sns.heatmap(df, annot=True, cmap='coolwarm', fmt='.3f', cbar=True, linewidths=0.5, linecolor='gray')

        plt.title('Class-based Table Plot of Metrics', fontsize=20, fontweight='bold', pad=20)
        plt.xlabel('Metrics', fontsize=14, labelpad=15)
        plt.ylabel('Categories', fontsize=14, labelpad=15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.savefig('classbased_table_plot.png', bbox_inches='tight', dpi=300)

        plt.show()
        plt.close()

    def _plot_overall_data(self, overall_args=None, save_path=None):
        """
        Create a bar plot visualization for overall validation metrics.

        :param overall_args: (list, optional) List of specific metrics to plot. If None, all metrics will be plotted.
        :param save_path: (str, optional) Path where the plot should be saved. If None, saves as 'overall_metrics_plot.png'.

        :return: None
        :raises AttributeError: If no valid overall data is loaded.
        """
        if not hasattr(self, 'overall_data') or self.overall_data.get('type') != 'overall':
            print("No valid 'overall' data found in the JSON.")
            return
        
        data = self.overall_data.get('data', {})
        df = pd.DataFrame({k: v['Validation'] for k, v in data.items()}, index=[0])
        
        if overall_args:
            df = df[overall_args]
        
        df = df.T.reset_index()
        df.columns = ['Metric', 'Value']  
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Metric', y='Value', hue='Metric', data=df, palette='pastel', edgecolor='black', legend=False)
        plt.title('Overall Metrics', fontsize=20, fontweight='bold', pad=20)
        plt.xlabel('Metrics', fontsize=14, labelpad=15)
        plt.ylabel('Values', fontsize=14, labelpad=15)

        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        for i, row in enumerate(df.itertuples()):
            plt.text(i, row.Value + 0.02, f'{row.Value:.4f}', 
                    ha='center', va='bottom', fontsize=12, 
                    color='black', fontweight='bold')
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.savefig('overall_metrics_plot.png', bbox_inches='tight', dpi=300)
        
        plt.show()
        plt.close()

    def _plot_blind_spot(self, blind_spot_args=None, save_path=None):
        """
        Create a bar plot visualization for blind spot metrics comparison.

        :param blind_spot_args: (list, optional) List of specific metrics to plot. Invalid metrics will be filtered out.
        :param save_path: (str, optional) Path where the plot should be saved. If None, saves as 'blind_spot_metrics_plot.png'.

        :return: None
        :raises AttributeError: If no valid blind spot data is loaded.
        """

        if not hasattr(self, 'blind_spot_data'):
            print("No valid 'blind_spot' data found in the JSON.")
            return

        if 'Average' in self.blind_spot_data:
            df = pd.DataFrame(self.blind_spot_data['Average'], index=[0]).T
        if blind_spot_args:
            available_metrics = df.index.tolist()
            valid_args = [arg for arg in blind_spot_args if arg in available_metrics]
            
            if len(valid_args) < len(blind_spot_args):
                missing_args = set(blind_spot_args) - set(valid_args)
            df = df.loc[valid_args]

        plt.figure(figsize=(14, 8))
        
        ax = df.T.plot(kind='bar', width=0.75, edgecolor='black', colormap='coolwarm')
        
        plt.title('Blind Spot Metrics Comparison', fontsize=20, fontweight='bold', pad=20)
        plt.xlabel('Metrics', fontsize=14, labelpad=15)
        plt.ylabel('Values', fontsize=14, labelpad=15)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Class', loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12, title_fontsize=14)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.savefig('blind_spot_metrics_plot.png', bbox_inches='tight', dpi=300)

        plt.show()
        plt.close()


    def _plot_by_meta_data(self, meta_data_args=None, save_path=None):
        """
        Create a bar plot visualization for metrics grouped by metadata.

        :param meta_data_args: (list, optional) List of specific metrics to plot. If None, all metrics will be plotted.
        :param save_path: (str, optional) Path where the plot should be saved. If None, saves as 'metrics_by_meta_data.png'.

        :return: None
        :raises AttributeError: If no valid metadata plot data is loaded.
        """      
        if not hasattr(self, 'plot_by_meta_data'):
            print("No valid 'plot_by_meta_data' data found in the JSON.")
            return

        data = self.plot_by_meta_data.get('data', {})
        df = pd.DataFrame(data).T
        
        if meta_data_args:
            df = df[meta_data_args]

        fig, ax = plt.subplots(figsize=(14, 10))

        df.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', color=sns.color_palette('rocket', n_colors=len(df.columns)))
        ax.set_title(f'Metrics for {", ".join(meta_data_args)}', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Metrics', fontsize=14)
        ax.set_ylabel('Values', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.savefig('metrics_by_meta_data.png', bbox_inches='tight', dpi=300)

        plt.show()
        plt.close()