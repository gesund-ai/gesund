import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 

class Semantic_Segmentation_Plot:
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
        with open(json_path, 'r') as file:
            data = json.load(file)
        return data

    def draw(self, plot_type, metrics=None, threshold=None, classbased_table_args=None, overall_args=None, 
                        blind_spot_args=None, meta_data_args=None, save_path=None):
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
        
    def _plot_violin_graph(self, metrics, threshold, save_path):
        if not hasattr(self, 'violin_data') or self.violin_data.get('type') != 'violin':
            print("No valid 'violin' data found in the JSON.")
            return
        
        data = self.violin_data.get('data', {})
        df = pd.DataFrame(data)
        
        if metrics:
            df = df[metrics]
        if threshold:
            df = df[df > threshold]
        
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=df, palette='viridis')
        plt.title('Violin Plot of Metrics', fontsize=16)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.savefig('violin_plot.png')
        plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()


    def _plot_classbased_table(self, classbased_table_args, save_path):
        if not hasattr(self, 'classbased_table') or self.classbased_table.get('type') != 'table':
            print("No valid 'table' data found in the JSON.")
            return

        data = self.classbased_table.get('data', {})
        
        flattened_data = {}
        for category, metrics in data.items():
            for subcategory, values in metrics.items():
                flattened_data[f"{category}_{subcategory}"] = values

        df = pd.DataFrame(flattened_data).T

        if classbased_table_args and isinstance(classbased_table_args, float):
            df = df[df > classbased_table_args]

        plt.figure(figsize=(15, 10))
        sns.heatmap(df, annot=True, cmap='viridis', fmt='.3f', cbar=True)
        plt.title('Class-based Table Plot of Metrics', fontsize=16)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Categories', fontsize=12)
        plt.tight_layout()
        plt.savefig('classbased_table_plot.png')
        plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _plot_overall_data(self, overall_args, save_path):
        if not hasattr(self, 'overall_data') or self.overall_data.get('type') != 'overall':
            print("No valid 'overall' data found in the JSON.")
            return

        data = self.overall_data.get('data', {})
        df = pd.DataFrame({k: v['Validation'] for k, v in data.items()}, index=[0])
        
        if overall_args:
            df = df[overall_args]

        plt.figure(figsize=(12, 6))
        df.T.plot(kind='bar')
        plt.title('Overall Metrics', fontsize=16)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('overall_metrics_plot.png')
        plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _plot_blind_spot(self, blind_spot_args, save_path):
        if not hasattr(self, 'blind_spot_data'):
            print("No valid 'blind_spot' data found in the JSON.")
            return

        df = pd.DataFrame(self.blind_spot_data)
        
        if blind_spot_args:
            df = df.loc[blind_spot_args]

        plt.figure(figsize=(12, 6))
        ax = df.T.plot(kind='bar', width=0.8)
        plt.title('Blind Spot Metrics Comparison', fontsize=16)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Class', loc='upper left', bbox_to_anchor=(1, 1))
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        
        plt.tight_layout()
        plt.savefig('blind_spot_metrics_plot.png')
        plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _plot_by_meta_data(self, meta_data_args, save_path):
        if not hasattr(self, 'plot_by_meta_data'):
            print("No valid 'plot_by_meta_data' data found in the JSON.")
            return

        data = self.plot_by_meta_data.get('data', {})
        df = pd.DataFrame(data).T
        
        if meta_data_args:
            df = df[meta_data_args]

        plt.figure(figsize=(12, 6))
        df.plot(kind='bar')
        plt.title('New Data Metrics', fontsize=16)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('metrics_by_meta_data.png')
        plt.show()    
        if save_path:
            plt.savefig(save_path)
        plt.close()