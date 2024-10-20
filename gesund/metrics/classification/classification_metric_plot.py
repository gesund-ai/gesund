import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 

class Classification_Plot:
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
        with open(json_path, 'r') as file:
            data = json.load(file)
        return data
    
    def draw(self, plot_type, metrics=None, threshold=None, class_type='Average',
                graph_type='graph_1',roc_class='normal', pr_class='normal',
                confidence_histogram_args=None, overall_args=None, mixed_args=None,

                 save_path=None):
        
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
        
    def _plot_class_distributions(self, metrics, threshold, save_path):
        if not hasattr(self, 'class_data') or self.class_data.get('type') != 'bar':
            print("No valid 'bar' data found in the JSON.")
            return
        
        validation_data = self.class_data.get('data', {}).get('Validation', {})
        df = pd.DataFrame(list(validation_data.items()), columns=['Class', 'Count'])
        if metrics:
            df = df[df['Class'].isin(metrics)]
        if threshold:
            df = df[df['Count'] >= threshold]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Class', y='Count', hue='Count', data=df, palette='viridis')
        plt.title('Class Distribution', fontsize=16)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.savefig('class_distributions.png')
        plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _plot_blind_spot(self, class_type, save_path):
        if not hasattr(self, 'metrics_data'):
            print("No metrics data found.")
            return
        
        class_metrics = self.metrics_data.get(class_type, {})
        df = pd.DataFrame(list(class_metrics.items()), columns=['Metric', 'Value'])
        df = df[df['Metric'] != 'Sample Size']
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Value', y='Metric', hue='Metric', data=df, palette='viridis')
        plt.title(f'{class_type} Metrics', fontsize=16)
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Metric', fontsize=12)
        for i, v in enumerate(df['Value']):
            plt.text(v, i, f' {v:.4f}', va='center')
        plt.savefig(f'{class_type}_metrics.png')
        plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def _plot_class_performance_by_threshold(self, graph_type, metrics, threshold, save_path):
        if not self.performance_by_threshold:
            print("No performance threshold data found.")
            return
        
        performance_metrics = self.performance_by_threshold.get('data', {}).get(graph_type, {})
        df = pd.DataFrame(list(performance_metrics.items()), columns=['Metric', 'Value'])
        if metrics:
            df = df[df['Metric'].isin(metrics)]
        if threshold:
            df["Value"] = df["Value"].map(float)
            df = df[df['Value'] >= threshold]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Value', y='Metric', hue='Metric', data=df, palette='viridis')
        plt.title(f'{graph_type} Performance Metrics', fontsize=16)
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Metric', fontsize=12)
        for i, v in enumerate(df['Value']):
            plt.text(v, i, f' {v:.4f}', va='center')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(f'{graph_type}_performance_metrics.png')
        plt.show()
        plt.close()


    def _plot_roc_statistics(self, roc_class, save_path):
        if not hasattr(self, 'roc_statistics'):
            print("No ROC statistics data found.")
            return
        
        roc_data = self.roc_statistics.get('data', {}).get('points', {}).get(roc_class, [])
        df = pd.DataFrame(roc_data)
        
        plt.figure(figsize=(12, 8))
        area_val = round(float(self.roc_statistics["data"]["aucs"][roc_class]), 2)
        plt.plot(df['fpr'], df['tpr'], marker='o', linestyle='-', label='ROC curve (area = {})'.format(area_val))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {roc_class}')
        plt.legend(loc='lower right')
        plt.savefig(f'roc_curve_{roc_class}.png')
        plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _plot_precision_recall_statistics(self, pr_class, save_path):
        if not hasattr(self, 'precision_recall_statistics'):
            print("No Precision-Recall statistics data found.")
            return
        
        pr_data = self.precision_recall_statistics.get('data', {}).get('points', {}).get(pr_class, [])
        df = pd.DataFrame(pr_data)
        
        plt.figure(figsize=(12, 8))
        plt.plot(df['x'], df['y'], marker='o', linestyle='-', label=f'Precision-Recall curve (area = {self.precision_recall_statistics["data"]["aucs"][pr_class]:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {pr_class}')
        plt.legend(loc='lower left')
        plt.savefig(f'precision_recall_curve_{pr_class}.png')
        plt.show()

        if save_path:
            plt.savefig(save_path)
        plt.close()


    def _plot_confidence_histogram(self, confidence_histogram_args, save_path):
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
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='x', y='y', hue='labels', data=points_df, palette='viridis')
        plt.title('Scatter Plot of Points', fontsize=16)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.legend(title='Labels')
        plt.savefig('scatter_plot_points.png')
        plt.show()
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='category', y='value', hue='value', data=histogram_df, palette='viridis')
        plt.title('Histogram', fontsize=16)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.savefig('histogram.png')
        plt.show()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _plot_overall_metrics(self, overall_args, save_path):
        if not self.overall_json_data or self.overall_json_data.get('type') != 'overall':
            print("No valid 'overall' data found in the new JSON.")
            return
        
        data = self.overall_json_data.get('data', {})
        df = pd.DataFrame([(k, v['Validation']) for k, v in data.items()], columns=['Metric', 'Value'])
        
        if overall_args:
            if 'metrics' in overall_args:
                df = df[df['Metric'].isin(overall_args['metrics'])]
            if 'threshold' in overall_args:
                df = df[df['Value'] > overall_args['threshold']]
        
        df = df.sort_values('Value', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Value', y='Metric', hue='Metric', data=df, palette='viridis')
        plt.title('Overall Metrics', fontsize=16)
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Metric', fontsize=12)
        for i, v in enumerate(df['Value']):
            plt.text(v, i, f' {v:.4f}', va='center')
        plt.tight_layout()
        plt.savefig('overall_metrics.png')
        plt.show()

        if save_path:
            plt.savefig(save_path)
        plt.close()
