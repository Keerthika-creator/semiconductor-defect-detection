"""
Visualization Utilities for Semiconductor Defect Detection

This module provides comprehensive visualization tools for:
1. Training progress analysis
2. Model performance evaluation
3. Confusion matrix visualization
4. Defect distribution analysis
5. Inference results display
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import json


class DefectVisualization:
    """Comprehensive visualization tools for defect detection system"""
    
    def __init__(self, output_dir: str = '/content/output'):
        """
        Initialize visualization tools
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_training_history(self, history: Dict, model_name: str, 
                             save: bool = True):
        """
        Plot training history (accuracy and loss curves)
        
        Args:
            history: Training history dictionary
            model_name: Name of the model ('wafer', 'die', 'router')
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        if 'accuracy' in history:
            axes[0, 0].plot(history['accuracy'], label='Training', linewidth=2)
            axes[0, 0].plot(history['val_accuracy'], label='Validation', linewidth=2)
            axes[0, 0].set_title(f'{model_name.upper()} - Accuracy Over Epochs', 
                                fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        if 'loss' in history:
            axes[0, 1].plot(history['loss'], label='Training', linewidth=2)
            axes[0, 1].plot(history['val_loss'], label='Validation', linewidth=2)
            axes[0, 1].set_title(f'{model_name.upper()} - Loss Over Epochs', 
                                fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Training', linewidth=2)
            axes[1, 0].plot(history['val_precision'], label='Validation', linewidth=2)
            axes[1, 0].set_title(f'{model_name.upper()} - Precision Over Epochs', 
                                fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Training', linewidth=2)
            axes[1, 1].plot(history['val_recall'], label='Validation', linewidth=2)
            axes[1, 1].set_title(f'{model_name.upper()} - Recall Over Epochs', 
                                fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'{model_name}_training_history.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history saved: {save_path}")
        
        plt.show()
    
    def plot_combined_training(self, wafer_history: Dict, die_history: Dict, 
                              save: bool = True):
        """
        Plot combined training history for both models
        
        Args:
            wafer_history: Wafer model training history
            die_history: Die model training history
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy comparison
        axes[0, 0].plot(wafer_history['accuracy'], label='Wafer (Train)', 
                       linewidth=2, linestyle='-')
        axes[0, 0].plot(wafer_history['val_accuracy'], label='Wafer (Val)', 
                       linewidth=2, linestyle='--')
        axes[0, 0].plot(die_history['accuracy'], label='Die (Train)', 
                       linewidth=2, linestyle='-')
        axes[0, 0].plot(die_history['val_accuracy'], label='Die (Val)', 
                       linewidth=2, linestyle='--')
        axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss comparison
        axes[0, 1].plot(wafer_history['loss'], label='Wafer (Train)', 
                       linewidth=2, linestyle='-')
        axes[0, 1].plot(wafer_history['val_loss'], label='Wafer (Val)', 
                       linewidth=2, linestyle='--')
        axes[0, 1].plot(die_history['loss'], label='Die (Train)', 
                       linewidth=2, linestyle='-')
        axes[0, 1].plot(die_history['val_loss'], label='Die (Val)', 
                       linewidth=2, linestyle='--')
        axes[0, 1].set_title('Loss Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision comparison
        if 'precision' in wafer_history and 'precision' in die_history:
            axes[1, 0].plot(wafer_history['val_precision'], label='Wafer', 
                           linewidth=2)
            axes[1, 0].plot(die_history['val_precision'], label='Die', 
                           linewidth=2)
            axes[1, 0].set_title('Validation Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall comparison
        if 'recall' in wafer_history and 'recall' in die_history:
            axes[1, 1].plot(wafer_history['val_recall'], label='Wafer', 
                           linewidth=2)
            axes[1, 1].plot(die_history['val_recall'], label='Die', 
                           linewidth=2)
            axes[1, 1].set_title('Validation Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'combined_training_history.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Combined training history saved: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix_detailed(self, cm: np.ndarray, classes: List[str], 
                                      model_name: str, save: bool = True):
        """
        Plot detailed confusion matrix with percentages
        
        Args:
            cm: Confusion matrix
            classes: Class labels
            model_name: Model name
            save: Whether to save the plot
        """
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Count'}, ax=axes[0])
        axes[0].set_title(f'{model_name.upper()} - Confusion Matrix (Counts)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn', 
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Percentage (%)'}, ax=axes[1])
        axes[1].set_title(f'{model_name.upper()} - Confusion Matrix (Percentages)', 
                         fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 
                                    f'{model_name}_confusion_matrix_detailed.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Detailed confusion matrix saved: {save_path}")
        
        plt.show()
    
    def plot_class_performance(self, classification_report: Dict, model_name: str,
                              save: bool = True):
        """
        Plot per-class performance metrics
        
        Args:
            classification_report: Scikit-learn classification report dict
            model_name: Model name
            save: Whether to save the plot
        """
        # Extract class metrics
        classes = [k for k in classification_report.keys() 
                  if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        metrics = {
            'Precision': [],
            'Recall': [],
            'F1-Score': []
        }
        
        for cls in classes:
            metrics['Precision'].append(classification_report[cls]['precision'])
            metrics['Recall'].append(classification_report[cls]['recall'])
            metrics['F1-Score'].append(classification_report[cls]['f1-score'])
        
        # Create DataFrame
        df = pd.DataFrame(metrics, index=classes)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{model_name.upper()} - Per-Class Performance Metrics', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for i, cls in enumerate(classes):
            for j, metric in enumerate(['Precision', 'Recall', 'F1-Score']):
                height = df[metric][i]
                ax.text(i + (j-1)*width, height + 0.02, f'{height:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 
                                    f'{model_name}_class_performance.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Class performance plot saved: {save_path}")
        
        plt.show()
    
    def plot_inference_latency_distribution(self, latencies: List[float], 
                                           model_name: str, save: bool = True):
        """
        Plot inference latency distribution
        
        Args:
            latencies: List of inference times in milliseconds
            model_name: Model name
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        axes[0].hist(latencies, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(np.mean(latencies), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(latencies):.2f} ms')
        axes[0].axvline(np.median(latencies), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(latencies):.2f} ms')
        axes[0].set_xlabel('Latency (ms)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{model_name.upper()} - Inference Latency Distribution', 
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(latencies, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('Latency (ms)')
        axes[1].set_title(f'{model_name.upper()} - Latency Box Plot', 
                         fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f"Min: {np.min(latencies):.2f} ms\n"
        stats_text += f"Q1: {np.percentile(latencies, 25):.2f} ms\n"
        stats_text += f"Median: {np.median(latencies):.2f} ms\n"
        stats_text += f"Q3: {np.percentile(latencies, 75):.2f} ms\n"
        stats_text += f"Max: {np.max(latencies):.2f} ms\n"
        stats_text += f"Std: {np.std(latencies):.2f} ms"
        
        axes[1].text(1.15, np.median(latencies), stats_text,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 
                                    f'{model_name}_latency_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Latency distribution saved: {save_path}")
        
        plt.show()
    
    def plot_confidence_distribution(self, confidences: List[float], 
                                    predictions: List[str],
                                    model_name: str, save: bool = True):
        """
        Plot confidence score distribution by prediction class
        
        Args:
            confidences: List of confidence scores
            predictions: List of predicted classes
            model_name: Model name
            save: Whether to save the plot
        """
        # Create DataFrame
        df = pd.DataFrame({
            'Confidence': confidences,
            'Prediction': predictions
        })
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Overall distribution
        axes[0].hist(confidences, bins=50, alpha=0.7, color='lightgreen', 
                    edgecolor='black')
        axes[0].axvline(0.6, color='orange', linestyle='--', linewidth=2,
                       label='Confidence Threshold (0.6)')
        axes[0].axvline(0.95, color='red', linestyle='--', linewidth=2,
                       label='Early Exit Threshold (0.95)')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{model_name.upper()} - Overall Confidence Distribution', 
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Per-class violin plot
        unique_classes = df['Prediction'].unique()
        if len(unique_classes) <= 10:  # Only plot if reasonable number of classes
            sns.violinplot(data=df, x='Prediction', y='Confidence', ax=axes[1])
            axes[1].set_xlabel('Predicted Class')
            axes[1].set_ylabel('Confidence Score')
            axes[1].set_title(f'{model_name.upper()} - Confidence by Class', 
                             fontsize=14, fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].axhline(0.6, color='orange', linestyle='--', linewidth=1, alpha=0.5)
            axes[1].axhline(0.95, color='red', linestyle='--', linewidth=1, alpha=0.5)
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 
                                    f'{model_name}_confidence_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confidence distribution saved: {save_path}")
        
        plt.show()
    
    def create_dashboard(self, results: Dict, save: bool = True):
        """
        Create comprehensive evaluation dashboard
        
        Args:
            results: Dictionary containing all evaluation results
            save: Whether to save the plot
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Semiconductor Defect Detection - Evaluation Dashboard', 
                    fontsize=18, fontweight='bold')
        
        # 1. Overall Accuracy Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        stages = ['Wafer', 'Die']
        accuracies = [results.get('wafer', {}).get('accuracy', 0),
                     results.get('die', {}).get('accuracy', 0)]
        ax1.bar(stages, accuracies, color=['#3498db', '#2ecc71'], alpha=0.8)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 2. Precision-Recall-F1 Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['Precision', 'Recall', 'F1-Score']
        wafer_metrics = [results.get('wafer', {}).get('precision', 0),
                        results.get('wafer', {}).get('recall', 0),
                        results.get('wafer', {}).get('f1_score', 0)]
        die_metrics = [results.get('die', {}).get('precision', 0),
                      results.get('die', {}).get('recall', 0),
                      results.get('die', {}).get('f1_score', 0)]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax2.bar(x - width/2, wafer_metrics, width, label='Wafer', alpha=0.8)
        ax2.bar(x + width/2, die_metrics, width, label='Die', alpha=0.8)
        ax2.set_ylabel('Score')
        ax2.set_title('Metrics Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Performance Metrics
        ax3 = fig.add_subplot(gs[0, 2])
        perf_labels = ['Latency\n(ms)', 'Throughput\n(FPS)', 'Model Size\n(MB)']
        perf_values = [
            results.get('performance', {}).get('cpu_inference', {}).get('mean_ms', 0),
            results.get('performance', {}).get('cpu_inference', {}).get('fps', 0),
            results.get('performance', {}).get('model_sizes_mb', {}).get('total', 0)
        ]
        colors = ['#e74c3c', '#f39c12', '#9b59b6']
        ax3.bar(perf_labels, perf_values, color=colors, alpha=0.8)
        ax3.set_title('Performance Metrics', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(perf_values):
            ax3.text(i, v + max(perf_values)*0.02, f'{v:.1f}', 
                    ha='center', fontweight='bold')
        
        # 4-6. Confusion Matrices (if available)
        if 'wafer' in results and 'confusion_matrix' in results['wafer']:
            ax4 = fig.add_subplot(gs[1, :])
            cm = np.array(results['wafer']['confusion_matrix'])
            classes = results['wafer'].get('classes', 
                                          [f'C{i}' for i in range(len(cm))])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes, ax=ax4)
            ax4.set_title('Wafer Classifier - Confusion Matrix', fontweight='bold')
            ax4.set_ylabel('True Label')
            ax4.set_xlabel('Predicted Label')
        
        # 7-9. Additional metrics
        ax7 = fig.add_subplot(gs[2, :])
        summary_text = "System Summary:\n\n"
        summary_text += f"✓ Wafer Accuracy: {results.get('wafer', {}).get('accuracy', 0):.4f}\n"
        summary_text += f"✓ Die Accuracy: {results.get('die', {}).get('accuracy', 0):.4f}\n"
        summary_text += f"✓ Average Latency: {results.get('performance', {}).get('cpu_inference', {}).get('mean_ms', 0):.2f} ms\n"
        summary_text += f"✓ Real-time FPS: {results.get('performance', {}).get('cpu_inference', {}).get('fps', 0):.2f}\n"
        summary_text += f"✓ Total Model Size: {results.get('performance', {}).get('model_sizes_mb', {}).get('total', 0):.2f} MB\n"
        summary_text += f"\nDeployment Status: {'✅ Ready for Edge' if results.get('performance', {}).get('cpu_inference', {}).get('fps', 0) > 10 else '⚠️ Optimization Needed'}"
        
        ax7.text(0.5, 0.5, summary_text, ha='center', va='center', 
                fontsize=14, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax7.axis('off')
        
        if save:
            save_path = os.path.join(self.output_dir, 'evaluation_dashboard.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Dashboard saved: {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print("\nAvailable visualization functions:")
    print("  - plot_training_history()")
    print("  - plot_combined_training()")
    print("  - plot_confusion_matrix_detailed()")
    print("  - plot_class_performance()")
    print("  - plot_inference_latency_distribution()")
    print("  - plot_confidence_distribution()")
    print("  - create_dashboard()")
