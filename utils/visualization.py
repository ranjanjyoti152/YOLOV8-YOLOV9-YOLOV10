"""
Visualization utilities for YOLO training project
Provides comprehensive visualization tools for data, training, and results
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import logging

# Import dependencies with graceful fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    print("Warning: matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
    HAS_PLOTTING = False
    plt = None
    patches = None
    sns = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("Warning: numpy is not installed. Install with: pip install numpy")
    HAS_NUMPY = False
    np = None

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    print("Warning: opencv-python is not installed. Install with: pip install opencv-python")
    HAS_OPENCV = False
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    print("Warning: Pillow is not installed. Install with: pip install Pillow")
    HAS_PIL = False
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("Warning: pandas is not installed. Install with: pip install pandas")
    HAS_PANDAS = False
    pd = None


class VisualizationManager:
    """Comprehensive visualization manager for YOLO training"""
    
    def __init__(self, save_dir: str = "./results/visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("VisualizationManager")
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_data_distribution(self, class_counts: Dict[str, int], 
                             title: str = "Class Distribution", 
                             save_name: str = "class_distribution.png") -> None:
        """Plot dataset class distribution"""
        
        plt.figure(figsize=(12, 8))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Create bar plot
        bars = plt.bar(classes, counts, color=sns.color_palette("viridis", len(classes)))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Classes', fontsize=12)
        plt.ylabel('Number of Instances', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Class distribution plot saved to {save_name}")
        
    def plot_training_progress(self, train_losses: List[float], 
                             val_losses: List[float],
                             train_metrics: Dict[str, List[float]] = None,
                             val_metrics: Dict[str, List[float]] = None,
                             save_name: str = "training_progress.png") -> None:
        """Plot comprehensive training progress"""
        
        epochs = list(range(1, len(train_losses) + 1))
        
        # Determine subplot layout
        n_plots = 2  # Loss plots
        if train_metrics:
            n_plots += len(train_metrics)
            
        rows = (n_plots + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
        
        if rows == 1:
            axes = [axes]
        axes = axes.flatten()
        
        # Plot losses
        axes[0].plot(epochs, train_losses, label='Training Loss', marker='o', linewidth=2)
        axes[0].plot(epochs, val_losses, label='Validation Loss', marker='s', linewidth=2)
        axes[0].set_title('Training and Validation Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot additional metrics
        plot_idx = 1
        if train_metrics and val_metrics:
            for metric_name in train_metrics.keys():
                if plot_idx < len(axes):
                    ax = axes[plot_idx]
                    ax.plot(epochs, train_metrics[metric_name], 
                           label=f'Training {metric_name}', marker='o', linewidth=2)
                    ax.plot(epochs, val_metrics[metric_name], 
                           label=f'Validation {metric_name}', marker='s', linewidth=2)
                    ax.set_title(f'{metric_name.capitalize()}', fontweight='bold')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(metric_name)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Training progress plot saved to {save_name}")
        
    def visualize_predictions(self, image_path: str, predictions: List[Dict],
                            ground_truth: List[Dict] = None,
                            class_names: List[str] = None,
                            save_name: str = None) -> None:
        """Visualize model predictions on an image"""
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2 if ground_truth else 1, figsize=(15, 8))
        
        if ground_truth:
            if isinstance(axes, np.ndarray):
                ax_pred, ax_gt = axes
            else:
                ax_pred = axes
                ax_gt = None
        else:
            ax_pred = axes if not isinstance(axes, np.ndarray) else axes[0]
            ax_gt = None
        
        # Plot predictions
        ax_pred.imshow(image)
        ax_pred.set_title('Predictions', fontweight='bold')
        ax_pred.axis('off')
        
        for pred in predictions:
            self._draw_bbox(ax_pred, pred, class_names, color='red')
        
        # Plot ground truth if available
        if ground_truth and ax_gt is not None:
            ax_gt.imshow(image)
            ax_gt.set_title('Ground Truth', fontweight='bold')
            ax_gt.axis('off')
            
            for gt in ground_truth:
                self._draw_bbox(ax_gt, gt, class_names, color='green')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        else:
            img_name = Path(image_path).stem
            plt.savefig(self.save_dir / f"{img_name}_predictions.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def _draw_bbox(self, ax, bbox_info: Dict, class_names: List[str] = None, 
                   color: str = 'red') -> None:
        """Draw bounding box on axis"""
        
        x, y, w, h = bbox_info['bbox']
        class_id = bbox_info.get('class_id', 0)
        confidence = bbox_info.get('confidence', 1.0)
        
        # Create rectangle
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {confidence:.2f}"
        else:
            label = f"Class {class_id}: {confidence:.2f}"
            
        ax.text(x, y - 5, label, color=color, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int],
                            class_names: List[str] = None,
                            save_name: str = "confusion_matrix.png") -> None:
        """Plot confusion matrix"""
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names or range(len(cm)),
                   yticklabels=class_names or range(len(cm)))
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Confusion matrix saved to {save_name}")
        
    def plot_pr_curve(self, precision: List[float], recall: List[float],
                     ap_score: float = None, 
                     save_name: str = "pr_curve.png") -> None:
        """Plot Precision-Recall curve"""
        
        plt.figure(figsize=(10, 8))
        
        plt.plot(recall, precision, linewidth=3, color='blue')
        plt.fill_between(recall, precision, alpha=0.3, color='blue')
        
        if ap_score is not None:
            plt.title(f'Precision-Recall Curve (AP = {ap_score:.3f})', 
                     fontsize=16, fontweight='bold')
        else:
            plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
            
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"PR curve saved to {save_name}")
        
    def plot_model_comparison(self, comparison_data: Optional[Any],
                            metrics: List[str] = None,
                            save_name: str = "model_comparison.png") -> None:
        """Plot model comparison across multiple metrics"""
        
        if not HAS_PLOTTING or not HAS_PANDAS:
            print("Warning: matplotlib/seaborn/pandas not available. Cannot plot model comparison.")
            return
            
        if comparison_data is None:
            return
        
        if metrics is None:
            metrics = ['map50', 'map50_95', 'precision', 'recall', 'f1_score']
            
        # Filter available metrics
        available_metrics = [m for m in metrics if m in comparison_data.columns]
        
        n_metrics = len(available_metrics)
        rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
        
        if rows == 1:
            axes = [axes] if n_metrics > 1 else [axes, None]
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            bars = ax.bar(comparison_data['model_name'], comparison_data[metric],
                         color=sns.color_palette("viridis", len(comparison_data)))
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{metric.upper()} Comparison', fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            if axes[i] is not None:
                axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Model comparison plot saved to {save_name}")
        
    def create_training_summary(self, model_name: str, 
                              final_metrics: Dict[str, float],
                              training_time: str,
                              save_name: str = None) -> None:
        """Create a visual training summary"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Metrics bar chart
        metrics_names = list(final_metrics.keys())
        metrics_values = list(final_metrics.values())
        
        bars = ax1.bar(metrics_names, metrics_values, 
                      color=sns.color_palette("viridis", len(metrics_names)))
        ax1.set_title('Final Metrics', fontweight='bold')
        ax1.set_ylabel('Value')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Training info text
        ax2.text(0.1, 0.8, f"Model: {model_name}", fontsize=14, fontweight='bold')
        ax2.text(0.1, 0.6, f"Training Time: {training_time}", fontsize=12)
        ax2.text(0.1, 0.4, f"Best mAP@0.5: {final_metrics.get('map50', 0):.3f}", fontsize=12)
        ax2.text(0.1, 0.2, f"Best mAP@0.5:0.95: {final_metrics.get('map50_95', 0):.3f}", fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Training Summary', fontweight='bold')
        
        # Placeholder for additional plots
        ax3.text(0.5, 0.5, 'Additional metrics\ncan be added here', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('Additional Analysis', fontweight='bold')
        ax3.axis('off')
        
        ax4.text(0.5, 0.5, 'Model architecture\nor performance details', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('Model Details', fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / f"{model_name}_summary.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        
        self.logger.info(f"Training summary saved for {model_name}")


def create_dataset_preview(image_paths: List[str], annotations: List[Dict],
                         class_names: List[str], save_path: str = None) -> None:
    """Create a preview grid of dataset images with annotations"""
    
    n_images = min(len(image_paths), 9)  # Show up to 9 images
    rows = int(np.ceil(np.sqrt(n_images)))
    cols = int(np.ceil(n_images / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(n_images):
        ax = axes[i] if isinstance(axes, (list, np.ndarray)) else axes
        
        # Load and display image
        image = cv2.imread(image_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        
        # Draw annotations if available
        if i < len(annotations):
            for ann in annotations[i]:
                x, y, w, h = ann['bbox']
                class_id = ann['class_id']
                
                rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                       edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                if class_id < len(class_names):
                    ax.text(x, y - 5, class_names[class_id], color='red', 
                           fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='white', alpha=0.8))
        
        ax.set_title(f'Image {i+1}', fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    if isinstance(axes, (list, np.ndarray)):
        for i in range(n_images, len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create visualization manager
    viz_manager = VisualizationManager()
    
    # Example class distribution
    class_counts = {
        'person': 1500,
        'car': 800,
        'bicycle': 400,
        'dog': 300,
        'cat': 250
    }
    
    viz_manager.plot_data_distribution(class_counts)
    
    # Example training progress
    train_losses = [0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25]
    val_losses = [0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38, 0.35]
    
    viz_manager.plot_training_progress(train_losses, val_losses)
