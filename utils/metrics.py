"""
Metrics tracking and evaluation utilities for YOLO models
Provides comprehensive metrics calculation and comparison
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import logging

# Import dependencies with graceful fallbacks
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("Warning: numpy is not installed. Install with: pip install numpy")
    HAS_NUMPY = False
    np = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("Warning: pandas is not installed. Install with: pip install pandas")
    HAS_PANDAS = False
    pd = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    print("Warning: matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
    HAS_PLOTTING = False
    plt = None
    sns = None

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    print("Warning: scikit-learn is not installed. Install with: pip install scikit-learn")
    HAS_SKLEARN = False
    confusion_matrix = None
    classification_report = None


@dataclass
class ModelMetrics:
    """Data class to store model evaluation metrics"""
    model_name: str
    dataset: str
    
    # Detection metrics
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Training metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    epochs_trained: int = 0
    
    # Performance metrics
    inference_time: float = 0.0  # ms per image
    model_size: float = 0.0  # MB
    flops: float = 0.0  # GFLOPs
    
    # Per-class metrics
    per_class_ap: Optional[Dict[str, float]] = None
    per_class_precision: Optional[Dict[str, float]] = None
    per_class_recall: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        """Create from dictionary"""
        return cls(**data)


class MetricsTracker:
    """Comprehensive metrics tracking and analysis"""
    
    def __init__(self, save_dir: str = "./results/metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("MetricsTracker")
        self.metrics_history: List[ModelMetrics] = []
        
    def add_metrics(self, metrics: ModelMetrics) -> None:
        """Add metrics to tracking history"""
        self.metrics_history.append(metrics)
        self.logger.info(f"Added metrics for {metrics.model_name}")
        
    def save_metrics(self, filename: str = "metrics_history.json") -> None:
        """Save metrics history to file"""
        save_path = self.save_dir / filename
        
        metrics_data = [metrics.to_dict() for metrics in self.metrics_history]
        
        with open(save_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
            
        self.logger.info(f"Metrics saved to {save_path}")
        
    def load_metrics(self, filename: str = "metrics_history.json") -> None:
        """Load metrics history from file"""
        load_path = self.save_dir / filename
        
        if not load_path.exists():
            self.logger.warning(f"Metrics file {load_path} not found")
            return
            
        with open(load_path, 'r') as f:
            metrics_data = json.load(f)
            
        self.metrics_history = [ModelMetrics.from_dict(data) for data in metrics_data]
        self.logger.info(f"Loaded {len(self.metrics_history)} metrics from {load_path}")
        
    def get_best_model(self, metric: str = "map50_95") -> Optional[ModelMetrics]:
        """Get best performing model based on specified metric"""
        if not self.metrics_history:
            return None
            
        return max(self.metrics_history, key=lambda x: getattr(x, metric, 0))
        
    def compare_models(self, metrics: List[str] = None) -> Optional[Any]:
        """Compare models across different metrics"""
        if not HAS_PANDAS:
            print("Warning: pandas not available. Cannot compare models.")
            return None
            
        if not self.metrics_history:
            return pd.DataFrame()
            
        if metrics is None:
            metrics = ['map50', 'map50_95', 'precision', 'recall', 'f1_score', 
                      'inference_time', 'model_size']
            
        data = []
        for model_metrics in self.metrics_history:
            row = {'model_name': model_metrics.model_name}
            for metric in metrics:
                row[metric] = getattr(model_metrics, metric, 0)
            data.append(row)
            
        return pd.DataFrame(data)
        
    def plot_comparison(self, metrics: List[str] = None, save_path: str = None) -> None:
        """Create comparison plots for different metrics"""
        if not self.metrics_history:
            self.logger.warning("No metrics to plot")
            return
            
        if metrics is None:
            metrics = ['map50', 'map50_95', 'precision', 'recall']
            
        df = self.compare_models(metrics)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):
            ax = axes[i]
            
            # Bar plot
            bars = ax.bar(df['model_name'], df[metric])
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
                       
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison plot saved to {save_path}")
        else:
            plt.savefig(self.save_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_training_curves(self, model_name: str, train_losses: List[float], 
                           val_losses: List[float], save_path: str = None) -> None:
        """Plot training and validation loss curves"""
        epochs = list(range(1, len(train_losses) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss', marker='o')
        plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
        
        plt.title(f'{model_name} - Training Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / f"{model_name}_training_curves.png", 
                       dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def create_performance_radar(self, save_path: str = None) -> None:
        """Create radar chart comparing model performance"""
        if len(self.metrics_history) < 2:
            self.logger.warning("Need at least 2 models for radar chart")
            return
            
        metrics = ['map50', 'map50_95', 'precision', 'recall', 'f1_score']
        
        # Normalize metrics to 0-1 scale for radar chart
        df = self.compare_models(metrics)
        
        # Set up radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.metrics_history)))
        
        for i, (_, row) in enumerate(df.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=row['model_name'], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / "performance_radar.png", 
                       dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def generate_report(self, save_path: str = None) -> str:
        """Generate comprehensive metrics report"""
        if not self.metrics_history:
            return "No metrics available for report generation."
            
        report = []
        report.append("YOLO MODEL COMPARISON REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        df = self.compare_models()
        report.append("SUMMARY STATISTICS")
        report.append("-" * 30)
        
        for metric in ['map50', 'map50_95', 'precision', 'recall', 'f1_score']:
            if metric in df.columns:
                best_model = df.loc[df[metric].idxmax(), 'model_name']
                best_value = df[metric].max()
                report.append(f"Best {metric.upper()}: {best_model} ({best_value:.4f})")
                
        report.append("")
        
        # Individual model details
        report.append("INDIVIDUAL MODEL PERFORMANCE")
        report.append("-" * 40)
        
        for metrics in self.metrics_history:
            report.append(f"\n{metrics.model_name.upper()}")
            report.append(f"  mAP@0.5: {metrics.map50:.4f}")
            report.append(f"  mAP@0.5:0.95: {metrics.map50_95:.4f}")
            report.append(f"  Precision: {metrics.precision:.4f}")
            report.append(f"  Recall: {metrics.recall:.4f}")
            report.append(f"  F1-Score: {metrics.f1_score:.4f}")
            report.append(f"  Inference Time: {metrics.inference_time:.2f} ms")
            report.append(f"  Model Size: {metrics.model_size:.2f} MB")
            
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {save_path}")
        else:
            report_path = self.save_dir / "comparison_report.txt"
            with open(report_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {report_path}")
            
        return report_text


def calculate_detection_metrics(predictions: List, ground_truth: List, 
                              iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate detection metrics from predictions and ground truth
    
    Args:
        predictions: List of prediction results
        ground_truth: List of ground truth annotations
        iou_threshold: IoU threshold for positive detection
        
    Returns:
        Dictionary containing calculated metrics
    """
    # This is a simplified version - in practice, you'd use proper
    # detection evaluation libraries like pycocotools
    
    tp = fp = fn = 0
    
    # Simplified calculation (replace with proper implementation)
    for pred, gt in zip(predictions, ground_truth):
        # Calculate IoU and determine TP, FP, FN
        # This is pseudo-code - implement proper IoU calculation
        pass
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


# Example usage
if __name__ == "__main__":
    # Create metrics tracker
    tracker = MetricsTracker()
    
    # Add sample metrics
    metrics1 = ModelMetrics(
        model_name="YOLOv8n",
        dataset="COCO",
        map50=0.85,
        map50_95=0.65,
        precision=0.82,
        recall=0.78,
        f1_score=0.80,
        inference_time=15.2,
        model_size=6.2
    )
    
    metrics2 = ModelMetrics(
        model_name="YOLOv9c",
        dataset="COCO",
        map50=0.88,
        map50_95=0.68,
        precision=0.85,
        recall=0.81,
        f1_score=0.83,
        inference_time=22.1,
        model_size=25.3
    )
    
    tracker.add_metrics(metrics1)
    tracker.add_metrics(metrics2)
    
    # Generate comparison
    comparison_df = tracker.compare_models()
    print(comparison_df)
    
    # Generate report
    report = tracker.generate_report()
    print(report)
