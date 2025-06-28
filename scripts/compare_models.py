#!/usr/bin/env python3
"""
Model comparison and benchmarking script
Comprehensive comparison of YOLOv8, YOLOv9, and YOLOv10 models
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Optional imports with error handling
try:
    import numpy as np
except ImportError:
    print("Warning: NumPy not installed. Install with: pip install numpy")
    np = None

try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not installed. Install with: pip install pandas")
    pd = None

try:
    import torch
except ImportError:
    print("Warning: PyTorch not installed. Install with: pip install torch")
    torch = None

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.metrics import MetricsTracker, ModelMetrics
from utils.visualization import VisualizationManager


class ModelComparator:
    """Comprehensive model comparison and benchmarking"""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Check dependencies
        if pd is None:
            print("Warning: pandas not available. Some comparison features will be limited.")
        if np is None:
            print("Warning: numpy not available. Some statistical features will be limited.")
        
        self.logger = setup_logger("ModelComparator")
        self.metrics_tracker = MetricsTracker()
        self.viz_manager = VisualizationManager()
        
    def load_training_results(self, experiment_dirs: List[str]) -> List[ModelMetrics]:
        """Load training results from experiment directories"""
        all_metrics = []
        
        for exp_dir in experiment_dirs:
            exp_path = Path(exp_dir)
            
            if not exp_path.exists():
                self.logger.warning(f"Experiment directory {exp_dir} not found")
                continue
                
            # Try to load metrics from different sources
            metrics = self._load_experiment_metrics(exp_path)
            if metrics:
                all_metrics.append(metrics)
                
        return all_metrics
        
    def _load_experiment_metrics(self, exp_path: Path) -> Optional[ModelMetrics]:
        """Load metrics from a single experiment directory"""
        
        # Try to find results.json or similar files
        possible_files = [
            exp_path / "results.json",
            exp_path / "metrics.json",
            exp_path / "val_results.json"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    # Extract model name from path or data
                    model_name = data.get('model_name', exp_path.name)
                    
                    # Create ModelMetrics object
                    metrics = ModelMetrics(
                        model_name=model_name,
                        dataset=data.get('dataset', 'unknown'),
                        map50=data.get('map50', data.get('metrics/mAP_0.5', 0.0)),
                        map50_95=data.get('map50_95', data.get('metrics/mAP_0.5:0.95', 0.0)),
                        precision=data.get('precision', data.get('metrics/precision', 0.0)),
                        recall=data.get('recall', data.get('metrics/recall', 0.0)),
                        f1_score=data.get('f1_score', 0.0),
                        train_loss=data.get('train_loss', 0.0),
                        val_loss=data.get('val_loss', 0.0),
                        epochs_trained=data.get('epochs_trained', data.get('epoch', 0)),
                        inference_time=data.get('inference_time', 0.0),
                        model_size=data.get('model_size', 0.0)
                    )
                    
                    return metrics
                    
                except Exception as e:
                    self.logger.warning(f"Error loading metrics from {file_path}: {e}")
                    continue
                    
        # Try to extract from YOLOv8 results
        weights_dir = exp_path / "weights"
        if weights_dir.exists():
            return self._extract_yolo_metrics(exp_path)
            
        return None
        
    def _extract_yolo_metrics(self, exp_path: Path) -> Optional[ModelMetrics]:
        """Extract metrics from YOLO training outputs"""
        
        # Look for results.csv (YOLOv8/v10 format)
        results_csv = exp_path / "results.csv"
        if results_csv.exists() and pd is not None:
            try:
                df = pd.read_csv(results_csv)
                
                # Get best epoch metrics
                if 'val/mAP50' in df.columns:
                    best_idx = df['val/mAP50'].idxmax()
                    
                    metrics = ModelMetrics(
                        model_name=exp_path.name,
                        dataset='training_dataset',
                        map50=df.loc[best_idx, 'val/mAP50'],
                        map50_95=df.loc[best_idx, 'val/mAP50-95'] if 'val/mAP50-95' in df.columns else 0.0,
                        precision=df.loc[best_idx, 'precision'] if 'precision' in df.columns else 0.0,
                        recall=df.loc[best_idx, 'recall'] if 'recall' in df.columns else 0.0,
                        train_loss=df.loc[best_idx, 'train/box_loss'] if 'train/box_loss' in df.columns else 0.0,
                        val_loss=df.loc[best_idx, 'val/box_loss'] if 'val/box_loss' in df.columns else 0.0,
                        epochs_trained=len(df)
                    )
                    
                    return metrics
                    
            except Exception as e:
                self.logger.warning(f"Error parsing results.csv: {e}")
        elif results_csv.exists() and pd is None:
            self.logger.warning("pandas not available, cannot parse results.csv")
                
        return None
        
    def benchmark_inference_speed(self, model_paths: List[str], 
                                test_images: List[str],
                                warmup_runs: int = 10,
                                benchmark_runs: int = 100) -> Dict[str, Dict]:
        """Benchmark inference speed for different models"""
        
        benchmark_results = {}
        
        for model_path in model_paths:
            model_path = Path(model_path)
            
            if not model_path.exists():
                self.logger.warning(f"Model {model_path} not found")
                continue
                
            model_name = model_path.stem
            self.logger.info(f"Benchmarking {model_name}...")
            
            try:
                # Load model based on type
                if 'yolov8' in model_name.lower() or 'yolov10' in model_name.lower():
                    try:
                        from ultralytics import YOLO
                        model = YOLO(str(model_path))
                    except ImportError:
                        self.logger.warning(f"ultralytics not available for {model_name}")
                        continue
                else:
                    self.logger.warning(f"Unsupported model type for {model_name}")
                    continue
                
                # Warmup
                self.logger.info(f"Warming up {model_name}...")
                for i in range(warmup_runs):
                    if i < len(test_images):
                        _ = model(test_images[i % len(test_images)], verbose=False)
                
                # Benchmark
                self.logger.info(f"Benchmarking {model_name}...")
                inference_times = []
                
                for i in range(benchmark_runs):
                    image_path = test_images[i % len(test_images)]
                    
                    start_time = time.time()
                    _ = model(image_path, verbose=False)
                    end_time = time.time()
                    
                    inference_times.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Calculate statistics
                benchmark_results[model_name] = {
                    'mean_inference_time': np.mean(inference_times),
                    'std_inference_time': np.std(inference_times),
                    'min_inference_time': np.min(inference_times),
                    'max_inference_time': np.max(inference_times),
                    'median_inference_time': np.median(inference_times),
                    'fps': 1000 / np.mean(inference_times)
                }
                
                self.logger.info(f"{model_name} benchmark completed: "
                               f"{benchmark_results[model_name]['mean_inference_time']:.2f}ms avg")
                
            except Exception as e:
                self.logger.error(f"Error benchmarking {model_name}: {e}")
                continue
                
        return benchmark_results
        
    def calculate_model_complexity(self, model_paths: List[str]) -> Dict[str, Dict]:
        """Calculate model complexity metrics (parameters, FLOPs, size)"""
        
        complexity_results = {}
        
        for model_path in model_paths:
            model_path = Path(model_path)
            
            if not model_path.exists():
                continue
                
            model_name = model_path.stem
            
            try:
                # Get file size
                file_size_mb = model_path.stat().st_size / (1024 * 1024)
                
                # Load model to get parameters
                if 'yolov8' in model_name.lower() or 'yolov10' in model_name.lower():
                    from ultralytics import YOLO
                    model = YOLO(str(model_path))
                    
                    # Get model info
                    model_info = model.info(verbose=False)
                    
                    complexity_results[model_name] = {
                        'file_size_mb': file_size_mb,
                        'parameters': model_info.get('parameters', 0),
                        'flops': model_info.get('GFLOPs', 0),
                        'layers': model_info.get('layers', 0)
                    }
                else:
                    # For other models, just get file size
                    complexity_results[model_name] = {
                        'file_size_mb': file_size_mb,
                        'parameters': 0,
                        'flops': 0,
                        'layers': 0
                    }
                
            except Exception as e:
                self.logger.error(f"Error analyzing {model_name}: {e}")
                continue
                
        return complexity_results
        
    def generate_comparison_report(self, metrics_list: List[ModelMetrics],
                                 benchmark_results: Dict = None,
                                 complexity_results: Dict = None) -> str:
        """Generate comprehensive comparison report"""
        
        report = []
        report.append("YOLO MODEL COMPARISON REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance Metrics
        report.append("DETECTION PERFORMANCE")
        report.append("-" * 30)
        
        performance_df = pd.DataFrame([
            {
                'Model': m.model_name,
                'mAP@0.5': f"{m.map50:.4f}",
                'mAP@0.5:0.95': f"{m.map50_95:.4f}",
                'Precision': f"{m.precision:.4f}",
                'Recall': f"{m.recall:.4f}",
                'F1-Score': f"{m.f1_score:.4f}"
            }
            for m in metrics_list
        ])
        
        report.append(performance_df.to_string(index=False))
        report.append("")
        
        # Best performing models
        if len(metrics_list) > 1:
            best_map50 = max(metrics_list, key=lambda x: x.map50)
            best_map95 = max(metrics_list, key=lambda x: x.map50_95)
            
            report.append("BEST PERFORMERS")
            report.append("-" * 20)
            report.append(f"Best mAP@0.5: {best_map50.model_name} ({best_map50.map50:.4f})")
            report.append(f"Best mAP@0.5:0.95: {best_map95.model_name} ({best_map95.map50_95:.4f})")
            report.append("")
        
        # Inference Speed
        if benchmark_results:
            report.append("INFERENCE PERFORMANCE")
            report.append("-" * 30)
            
            speed_df = pd.DataFrame([
                {
                    'Model': model,
                    'Avg Time (ms)': f"{data['mean_inference_time']:.2f}",
                    'Std (ms)': f"{data['std_inference_time']:.2f}",
                    'FPS': f"{data['fps']:.1f}"
                }
                for model, data in benchmark_results.items()
            ])
            
            report.append(speed_df.to_string(index=False))
            report.append("")
            
            # Fastest model
            fastest_model = min(benchmark_results.items(), 
                              key=lambda x: x[1]['mean_inference_time'])
            report.append(f"Fastest Model: {fastest_model[0]} "
                         f"({fastest_model[1]['mean_inference_time']:.2f}ms)")
            report.append("")
        
        # Model Complexity
        if complexity_results:
            report.append("MODEL COMPLEXITY")
            report.append("-" * 25)
            
            complexity_df = pd.DataFrame([
                {
                    'Model': model,
                    'Size (MB)': f"{data['file_size_mb']:.2f}",
                    'Parameters (M)': f"{data['parameters']/1e6:.2f}" if data['parameters'] > 0 else "N/A",
                    'GFLOPs': f"{data['flops']:.2f}" if data['flops'] > 0 else "N/A"
                }
                for model, data in complexity_results.items()
            ])
            
            report.append(complexity_df.to_string(index=False))
            report.append("")
            
            # Smallest model
            smallest_model = min(complexity_results.items(), 
                                key=lambda x: x[1]['file_size_mb'])
            report.append(f"Smallest Model: {smallest_model[0]} "
                         f"({smallest_model[1]['file_size_mb']:.2f}MB)")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 20)
        
        if len(metrics_list) > 1:
            # Find balanced model (good accuracy + speed)
            if benchmark_results:
                balanced_scores = []
                for m in metrics_list:
                    if m.model_name in benchmark_results:
                        # Normalize scores (higher is better for accuracy, lower for inference time)
                        acc_score = m.map50_95
                        speed_score = 1 / benchmark_results[m.model_name]['mean_inference_time']
                        balanced_score = (acc_score + speed_score) / 2
                        balanced_scores.append((m.model_name, balanced_score))
                
                if balanced_scores:
                    best_balanced = max(balanced_scores, key=lambda x: x[1])
                    report.append(f"Best Balanced Model: {best_balanced[0]}")
            
            report.append(f"Highest Accuracy: {best_map95.model_name}")
            if benchmark_results:
                fastest = min(benchmark_results.items(), key=lambda x: x[1]['mean_inference_time'])
                report.append(f"Fastest Inference: {fastest[0]}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
        
    def create_visualizations(self, metrics_list: List[ModelMetrics],
                            benchmark_results: Dict = None,
                            complexity_results: Dict = None) -> None:
        """Create comprehensive visualization suite"""
        
        # Performance comparison
        comparison_df = pd.DataFrame([
            {
                'model_name': m.model_name,
                'map50': m.map50,
                'map50_95': m.map50_95,
                'precision': m.precision,
                'recall': m.recall,
                'f1_score': m.f1_score
            }
            for m in metrics_list
        ])
        
        self.viz_manager.plot_model_comparison(
            comparison_df, 
            save_name="detailed_model_comparison.png"
        )
        
        # Radar chart
        if len(metrics_list) >= 2:
            for metrics in metrics_list:
                self.metrics_tracker.add_metrics(metrics)
            self.metrics_tracker.create_performance_radar(
                save_path=self.results_dir / "performance_radar.png"
            )
        
        # Speed vs Accuracy plot
        if benchmark_results:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for m in metrics_list:
                if m.model_name in benchmark_results:
                    x = benchmark_results[m.model_name]['mean_inference_time']
                    y = m.map50_95
                    ax.scatter(x, y, s=100, label=m.model_name)
                    ax.annotate(m.model_name, (x, y), 
                              xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Inference Time (ms)')
            ax.set_ylabel('mAP@0.5:0.95')
            ax.set_title('Speed vs Accuracy Trade-off')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.results_dir / "speed_vs_accuracy.png", dpi=300, bbox_inches='tight')
            plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Model Comparison Tool")
    parser.add_argument("--experiments", nargs="+", required=True,
                       help="List of experiment directories to compare")
    parser.add_argument("--models", nargs="+", 
                       help="List of model files for benchmarking")
    parser.add_argument("--test-images", nargs="+",
                       help="Test images for speed benchmarking")
    parser.add_argument("--output-dir", default="./results/comparison",
                       help="Output directory for results")
    parser.add_argument("--benchmark-speed", action="store_true",
                       help="Run inference speed benchmarking")
    parser.add_argument("--analyze-complexity", action="store_true",
                       help="Analyze model complexity")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparator = ModelComparator(str(output_dir))
    
    # Load training results
    metrics_list = comparator.load_training_results(args.experiments)
    
    if not metrics_list:
        print("No valid metrics found in experiment directories")
        return 1
    
    print(f"Loaded metrics for {len(metrics_list)} models")
    
    benchmark_results = None
    complexity_results = None
    
    # Run speed benchmarking
    if args.benchmark_speed and args.models and args.test_images:
        print("Running inference speed benchmarking...")
        benchmark_results = comparator.benchmark_inference_speed(
            args.models, args.test_images
        )
    
    # Analyze model complexity
    if args.analyze_complexity and args.models:
        print("Analyzing model complexity...")
        complexity_results = comparator.calculate_model_complexity(args.models)
    
    # Generate report
    report = comparator.generate_comparison_report(
        metrics_list, benchmark_results, complexity_results
    )
    
    # Save report
    report_path = output_dir / "comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Comparison report saved to: {report_path}")
    print("\n" + report)
    
    # Create visualizations
    comparator.create_visualizations(
        metrics_list, benchmark_results, complexity_results
    )
    
    print(f"Visualizations saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
