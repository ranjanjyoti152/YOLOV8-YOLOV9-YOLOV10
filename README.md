# YOLO Training Project: YOLOv8, YOLOv9 & YOLOv10

![YOLO Banner](https://img.shields.io/badge/YOLO-v8%20%7C%20v9%20%7C%20v10-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A comprehensive training and evaluation framework for YOLOv8, YOLOv9, and YOLOv10 object detection models. This project provides a unified interface for training, comparing, and analyzing different YOLO model variants with extensive visualization and metrics tracking capabilities.

## 🚀 Features

### Core Functionality
- **Multi-YOLO Support**: Train and compare YOLOv8, YOLOv9, and YOLOv10 models
- **Comprehensive Metrics**: Track mAP, precision, recall, F1-score, and inference times
- **Advanced Visualization**: Training curves, comparison charts, confusion matrices
- **Experiment Tracking**: Integration with TensorBoard and Weights & Biases
- **Configuration Management**: YAML-based configuration system
- **Automated Data Preparation**: Download and validate datasets automatically

### Advanced Features
- **Model Comparison Dashboard**: Side-by-side performance analysis
- **Hyperparameter Optimization**: Grid search and Bayesian optimization
- **Export Capabilities**: ONNX, TensorRT, CoreML export support
- **Graceful Dependency Handling**: Works even with missing optional dependencies
- **VS Code Integration**: Pre-configured tasks and debugging setup

## 📦 Installation

### Quick Setup
```bash
git clone https://github.com/ranjanjyoti152/YOLOV8-YOLOV9-YOLOV10.git
cd YOLOV8-YOLOV9-YOLOV10
chmod +x setup.py
./setup.py
```

### Manual Installation
```bash
# Clone the repository
git clone https://github.com/ranjanjyoti152/YOLOV8-YOLOV9-YOLOV10.git
cd YOLOV8-YOLOV9-YOLOV10

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

### 1. Download Dataset
```bash
python scripts/prepare_data.py --action download
```

### 2. Train Models
```bash
# Train YOLOv8n
python scripts/train_yolov8.py --config configs/yolov8n_config.yaml

# Train YOLOv9c
python scripts/train_yolov9.py --config configs/yolov9c_config.yaml

# Train YOLOv10n
python scripts/train_yolov10.py --config configs/yolov10n_config.yaml
```

### 3. Compare Results
```bash
python scripts/compare_models.py --experiments \
    ./results/yolov8n_experiment \
    ./results/yolov9c_experiment \
    ./results/yolov10n_experiment
```

## 📊 Project Structure

```
├── 📁 configs/              # Configuration files
│   ├── config.py           # Configuration management
│   ├── yolov8n_config.yaml
│   ├── yolov9c_config.yaml
│   └── yolov10n_config.yaml
├── 📁 data/                # Dataset storage
├── 📁 models/              # Trained models
├── 📁 results/             # Training results and logs
├── 📁 scripts/             # Training and utility scripts
│   ├── train_yolov8.py     # YOLOv8 training script
│   ├── train_yolov9.py     # YOLOv9 training script
│   ├── train_yolov10.py    # YOLOv10 training script
│   ├── prepare_data.py     # Data preparation utilities
│   └── compare_models.py   # Model comparison tools
├── 📁 utils/               # Utility modules
│   ├── logger.py           # Logging utilities
│   ├── metrics.py          # Metrics calculation
│   └── visualization.py    # Visualization tools
├── 📁 .github/             # GitHub workflows and docs
├── 📁 .vscode/             # VS Code configuration
├── requirements.txt        # Python dependencies
├── setup.py               # Setup script
└── README.md              # This file
```

## ⚙️ Configuration

### Model Configuration
Each YOLO version has its own configuration file with optimized hyperparameters:

```yaml
# Example: yolov8n_config.yaml
model:
  name: "yolov8n"
  input_size: 640
  
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.01
  
data:
  dataset_path: "./data/coco128"
  train_split: 0.8
  val_split: 0.2
```

### Custom Configuration
```python
from configs.config import get_default_config

# Get base config
config = get_default_config('yolov8n')

# Modify parameters
config.training.epochs = 200
config.training.batch_size = 32

# Save custom config
config.save_yaml('./configs/my_custom_config.yaml')
```

## 📈 Training Examples

### Basic Training
```python
from scripts.train_yolov8 import YOLOv8Trainer

trainer = YOLOv8Trainer(config_path="configs/yolov8n_config.yaml")
trainer.train()
```

### Advanced Training with Callbacks
```python
trainer = YOLOv8Trainer(
    config_path="configs/yolov8n_config.yaml",
    use_wandb=True,
    use_tensorboard=True
)

# Add custom callbacks
trainer.add_callback('on_epoch_end', custom_callback)
trainer.train()
```

## 🔍 Model Comparison

### Performance Metrics
```python
from utils.metrics import MetricsTracker

tracker = MetricsTracker()
comparison_df = tracker.compare_models()
print(comparison_df)
```

### Visualization
```python
from utils.visualization import VisualizationManager

viz = VisualizationManager()
viz.plot_model_comparison(comparison_data)
viz.plot_training_curves(model_name, train_losses, val_losses)
```

## 🛠️ VS Code Integration

This project comes with pre-configured VS Code tasks:

- **Install Dependencies**: `Ctrl+Shift+P` → `Tasks: Run Task` → `Install Dependencies`
- **Download Dataset**: `Tasks: Run Task` → `Download COCO128 Dataset`
- **Train Models**: `Tasks: Run Task` → `Train YOLOv8n/v9c/v10n`
- **Start TensorBoard**: `Tasks: Run Task` → `Start TensorBoard`

## 🧪 Testing

### Run Tests
```bash
# Basic import tests
python -c "from scripts.train_yolov8 import YOLOv8Trainer; print('Success')"

# Validate dataset
python scripts/prepare_data.py --action validate --dataset-path ./data/coco128

# Quick model test
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLO test passed')"
```

## 📊 Metrics and Evaluation

### Supported Metrics
- **Detection Metrics**: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1-Score
- **Performance Metrics**: Inference Time, Model Size, FLOPs
- **Per-class Metrics**: Class-specific AP, Precision, Recall

### Visualization Options
- Training curves (loss, metrics over epochs)
- Model comparison charts
- Confusion matrices
- PR curves
- Performance radar charts

## 🔧 Advanced Usage

### Custom Dataset Training
```python
# Prepare custom dataset
python scripts/prepare_data.py --action create_custom \
    --images-path /path/to/images \
    --labels-path /path/to/labels

# Update config
config = get_default_config('yolov8n')
config.data.dataset_path = "./data/custom_dataset"
config.save_yaml('./configs/custom_config.yaml')

# Train
python scripts/train_yolov8.py --config configs/custom_config.yaml
```

### Hyperparameter Optimization
```python
from scripts.train_yolov8 import YOLOv8Trainer
from utils.optimization import GridSearchCV

# Define parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 200]
}

# Run grid search
optimizer = GridSearchCV(YOLOv8Trainer, param_grid)
best_params = optimizer.search()
```

## 🐳 Docker Support

### Build Docker Image
```bash
docker build -t yolo-training .
```

### Run Training in Docker
```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
    yolo-training python scripts/train_yolov8.py --config configs/yolov8n_config.yaml
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [WongKinYiu](https://github.com/WongKinYiu/yolov9) for YOLOv9
- [THU-MIG](https://github.com/THU-MIG/yolov10) for YOLOv10
- COCO dataset for evaluation benchmarks

## 📚 References

- [YOLOv8 Paper](https://arxiv.org/abs/2305.09972)
- [YOLOv9 Paper](https://arxiv.org/abs/2402.13616)
- [YOLOv10 Paper](https://arxiv.org/abs/2405.14458)

## 📞 Support

- 📧 Email: ranjanjyoti152@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/ranjanjyoti152/YOLOV8-YOLOV9-YOLOV10/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/ranjanjyoti152/YOLOV8-YOLOV9-YOLOV10/discussions)

---

<div align="center">
  <p>⭐ Star this repository if you find it helpful!</p>
  <p>Made with ❤️ for the Computer Vision community</p>
</div>
