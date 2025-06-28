# Getting Started with YOLO Training Project

Welcome to the comprehensive YOLO training project! This guide will help you get started quickly.

## 🚀 Quick Setup

### 1. Run the Setup Script

```bash
python setup.py
```

This will:
- Check your Python version (3.8+ required)
- Install all dependencies
- Create necessary directories
- Download sample dataset (COCO128)
- Test the installation

### 2. Alternative: Manual Setup

If you prefer manual setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Download sample dataset
python scripts/prepare_data.py --action download

# Test installation
python -c "from ultralytics import YOLO; print('Setup successful!')"
```

## 🎯 First Training Run

### Option 1: Use VS Code Tasks

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Tasks: Run Task"
3. Select "Train YOLOv8n"

### Option 2: Command Line

```bash
python scripts/train_yolov8.py --config configs/yolov8n_config.yaml --mode full
```

### Option 3: Quick Start Script

```bash
# Linux/Mac
./quick_start.sh

# Windows
quick_start.bat
```

## 📊 Monitor Training

### Weights & Biases (Recommended)

1. Sign up at [wandb.ai](https://wandb.ai)
2. Run: `wandb login`
3. Training metrics will be automatically logged

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir results/ --port 6006

# Open browser to http://localhost:6006
```

### Local Logs

- Console output shows real-time progress
- Detailed logs saved in `logs/` directory
- Results and visualizations in `results/` directory

## 🔧 Available Tasks (VS Code)

Access via `Ctrl+Shift+P` → "Tasks: Run Task":

- **Install Dependencies** - Install all required packages
- **Download COCO128 Dataset** - Get sample dataset
- **Train YOLOv8n** - Train YOLOv8 nano model
- **Train YOLOv9c** - Train YOLOv9 compact model
- **Train YOLOv10n** - Train YOLOv10 nano model
- **Train All Models** - Sequential training of all models
- **Validate Dataset** - Check dataset integrity
- **Compare Models** - Generate comparison report
- **Start TensorBoard** - Launch TensorBoard server

## 🎮 Debug Configurations

Launch configurations available in VS Code (`F5` or Debug panel):

- **Train YOLOv8** - Debug YOLOv8 training
- **Train YOLOv9** - Debug YOLOv9 training
- **Train YOLOv10** - Debug YOLOv10 training
- **Prepare Data** - Debug data preparation
- **Compare Models** - Debug model comparison

## 📁 Project Structure Overview

```
yolov8/
├── configs/           # YAML configuration files
├── scripts/          # Training and utility scripts
├── utils/           # Helper modules (logging, metrics, viz)
├── data/            # Your datasets go here
├── models/          # Trained models and repos
├── results/         # Training outputs and metrics
├── logs/           # Log files
└── .vscode/        # VS Code settings and tasks
```

## 🛠️ Custom Dataset

### 1. Prepare Your Data

Organize your dataset like this:

```
data/your_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 2. Create Dataset Configuration

```bash
python scripts/prepare_data.py --action create-yaml \
    --dataset-path ./data/your_dataset \
    --class-names class1 class2 class3 \
    --output ./data/your_dataset.yaml
```

### 3. Update Training Config

Edit `configs/yolov8n_config.yaml`:

```yaml
data:
  dataset_path: ./data/your_dataset
  num_classes: 3  # Number of your classes
  class_names: ['class1', 'class2', 'class3']
```

### 4. Train on Your Data

```bash
python scripts/train_yolov8.py --config configs/yolov8n_config.yaml
```

## 🏆 Model Comparison

After training multiple models:

```bash
python scripts/compare_models.py \
    --experiments ./results/yolov8n_experiment ./results/yolov9c_experiment \
    --models ./results/yolov8n_experiment/weights/best.pt ./results/yolov9c_experiment/weights/best.pt \
    --benchmark-speed --analyze-complexity
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Use smaller image size
   - Enable `amp: true` for mixed precision

2. **Slow Training**
   - Check GPU utilization: `nvidia-smi`
   - Increase `batch_size` if memory allows
   - Use SSD for dataset storage

3. **Poor Performance**
   - Validate dataset: `python scripts/prepare_data.py --action validate --dataset-path ./data/your_dataset`
   - Check class balance
   - Increase training epochs
   - Try different learning rates

### Getting Help

1. **Check logs**: `tail -f logs/latest.log`
2. **Validate setup**: `python setup.py --skip-data --skip-deps`
3. **Test installation**: VS Code Task "Quick Test YOLOv8"

## 📚 Next Steps

1. **Read the full README.md** for detailed documentation
2. **Experiment with hyperparameters** in config files
3. **Try different model sizes** (n, s, m, l, x variants)
4. **Set up experiment tracking** with wandb
5. **Deploy your trained model** for inference

## 🤝 Need Help?

- 📖 Check `README.md` for comprehensive documentation
- 🐛 Create an issue for bugs or questions
- 💬 Join the discussion for community support

Happy training! 🚀
