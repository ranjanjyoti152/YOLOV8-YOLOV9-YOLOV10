# Copilot Instructions for YOLO Training Project

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview
This is a comprehensive YOLO (You Only Look Once) object detection training project that supports YOLOv8, YOLOv9, and YOLOv10 models. The project is designed for training, evaluation, and comparison of these different YOLO versions.

## Code Style and Conventions
- Use Python 3.8+ features and type hints
- Follow PEP 8 style guidelines
- Use descriptive variable and function names
- Include comprehensive docstrings for all functions and classes
- Use logging instead of print statements for debugging
- Handle exceptions gracefully with proper error messages

## Project Structure Guidelines
- `scripts/` - Training and evaluation scripts for each YOLO version
- `utils/` - Utility functions for data processing, visualization, and metrics
- `configs/` - Configuration files for different training setups
- `data/` - Dataset storage and preparation scripts
- `models/` - Trained model storage and model definition files
- `results/` - Training logs, metrics, and output visualizations

## Framework Specific Instructions
- Use ultralytics for YOLOv8 implementation
- Use official repositories for YOLOv9 and YOLOv10
- Prefer PyTorch over other deep learning frameworks
- Use wandb or tensorboard for experiment tracking
- Implement proper data augmentation techniques
- Include model comparison and benchmarking utilities

## Best Practices
- Always validate data before training
- Implement proper train/validation/test splits
- Use configuration files for hyperparameters
- Save model checkpoints regularly during training
- Include visualization tools for predictions and metrics
- Implement proper GPU memory management
- Add progress bars and logging for long-running processes
