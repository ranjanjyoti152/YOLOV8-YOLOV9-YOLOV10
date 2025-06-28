#!/usr/bin/env python3
"""
Setup script for YOLO training project
Helps users get started quickly with the project
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import platform


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_gpu():
    """Check for GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠ No GPU detected, training will use CPU (slower)")
            return False
    except ImportError:
        print("⚠ PyTorch not installed, GPU check skipped")
        return False


def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def download_sample_dataset():
    """Download COCO128 sample dataset"""
    print("Downloading sample dataset (COCO128)...")
    
    try:
        subprocess.run([
            sys.executable, "scripts/prepare_data.py", "--action", "download"
        ], check=True)
        print("✓ Sample dataset downloaded")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download dataset: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "data", "models", "results", "logs", 
        "results/metrics", "results/visualizations"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")


def test_installation():
    """Test the installation by running a quick inference"""
    print("Testing installation...")
    
    try:
        # Test YOLOv8 installation
        subprocess.run([
            sys.executable, "-c",
            "from ultralytics import YOLO; "
            "model = YOLO('yolov8n.pt'); "
            "print('✓ YOLOv8 installation test passed')"
        ], check=True, capture_output=True)
        
        print("✓ Installation test passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation test failed: {e}")
        return False


def setup_git_hooks():
    """Setup git hooks for development"""
    if Path(".git").exists():
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "pre-commit"
            ], check=True, capture_output=True)
            
            # Create .pre-commit-config.yaml if it doesn't exist
            pre_commit_config = """repos:
-   repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
    -   id: black
        language_version: python3
-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
"""
            
            with open(".pre-commit-config.yaml", "w") as f:
                f.write(pre_commit_config)
            
            subprocess.run(["pre-commit", "install"], check=True, capture_output=True)
            print("✓ Git hooks configured")
        except:
            print("⚠ Git hooks setup skipped")


def create_example_scripts():
    """Create example scripts for quick start"""
    
    # Quick start script
    quick_start = """#!/bin/bash
# Quick start script for YOLO training

echo "Starting YOLO training pipeline..."

# Download sample data if not exists
if [ ! -d "data/coco128" ]; then
    echo "Downloading sample dataset..."
    python scripts/prepare_data.py --action download
fi

# Train YOLOv8 (fastest to start with)
echo "Training YOLOv8n model..."
python scripts/train_yolov8.py --config configs/yolov8n_config.yaml --mode full

echo "Training completed! Check results/ directory for outputs."
"""
    
    with open("quick_start.sh", "w") as f:
        f.write(quick_start)
    
    os.chmod("quick_start.sh", 0o755)
    
    # Windows batch file
    quick_start_bat = """@echo off
echo Starting YOLO training pipeline...

if not exist "data\\coco128" (
    echo Downloading sample dataset...
    python scripts\\prepare_data.py --action download
)

echo Training YOLOv8n model...
python scripts\\train_yolov8.py --config configs\\yolov8n_config.yaml --mode full

echo Training completed! Check results\\ directory for outputs.
pause
"""
    
    with open("quick_start.bat", "w") as f:
        f.write(quick_start_bat)
    
    print("✓ Example scripts created")


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Download a dataset or use the sample COCO128:")
    print("   python scripts/prepare_data.py --action download")
    print()
    print("2. Train your first model:")
    print("   python scripts/train_yolov8.py --config configs/yolov8n_config.yaml")
    print()
    print("3. Or use the quick start script:")
    if platform.system() == "Windows":
        print("   quick_start.bat")
    else:
        print("   ./quick_start.sh")
    print()
    print("4. Compare models after training:")
    print("   python scripts/compare_models.py --experiments results/*")
    print()
    print("5. Use VS Code tasks (Ctrl+Shift+P -> 'Tasks: Run Task'):")
    print("   - Install Dependencies")
    print("   - Train YOLOv8n")
    print("   - Download COCO128 Dataset")
    print("   - Compare Models")
    print()
    print("📚 Documentation: README.md")
    print("🐛 Issues: Check logs/ directory for debugging")
    print("📊 Monitor training: Use wandb or tensorboard")
    print()
    print("Happy training! 🚀")


def main():
    parser = argparse.ArgumentParser(description="Setup YOLO Training Project")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="Skip dependency installation")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip sample dataset download")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip installation test")
    parser.add_argument("--dev", action="store_true",
                       help="Setup for development (includes git hooks)")
    
    args = parser.parse_args()
    
    print("YOLO Training Project Setup")
    print("="*40)
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("Failed to install dependencies. Please run manually:")
            print("pip install -r requirements.txt")
            sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    # Download sample dataset
    if not args.skip_data:
        download_sample_dataset()
    
    # Test installation
    if not args.skip_test:
        test_installation()
    
    # Development setup
    if args.dev:
        setup_git_hooks()
    
    # Create example scripts
    create_example_scripts()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
