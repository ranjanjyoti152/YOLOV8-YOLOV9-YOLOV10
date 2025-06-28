#!/usr/bin/env python3
"""
YOLOv9 Training Script
Training script for YOLOv9 models using the official implementation
"""

import logging
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import os
import shutil

# Optional imports with error handling
try:
    import torch
except ImportError:
    print("Warning: PyTorch not installed. Please install with: pip install torch")
    torch = None

try:
    import wandb
except ImportError:
    print("Warning: wandb not installed. Install with: pip install wandb")
    wandb = None

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import YOLOConfig
from utils.logger import setup_logger
from utils.metrics import MetricsTracker


class YOLOv9Trainer:
    """YOLOv9 training class"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        # Check dependencies first
        if torch is None:
            raise ImportError("PyTorch is required. Install with: pip install torch")
            
        self.config = YOLOConfig.from_yaml(config_path)
        self.logger = setup_logger("YOLOv9Trainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # YOLOv9 repository path
        self.yolov9_repo = Path("./models/yolov9")
        
        # Initialize tracking
        self.metrics_tracker = MetricsTracker()
        
        # Setup experiment tracking
        self._setup_experiment_tracking()
        
        self.logger.info(f"Initialized YOLOv9 trainer with device: {self.device}")
        
    def _setup_experiment_tracking(self) -> None:
        """Setup experiment tracking with wandb"""
        if self.config.experiment.use_wandb and wandb is not None:
            try:
                wandb.init(
                    project=self.config.experiment.project_name,
                    name=self.config.experiment.experiment_name,
                    config={
                        "model": self.config.model.__dict__,
                        "training": self.config.training.__dict__,
                        "data": self.config.data.__dict__
                    }
                )
                self.logger.info("Weights & Biases tracking initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize wandb: {e}")
        elif self.config.experiment.use_wandb and wandb is None:
            self.logger.warning("wandb not available. Install with: pip install wandb")
                
    def setup_yolov9_repository(self) -> None:
        """Clone and setup YOLOv9 repository if not exists"""
        if not self.yolov9_repo.exists():
            self.logger.info("Cloning YOLOv9 repository...")
            try:
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/WongKinYiu/yolov9.git",
                    str(self.yolov9_repo)
                ], check=True)
                
                self.logger.info("YOLOv9 repository cloned successfully")
                
                # Install additional requirements
                requirements_path = self.yolov9_repo / "requirements.txt"
                if requirements_path.exists():
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", 
                        "-r", str(requirements_path)
                    ], check=True)
                    
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to setup YOLOv9 repository: {e}")
                raise
        else:
            self.logger.info("YOLOv9 repository already exists")
            
    def prepare_data(self) -> str:
        """Prepare dataset configuration for YOLOv9 training"""
        data_yaml_path = Path("./data/yolov9_dataset.yaml")
        
        # Create dataset YAML configuration
        dataset_config = {
            'path': str(Path(self.config.data.dataset_path).absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': self.config.data.num_classes,
            'names': self.config.data.class_names or [f'class{i}' for i in range(self.config.data.num_classes)]
        }
        
        # Save dataset configuration
        data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
            
        self.logger.info(f"Dataset configuration saved to: {data_yaml_path}")
        return str(data_yaml_path)
        
    def download_pretrained_weights(self) -> str:
        """Download pretrained weights for YOLOv9"""
        weights_dir = Path("./models/weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Map model names to weight URLs
        weight_urls = {
            "yolov9c": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9c.pt",
            "yolov9e": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9e.pt",
            "yolov9s": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9s.pt",
            "yolov9m": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9m.pt",
        }
        
        model_name = self.config.model.model_name
        weight_path = weights_dir / f"{model_name}.pt"
        
        if not weight_path.exists() and self.config.model.pretrained:
            if model_name in weight_urls:
                self.logger.info(f"Downloading {model_name} pretrained weights...")
                try:
                    subprocess.run([
                        "wget", "-O", str(weight_path), weight_urls[model_name]
                    ], check=True)
                    self.logger.info("Pretrained weights downloaded successfully")
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Failed to download weights: {e}")
                    return ""
            else:
                self.logger.warning(f"No pretrained weights available for {model_name}")
                return ""
                
        return str(weight_path) if weight_path.exists() else ""
        
    def train(self) -> None:
        """Main training function for YOLOv9"""
        self.logger.info("Starting YOLOv9 training...")
        
        # Setup YOLOv9 repository
        self.setup_yolov9_repository()
        
        # Prepare data
        data_yaml = self.prepare_data()
        
        # Download pretrained weights
        weights_path = self.download_pretrained_weights()
        
        # Setup results directory
        results_dir = Path("./results") / self.config.experiment.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare training command
        train_script = self.yolov9_repo / "train.py"
        
        cmd = [
            sys.executable, str(train_script),
            "--data", data_yaml,
            "--cfg", f"models/{self.config.model.model_name}.yaml",
            "--epochs", str(self.config.training.epochs),
            "--batch-size", str(self.config.training.batch_size),
            "--img", str(self.config.data.image_size),
            "--device", "0" if torch.cuda.is_available() else "cpu",
            "--project", str(results_dir.parent),
            "--name", results_dir.name,
            "--exist-ok"
        ]
        
        # Add pretrained weights if available
        if weights_path and self.config.model.pretrained:
            cmd.extend(["--weights", weights_path])
            
        # Add optimizer and other training parameters
        if self.config.training.optimizer.lower() == "adam":
            cmd.append("--adam")
            
        try:
            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            # Change to YOLOv9 directory for training
            original_cwd = os.getcwd()
            os.chdir(self.yolov9_repo)
            
            # Run training
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            self.logger.info("YOLOv9 training completed successfully!")
            self.logger.info(f"Training output: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            os.chdir(original_cwd)  # Ensure we change back on error
            self.logger.error(f"Training failed: {e}")
            self.logger.error(f"Training stderr: {e.stderr}")
            raise
            
    def evaluate(self, model_path: str, data_yaml: str) -> None:
        """Evaluate trained YOLOv9 model"""
        self.logger.info("Starting YOLOv9 model evaluation...")
        
        # Setup YOLOv9 repository
        self.setup_yolov9_repository()
        
        val_script = self.yolov9_repo / "val.py"
        
        cmd = [
            sys.executable, str(val_script),
            "--data", data_yaml,
            "--weights", model_path,
            "--img", str(self.config.data.image_size),
            "--batch-size", str(self.config.training.batch_size),
            "--conf-thres", str(self.config.model.confidence_threshold),
            "--iou-thres", str(self.config.model.iou_threshold),
            "--device", "0" if torch.cuda.is_available() else "cpu",
            "--verbose"
        ]
        
        try:
            original_cwd = os.getcwd()
            os.chdir(self.yolov9_repo)
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            os.chdir(original_cwd)
            
            self.logger.info("YOLOv9 evaluation completed successfully!")
            self.logger.info(f"Evaluation output: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            os.chdir(original_cwd)
            self.logger.error(f"Evaluation failed: {e}")
            self.logger.error(f"Evaluation stderr: {e.stderr}")
            raise
            
    def run_full_pipeline(self) -> None:
        """Run complete training and evaluation pipeline"""
        try:
            # Train model
            self.train()
            
            # Evaluate model
            best_model_path = Path("./results") / self.config.experiment.experiment_name / "weights/best.pt"
            data_yaml = self.prepare_data()
            
            if best_model_path.exists():
                self.evaluate(str(best_model_path), data_yaml)
                
            # Finish wandb run
            if self.config.experiment.use_wandb and wandb is not None:
                wandb.finish()
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            if self.config.experiment.use_wandb and wandb is not None:
                wandb.finish()
            raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLOv9 Training Script")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "full"],
        default="full",
        help="Training mode: train, eval, or full pipeline"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file {args.config} not found!")
        return 1
        
    try:
        trainer = YOLOv9Trainer(args.config)
        
        if args.mode == "train":
            trainer.train()
        elif args.mode == "eval":
            # For evaluation mode, assume model exists
            model_path = Path("./results") / trainer.config.experiment.experiment_name / "weights/best.pt"
            data_yaml = trainer.prepare_data()
            trainer.evaluate(str(model_path), data_yaml)
        else:
            trainer.run_full_pipeline()
            
        return 0
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
