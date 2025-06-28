#!/usr/bin/env python3
"""
YOLOv10 Training Script
Training script for YOLOv10 models using the official implementation
"""

import logging
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import os

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


class YOLOv10Trainer:
    """YOLOv10 training class"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        # Check dependencies first
        if torch is None:
            raise ImportError("PyTorch is required. Install with: pip install torch")
            
        self.config = YOLOConfig.from_yaml(config_path)
        self.logger = setup_logger("YOLOv10Trainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # YOLOv10 repository path
        self.yolov10_repo = Path("./models/yolov10")
        
        # Initialize tracking
        self.metrics_tracker = MetricsTracker()
        
        # Setup experiment tracking
        self._setup_experiment_tracking()
        
        self.logger.info(f"Initialized YOLOv10 trainer with device: {self.device}")
        
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
                
    def setup_yolov10_repository(self) -> None:
        """Clone and setup YOLOv10 repository if not exists"""
        if not self.yolov10_repo.exists():
            self.logger.info("Cloning YOLOv10 repository...")
            try:
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/THU-MIG/yolov10.git",
                    str(self.yolov10_repo)
                ], check=True)
                
                self.logger.info("YOLOv10 repository cloned successfully")
                
                # Install YOLOv10 package
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "-e", str(self.yolov10_repo)
                ], check=True)
                
                self.logger.info("YOLOv10 package installed successfully")
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to setup YOLOv10 repository: {e}")
                raise
        else:
            self.logger.info("YOLOv10 repository already exists")
            
    def prepare_data(self) -> str:
        """Prepare dataset configuration for YOLOv10 training"""
        data_yaml_path = Path("./data/yolov10_dataset.yaml")
        
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
        
    def train_with_ultralytics(self) -> None:
        """Train YOLOv10 using ultralytics interface"""
        self.logger.info("Starting YOLOv10 training with ultralytics...")
        
        try:
            # Import YOLOv10 after setup
            try:
                from ultralytics import YOLOv10
            except ImportError:
                raise ImportError("YOLOv10 not available. Please install ultralytics and YOLOv10 repo")
            
            # Prepare data
            data_yaml = self.prepare_data()
            
            # Initialize model
            if self.config.model.pretrained:
                model = YOLOv10(f"{self.config.model.model_name}.pt")
                self.logger.info(f"Loaded pretrained {self.config.model.model_name} model")
            else:
                model = YOLOv10(f"{self.config.model.model_name}.yaml")
                self.logger.info(f"Initialized {self.config.model.model_name} model from scratch")
            
            # Setup results directory
            results_dir = Path("./results") / self.config.experiment.experiment_name
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Train the model
            results = model.train(
                data=data_yaml,
                epochs=self.config.training.epochs,
                imgsz=self.config.data.image_size,
                batch=self.config.training.batch_size,
                lr0=self.config.training.learning_rate,
                optimizer=self.config.training.optimizer,
                weight_decay=self.config.training.weight_decay,
                momentum=self.config.training.momentum,
                warmup_epochs=self.config.training.warmup_epochs,
                patience=self.config.training.patience,
                save_period=self.config.training.save_period,
                amp=self.config.training.amp,
                project=str(results_dir.parent),
                name=results_dir.name,
                exist_ok=True,
                verbose=True
            )
            
            self.logger.info("YOLOv10 training completed successfully!")
            
            # Save final model
            model_save_path = results_dir / "final_model.pt"
            model.save(str(model_save_path))
            self.logger.info(f"Final model saved to: {model_save_path}")
            
        except ImportError:
            self.logger.warning("YOLOv10 ultralytics interface not available, falling back to manual training")
            self.train_manual()
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
            
    def train_manual(self) -> None:
        """Manual training using YOLOv10 repository"""
        self.logger.info("Starting YOLOv10 manual training...")
        
        # Setup YOLOv10 repository
        self.setup_yolov10_repository()
        
        # Prepare data
        data_yaml = self.prepare_data()
        
        # Setup results directory
        results_dir = Path("./results") / self.config.experiment.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for training script
        train_script = self.yolov10_repo / "train.py"
        if not train_script.exists():
            # Try alternative training script locations
            train_script = self.yolov10_repo / "yolov10/train.py"
            if not train_script.exists():
                self.logger.error("Training script not found in YOLOv10 repository")
                raise FileNotFoundError("YOLOv10 training script not found")
        
        # Prepare training command
        cmd = [
            sys.executable, str(train_script),
            "--data", data_yaml,
            "--epochs", str(self.config.training.epochs),
            "--batch-size", str(self.config.training.batch_size),
            "--img", str(self.config.data.image_size),
            "--device", "0" if torch.cuda.is_available() else "cpu",
            "--project", str(results_dir.parent),
            "--name", results_dir.name,
            "--exist-ok"
        ]
        
        # Add model configuration
        if self.config.model.pretrained:
            cmd.extend(["--weights", f"{self.config.model.model_name}.pt"])
        else:
            cmd.extend(["--cfg", f"{self.config.model.model_name}.yaml"])
            
        try:
            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            # Change to YOLOv10 directory for training
            original_cwd = os.getcwd()
            os.chdir(self.yolov10_repo)
            
            # Run training
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            self.logger.info("YOLOv10 training completed successfully!")
            self.logger.info(f"Training output: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            os.chdir(original_cwd)  # Ensure we change back on error
            self.logger.error(f"Training failed: {e}")
            self.logger.error(f"Training stderr: {e.stderr}")
            raise
            
    def train(self) -> None:
        """Main training function - tries ultralytics first, then manual"""
        try:
            self.train_with_ultralytics()
        except Exception as e:
            self.logger.warning(f"Ultralytics training failed: {e}")
            self.logger.info("Attempting manual training...")
            self.train_manual()
            
    def evaluate(self, model_path: str, data_yaml: str) -> None:
        """Evaluate trained YOLOv10 model"""
        self.logger.info("Starting YOLOv10 model evaluation...")
        
        try:
            # Try ultralytics evaluation first
            try:
                from ultralytics import YOLOv10
            except ImportError:
                raise ImportError("YOLOv10 not available. Falling back to manual evaluation.")
            
            model = YOLOv10(model_path)
            
            results = model.val(
                data=data_yaml,
                imgsz=self.config.data.image_size,
                batch=self.config.training.batch_size,
                conf=self.config.model.confidence_threshold,
                iou=self.config.model.iou_threshold,
                verbose=True
            )
            
            self.logger.info("YOLOv10 evaluation completed successfully!")
            
            if self.config.experiment.use_wandb and wandb is not None:
                wandb.log({"final_map50": results.box.map50})
                wandb.log({"final_map95": results.box.map})
                
        except ImportError:
            self.logger.warning("YOLOv10 ultralytics interface not available for evaluation")
            self._evaluate_manual(model_path, data_yaml)
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
            
    def _evaluate_manual(self, model_path: str, data_yaml: str) -> None:
        """Manual evaluation using YOLOv10 repository"""
        # Setup YOLOv10 repository
        self.setup_yolov10_repository()
        
        val_script = self.yolov10_repo / "val.py"
        if not val_script.exists():
            val_script = self.yolov10_repo / "yolov10/val.py"
            
        if val_script.exists():
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
                os.chdir(self.yolov10_repo)
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                os.chdir(original_cwd)
                
                self.logger.info("YOLOv10 evaluation completed successfully!")
                self.logger.info(f"Evaluation output: {result.stdout}")
                
            except subprocess.CalledProcessError as e:
                os.chdir(original_cwd)
                self.logger.error(f"Evaluation failed: {e}")
                raise
        else:
            self.logger.warning("Evaluation script not found in YOLOv10 repository")
            
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
    parser = argparse.ArgumentParser(description="YOLOv10 Training Script")
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
        trainer = YOLOv10Trainer(args.config)
        
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
