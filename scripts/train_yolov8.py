#!/usr/bin/env python3
"""
YOLOv8 Training Script
Comprehensive training script for YOLOv8 models using Ultralytics
"""

import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import sys
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

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import YOLOConfig
from utils.logger import setup_logger
from utils.metrics import MetricsTracker
from utils.visualization import VisualizationManager


class YOLOv8Trainer:
    """YOLOv8 training class with comprehensive logging and monitoring"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        # Check dependencies first
        if torch is None:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        if YOLO is None:
            raise ImportError("Ultralytics is required. Install with: pip install ultralytics")
            
        self.config = YOLOConfig.from_yaml(config_path)
        self.logger = setup_logger("YOLOv8Trainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tracking
        self.metrics_tracker = MetricsTracker()
        self.viz_manager = VisualizationManager()
        
        # Setup experiment tracking
        self._setup_experiment_tracking()
        
        self.logger.info(f"Initialized YOLOv8 trainer with device: {self.device}")
        self.logger.info(f"Configuration loaded from: {config_path}")
        
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
                
    def prepare_data(self) -> str:
        """Prepare dataset configuration for training"""
        data_yaml_path = Path("./data/dataset.yaml")
        
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
        
    def load_model(self):
        """Load and configure YOLOv8 model"""
        try:
            if self.config.model.pretrained:
                model = YOLO(f"{self.config.model.model_name}.pt")
                self.logger.info(f"Loaded pretrained {self.config.model.model_name} model")
            else:
                model = YOLO(f"{self.config.model.model_name}.yaml")
                self.logger.info(f"Initialized {self.config.model.model_name} model from scratch")
                
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
    def train(self) -> Dict[str, Any]:
        """Main training function"""
        self.logger.info("Starting YOLOv8 training...")
        
        # Prepare data
        data_yaml = self.prepare_data()
        
        # Load model
        model = self.load_model()
        
        # Setup results directory
        results_dir = Path("./results") / self.config.experiment.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        try:
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
            
            self.logger.info("Training completed successfully!")
            
            # Save final model
            model_save_path = results_dir / "final_model.pt"
            model.save(str(model_save_path))
            self.logger.info(f"Final model saved to: {model_save_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
            
    def evaluate(self, model_path: str, data_yaml: str) -> Dict[str, Any]:
        """Evaluate trained model"""
        self.logger.info("Starting model evaluation...")
        
        try:
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(
                data=data_yaml,
                imgsz=self.config.data.image_size,
                batch=self.config.training.batch_size,
                conf=self.config.model.confidence_threshold,
                iou=self.config.model.iou_threshold,
                verbose=True
            )
            
            self.logger.info("Evaluation completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
            
    def run_full_pipeline(self) -> None:
        """Run complete training and evaluation pipeline"""
        try:
            # Train model
            training_results = self.train()
            
            # Evaluate model
            best_model_path = Path("./results") / self.config.experiment.experiment_name / "weights/best.pt"
            data_yaml = self.prepare_data()
            
            if best_model_path.exists():
                eval_results = self.evaluate(str(best_model_path), data_yaml)
                
                if self.config.experiment.use_wandb and wandb is not None:
                    wandb.log({"final_map50": eval_results.box.map50})
                    wandb.log({"final_map95": eval_results.box.map})
                    
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
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
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
        trainer = YOLOv8Trainer(args.config)
        
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
