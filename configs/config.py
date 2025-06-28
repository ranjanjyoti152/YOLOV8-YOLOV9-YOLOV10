"""
Main configuration file for YOLO training project
Supports YOLOv8, YOLOv9, and YOLOv10 training configurations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    """Data configuration class"""
    dataset_path: str
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_classes: int = 80
    class_names: Optional[List[str]] = None
    image_size: Union[int, List[int]] = 640
    augmentation: bool = True
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]


@dataclass
class ModelConfig:
    """Model configuration class"""
    model_name: str  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolov9c, yolov9e, yolov10n, yolov10s, etc.
    pretrained: bool = True
    num_classes: int = 80
    input_size: Union[int, List[int]] = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7
    
    
@dataclass
class TrainingConfig:
    """Training configuration class"""
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    optimizer: str = "SGD"  # SGD, Adam, AdamW
    scheduler: str = "cosine"  # cosine, linear, step
    weight_decay: float = 0.0005
    momentum: float = 0.937
    warmup_epochs: int = 3
    patience: int = 50
    save_period: int = 10
    amp: bool = True  # Automatic Mixed Precision
    
    
@dataclass
class ExperimentConfig:
    """Experiment tracking configuration"""
    project_name: str = "yolo-comparison"
    experiment_name: str = "experiment_1"
    use_wandb: bool = True
    use_tensorboard: bool = True
    save_predictions: bool = True
    save_metrics: bool = True


@dataclass
class YOLOConfig:
    """Main YOLO configuration class"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    experiment: ExperimentConfig
    
    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'experiment': self.experiment.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'YOLOConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            experiment=ExperimentConfig(**config_dict['experiment'])
        )


def get_default_config(model_version: str = "yolov8n") -> YOLOConfig:
    """Get default configuration for specified YOLO version"""
    
    data_config = DataConfig(
        dataset_path="./data/coco128",
        num_classes=80,
        image_size=640
    )
    
    model_config = ModelConfig(
        model_name=model_version,
        num_classes=80,
        input_size=640
    )
    
    training_config = TrainingConfig(
        epochs=100,
        batch_size=16,
        learning_rate=0.01
    )
    
    experiment_config = ExperimentConfig(
        experiment_name=f"{model_version}_experiment"
    )
    
    return YOLOConfig(
        data=data_config,
        model=model_config,
        training=training_config,
        experiment=experiment_config
    )


if __name__ == "__main__":
    # Example usage
    config = get_default_config("yolov8n")
    config.save_yaml("./configs/yolov8n_config.yaml")
    print("Default configuration saved to ./configs/yolov8n_config.yaml")
