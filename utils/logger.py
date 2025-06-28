"""
Logging utilities for YOLO training project
Provides consistent logging across all training scripts
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with consistent formatting
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Default format
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_experiment_logger(experiment_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger for a specific experiment
    
    Args:
        experiment_name: Name of the experiment
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{experiment_name}_{timestamp}.log"
    
    return setup_logger(
        name=experiment_name,
        log_file=str(log_file),
        level=level
    )


class TrainingLogger:
    """Enhanced logger for training with progress tracking"""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = setup_logger(name, log_file)
        self.epoch_start_time = None
        self.training_start_time = None
        
    def start_training(self) -> None:
        """Log training start"""
        self.training_start_time = datetime.now()
        self.logger.info("=" * 50)
        self.logger.info("TRAINING STARTED")
        self.logger.info(f"Start time: {self.training_start_time}")
        self.logger.info("=" * 50)
        
    def end_training(self) -> None:
        """Log training end"""
        if self.training_start_time:
            duration = datetime.now() - self.training_start_time
            self.logger.info("=" * 50)
            self.logger.info("TRAINING COMPLETED")
            self.logger.info(f"Total duration: {duration}")
            self.logger.info("=" * 50)
        
    def start_epoch(self, epoch: int, total_epochs: int) -> None:
        """Log epoch start"""
        self.epoch_start_time = datetime.now()
        self.logger.info(f"Epoch {epoch}/{total_epochs} started")
        
    def end_epoch(self, epoch: int, metrics: dict) -> None:
        """Log epoch end with metrics"""
        if self.epoch_start_time:
            duration = datetime.now() - self.epoch_start_time
            
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Epoch {epoch} completed in {duration} | {metrics_str}")
            
    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        """Log metrics dictionary"""
        for key, value in metrics.items():
            self.logger.info(f"{prefix}{key}: {value}")
            
    def log_model_info(self, model_info: dict) -> None:
        """Log model information"""
        self.logger.info("Model Information:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")
            
    def log_config(self, config: dict) -> None:
        """Log configuration"""
        self.logger.info("Configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")


# Pre-configured loggers for common use cases
def get_data_logger() -> logging.Logger:
    """Get logger for data processing operations"""
    return setup_logger("DataProcessor", level=logging.INFO)


def get_model_logger() -> logging.Logger:
    """Get logger for model operations"""
    return setup_logger("ModelManager", level=logging.INFO)


def get_evaluation_logger() -> logging.Logger:
    """Get logger for evaluation operations"""
    return setup_logger("Evaluator", level=logging.INFO)


# Example usage
if __name__ == "__main__":
    # Test the logging setup
    logger = setup_logger("test_logger", "test.log")
    logger.info("This is a test message")
    
    # Test training logger
    training_logger = TrainingLogger("training_test")
    training_logger.start_training()
    training_logger.start_epoch(1, 10)
    training_logger.end_epoch(1, {"loss": 0.5, "accuracy": 0.85})
    training_logger.end_training()
