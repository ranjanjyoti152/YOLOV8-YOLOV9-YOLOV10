#!/usr/bin/env python3
"""
Data preparation utilities for YOLO training
Handles dataset downloading, conversion, and validation
"""

import os
import sys
import shutil
import requests
import zipfile
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
import argparse

# Optional imports with error handling
try:
    import cv2
except ImportError:
    print("Warning: OpenCV not installed. Install with: pip install opencv-python")
    cv2 = None

try:
    import numpy as np
except ImportError:
    print("Warning: NumPy not installed. Install with: pip install numpy")
    np = None

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    # Fallback progress indicator
    def tqdm(iterable, **kwargs):
        return iterable

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger


class DatasetPreparator:
    """Comprehensive dataset preparation utility"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.logger = setup_logger("DatasetPreparator")
        
    def download_coco128(self) -> str:
        """Download COCO128 dataset for testing"""
        dataset_path = self.data_dir / "coco128"
        
        if dataset_path.exists():
            self.logger.info("COCO128 dataset already exists")
            return str(dataset_path)
            
        self.logger.info("Downloading COCO128 dataset...")
        
        # Download URL
        url = "https://ultralytics.com/assets/coco128.zip"
        zip_path = self.data_dir / "coco128.zip"
        
        try:
            # Download file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading COCO128",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Extract zip file
            self.logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Clean up zip file
            zip_path.unlink()
            
            self.logger.info(f"COCO128 dataset downloaded to {dataset_path}")
            return str(dataset_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download COCO128: {e}")
            if zip_path.exists():
                zip_path.unlink()
            raise
            
    def validate_dataset(self, dataset_path: str) -> Dict[str, any]:
        """Validate dataset structure and annotations"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check directory structure
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        optional_dirs = ['images/test', 'labels/test']
        
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                validation_results['errors'].append(f"Missing required directory: {dir_name}")
                validation_results['valid'] = False
                
        for dir_name in optional_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                validation_results['warnings'].append(f"Missing optional directory: {dir_name}")
        
        if not validation_results['valid']:
            return validation_results
            
        # Validate images and labels
        for split in ['train', 'val', 'test']:
            images_dir = dataset_path / 'images' / split
            labels_dir = dataset_path / 'labels' / split
            
            if not images_dir.exists():
                continue
                
            image_files = list(images_dir.glob('*'))
            image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            label_files = list(labels_dir.glob('*.txt')) if labels_dir.exists() else []
            
            # Statistics
            validation_results['statistics'][split] = {
                'num_images': len(image_files),
                'num_labels': len(label_files),
                'missing_labels': [],
                'invalid_labels': []
            }
            
            # Check for missing labels
            for image_file in image_files:
                label_file = labels_dir / (image_file.stem + '.txt')
                if not label_file.exists():
                    validation_results['statistics'][split]['missing_labels'].append(str(image_file.name))
            
            # Validate label format
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split()
                        if len(parts) != 5:
                            validation_results['statistics'][split]['invalid_labels'].append(
                                f"{label_file.name}:{line_num} - Invalid format"
                            )
                            continue
                            
                        try:
                            class_id = int(parts[0])
                            x, y, w, h = map(float, parts[1:])
                            
                            # Check if coordinates are normalized (0-1)
                            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                validation_results['statistics'][split]['invalid_labels'].append(
                                    f"{label_file.name}:{line_num} - Coordinates not normalized"
                                )
                                
                        except ValueError:
                            validation_results['statistics'][split]['invalid_labels'].append(
                                f"{label_file.name}:{line_num} - Invalid values"
                            )
                            
                except Exception as e:
                    validation_results['statistics'][split]['invalid_labels'].append(
                        f"{label_file.name} - Error reading file: {e}"
                    )
        
        # Summary
        total_images = sum(stats['num_images'] for stats in validation_results['statistics'].values())
        total_labels = sum(stats['num_labels'] for stats in validation_results['statistics'].values())
        total_missing = sum(len(stats['missing_labels']) for stats in validation_results['statistics'].values())
        total_invalid = sum(len(stats['invalid_labels']) for stats in validation_results['statistics'].values())
        
        validation_results['summary'] = {
            'total_images': total_images,
            'total_labels': total_labels,
            'missing_labels': total_missing,
            'invalid_labels': total_invalid
        }
        
        if total_missing > 0:
            validation_results['warnings'].append(f"{total_missing} images have missing labels")
            
        if total_invalid > 0:
            validation_results['warnings'].append(f"{total_invalid} invalid label entries found")
        
        self.logger.info(f"Dataset validation completed: {validation_results['summary']}")
        
        return validation_results
        
    def create_dataset_yaml(self, dataset_path: str, class_names: List[str], 
                          output_path: str = None) -> str:
        """Create dataset YAML configuration file"""
        dataset_path = Path(dataset_path)
        
        if output_path is None:
            output_path = dataset_path / "dataset.yaml"
        else:
            output_path = Path(output_path)
            
        dataset_config = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, indent=2)
            
        self.logger.info(f"Dataset YAML created: {output_path}")
        return str(output_path)
        
    def analyze_dataset(self, dataset_path: str) -> Dict[str, any]:
        """Analyze dataset and provide statistics"""
        dataset_path = Path(dataset_path)
        
        analysis = {
            'class_distribution': {},
            'image_statistics': {},
            'annotation_statistics': {}
        }
        
        # Analyze each split
        for split in ['train', 'val', 'test']:
            images_dir = dataset_path / 'images' / split
            labels_dir = dataset_path / 'labels' / split
            
            if not images_dir.exists():
                continue
                
            split_analysis = {
                'num_images': 0,
                'num_annotations': 0,
                'class_counts': {},
                'image_sizes': [],
                'bbox_areas': []
            }
            
            # Get image files
            image_files = list(images_dir.glob('*'))
            image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            split_analysis['num_images'] = len(image_files)
            
            # Analyze images and labels
            for image_file in tqdm(image_files, desc=f"Analyzing {split}"):
                # Read image
                try:
                    if cv2 is not None:
                        img = cv2.imread(str(image_file))
                        if img is not None:
                            h, w = img.shape[:2]
                            split_analysis['image_sizes'].append((w, h))
                    else:
                        # Fallback without opencv
                        split_analysis['image_sizes'].append((640, 640))  # Default size
                except:
                    continue
                
                # Read corresponding label
                label_file = labels_dir / (image_file.stem + '.txt')
                if label_file.exists():
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                                
                            parts = line.split()
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                x, y, w_norm, h_norm = map(float, parts[1:])
                                
                                # Count class occurrences
                                split_analysis['class_counts'][class_id] = \
                                    split_analysis['class_counts'].get(class_id, 0) + 1
                                
                                # Calculate bbox area (in pixels)
                                bbox_area = w_norm * h_norm * w * h
                                split_analysis['bbox_areas'].append(bbox_area)
                                
                                split_analysis['num_annotations'] += 1
                                
                    except Exception as e:
                        self.logger.warning(f"Error reading label {label_file}: {e}")
            
            analysis[split] = split_analysis
            
        # Combine statistics
        total_class_counts = {}
        total_images = 0
        total_annotations = 0
        all_image_sizes = []
        all_bbox_areas = []
        
        for split_data in analysis.values():
            if isinstance(split_data, dict) and 'num_images' in split_data:
                total_images += split_data['num_images']
                total_annotations += split_data['num_annotations']
                all_image_sizes.extend(split_data['image_sizes'])
                all_bbox_areas.extend(split_data['bbox_areas'])
                
                for class_id, count in split_data['class_counts'].items():
                    total_class_counts[class_id] = total_class_counts.get(class_id, 0) + count
        
        analysis['overall'] = {
            'total_images': total_images,
            'total_annotations': total_annotations,
            'class_distribution': total_class_counts,
            'avg_annotations_per_image': total_annotations / total_images if total_images > 0 else 0,
            'image_size_stats': {},
            'bbox_area_stats': {}
        }
        
        # Calculate statistics if numpy is available
        if np is not None and all_image_sizes:
            analysis['overall']['image_size_stats'] = {
                'mean_width': float(np.mean([size[0] for size in all_image_sizes])),
                'mean_height': float(np.mean([size[1] for size in all_image_sizes])),
                'min_width': min([size[0] for size in all_image_sizes]),
                'max_width': max([size[0] for size in all_image_sizes]),
                'min_height': min([size[1] for size in all_image_sizes]),
                'max_height': max([size[1] for size in all_image_sizes]),
            }
            
        if np is not None and all_bbox_areas:
            analysis['overall']['bbox_area_stats'] = {
                'mean_area': float(np.mean(all_bbox_areas)),
                'median_area': float(np.median(all_bbox_areas)),
                'min_area': min(all_bbox_areas),
                'max_area': max(all_bbox_areas),
            }
        
        self.logger.info("Dataset analysis completed")
        return analysis
        
    def save_analysis(self, analysis: Dict, output_path: str = None) -> None:
        """Save dataset analysis to file"""
        if output_path is None:
            output_path = self.data_dir / "dataset_analysis.json"
        else:
            output_path = Path(output_path)
            
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if np is not None:
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
            return obj
            
        # Deep convert the analysis dictionary
        def deep_convert(d):
            if isinstance(d, dict):
                return {k: deep_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [deep_convert(v) for v in d]
            else:
                return convert_numpy(d)
        
        converted_analysis = deep_convert(analysis)
        
        with open(output_path, 'w') as f:
            json.dump(converted_analysis, f, indent=2)
            
        self.logger.info(f"Analysis saved to {output_path}")


def main():
    """Main function for data preparation"""
    parser = argparse.ArgumentParser(description="Dataset Preparation Tool")
    parser.add_argument("--action", choices=["download", "validate", "analyze", "create-yaml"], 
                       required=True, help="Action to perform")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset")
    parser.add_argument("--class-names", type=str, nargs="+", help="List of class names")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    preparator = DatasetPreparator()
    
    if args.action == "download":
        dataset_path = preparator.download_coco128()
        print(f"Dataset downloaded to: {dataset_path}")
        
    elif args.action == "validate":
        if not args.dataset_path:
            print("Error: --dataset-path required for validation")
            return 1
            
        results = preparator.validate_dataset(args.dataset_path)
        print(f"Validation results: {json.dumps(results, indent=2)}")
        
    elif args.action == "analyze":
        if not args.dataset_path:
            print("Error: --dataset-path required for analysis")
            return 1
            
        analysis = preparator.analyze_dataset(args.dataset_path)
        preparator.save_analysis(analysis, args.output)
        print("Analysis completed and saved")
        
    elif args.action == "create-yaml":
        if not args.dataset_path or not args.class_names:
            print("Error: --dataset-path and --class-names required")
            return 1
            
        yaml_path = preparator.create_dataset_yaml(
            args.dataset_path, args.class_names, args.output
        )
        print(f"Dataset YAML created: {yaml_path}")
        
    return 0


if __name__ == "__main__":
    exit(main())
