#!/usr/bin/env python3
"""
Production-ready Trout Egg Counter

This script processes images of trout eggs using a trained Roboflow model to
detect and count different types of eggs (eyed, blank, dead). It's designed for
high-performance computing environments and SLURM deployment.

Author: Converted from Jupyter notebook
Date: 2025
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Optional, Tuple
import traceback

import supervision as sv
from inference import get_roboflow_model
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


class EggCounterConfig:
    """Configuration class for the egg counter."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.model_id = "egg_training-bi/1"
        self.api_key = "l6XPyOniqM4Ecq129cpf"
        self.confidence_threshold = 0.45
        self.iou_threshold = 0.5
        self.slice_size = (640, 640)
        self.qr_scale_x = 1.6
        self.qr_scale_y = 2.25
        self.output_image_size = (4000, 6000)
        self.max_workers = mp.cpu_count()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        config_dict = {
            attr: getattr(self, attr) for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


class EggCounter:
    """Main egg counting class."""
    
    def __init__(self, config: EggCounterConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Roboflow model."""
        try:
            self.logger.info(f"Loading model: {self.config.model_id}")
            self.model = get_roboflow_model(
                model_id=self.config.model_id,
                api_key=self.config.api_key
            )
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _callback(self, image_slice: np.ndarray) -> sv.Detections:
        """Callback function for inference on image slices."""
        try:
            results = self.model.infer(
                image_slice,
                confidence=self.config.confidence_threshold,
                iou_threshold=self.config.iou_threshold
            )[0]
            return sv.Detections.from_inference(results)
        except Exception as e:
            self.logger.error(f"Error in inference callback: {e}")
            return sv.Detections.empty()
    
    def _process_qr_codes(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Process QR codes in the image and create mask."""
        qcd = cv2.QRCodeDetector()
        retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(image)
        
        if not retval:
            # No QR code found, use filename as identifier
            return image, []
        
        try:
            # Calculate the center of the QR code
            center = np.mean(points, axis=1)
            
                                      # Expand the points by the specified factors
             expanded_points = np.copy(points)
             expanded_points[:, :, 0] = (
                 self.config.qr_scale_x * (points[:, :, 0] - center[:, 0]) +
                 center[:, 0]
             )
             expanded_points[:, :, 1] = (
                 self.config.qr_scale_y * (points[:, :, 1] - center[:, 1]) +
                 center[:, 1]
             )
            
            # Convert points to integer coordinates
            expanded_points = expanded_points.astype(int)
            
            # Create a mask with the same dimensions as the image
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            
            # Fill the mask with the expanded points to exclude QR code area
            cv2.fillPoly(mask, expanded_points, 0)
            
            # Apply the mask to the image
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            
            self.logger.debug(f"QR codes processed: {decoded_info}")
            return masked_image, decoded_info
            
        except Exception as e:
            self.logger.warning(f"Error processing QR codes: {e}")
            return image, []
    
    def _run_inference(self, image: np.ndarray) -> sv.Detections:
        """Run inference on the image using slicing."""
        try:
            slicer = sv.InferenceSlicer(
                callback=self._callback,
                slice_wh=self.config.slice_size
            )
            return slicer(image=image)
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            return sv.Detections.empty()
    
    def _save_annotated_image(self, image: np.ndarray, detections: sv.Detections,
                             output_path: str):
        """Save annotated image with detection boxes."""
        try:
            box_annotator = sv.BoxAnnotator()
            annotated_image = box_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            
            # Convert BGR to RGB for PIL
            image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            annotated_image_pil = Image.fromarray(image_rgb)
            
            # Resize if specified
            if self.config.output_image_size:
                annotated_image_pil = annotated_image_pil.resize(self.config.output_image_size)
            
            annotated_image_pil.save(output_path)
            self.logger.debug(f"Annotated image saved: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving annotated image {output_path}: {e}")
    
    def _create_detection_dataframe(self, detections: sv.Detections,
                                   identifier: str) -> pd.DataFrame:
        """Create DataFrame from detections."""
        try:
            if len(detections) == 0:
                # Return empty DataFrame with expected columns
                df = pd.DataFrame({identifier: [0]})
                df = df.assign(total_eggs=0)
                return df
            
            class_names = detections['class_name']
            class_id = detections.class_id
            confidence = detections.confidence
            
            # Create DataFrame and count by class
            detection_df = pd.DataFrame({
                'ID': class_names,
                'class_id': class_id,
                'confidence': confidence
            })
            
            # Count detections by class
            counts_df = (detection_df
                        .value_counts(subset='ID')
                        .to_frame(identifier)
                        .T
                        .assign(total_eggs=len(detections)))
            
            return counts_df
            
        except Exception as e:
            self.logger.error(f"Error creating detection DataFrame: {e}")
            # Return empty DataFrame
            df = pd.DataFrame({identifier: [0]})
            return df.assign(total_eggs=0)
    
    def count_eggs(self, image_path: str, output_dir: str) -> pd.DataFrame:
        """Process a single image and count eggs."""
        try:
            self.logger.info(f"Processing image: {image_path}")
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Process QR codes and get identifier
            processed_image, decoded_info = self._process_qr_codes(image)
            
            # Use QR code info or filename as identifier
            if decoded_info:
                identifier = decoded_info[0]
            else:
                identifier = os.path.basename(image_path)
            
            # Run inference
            detections = self._run_inference(processed_image)
            
            # Save annotated image
            output_image_path = os.path.join(output_dir, f"{identifier}.png")
            self._save_annotated_image(processed_image, detections, output_image_path)
            
            # Create results DataFrame
            results_df = self._create_detection_dataframe(detections, identifier)
            
            self.logger.info(f"Completed processing: {identifier} - {len(detections)} detections")
            return results_df
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            identifier = os.path.basename(image_path)
            # Return empty result
            df = pd.DataFrame({identifier: [0]})
            return df.assign(total_eggs=0)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("egg_counter")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_image_paths(input_dir: str, extensions: List[str] = None) -> List[str]:
    """Get all image paths from input directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    image_paths = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    for ext in extensions:
        image_paths.extend(list(input_path.glob(f"*{ext}")))
    
    return [str(p) for p in sorted(image_paths)]


def process_images_parallel(egg_counter: EggCounter, image_paths: List[str],
                          output_dir: str, max_workers: int = None) -> List[pd.DataFrame]:
    """Process images in parallel."""
    if max_workers is None:
        max_workers = egg_counter.config.max_workers
    
    results = []
    failed_images = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(egg_counter.count_eggs, img_path, output_dir): img_path
            for img_path in image_paths
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_path), 
                          total=len(image_paths), 
                          desc="Processing images"):
            img_path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                egg_counter.logger.error(f"Failed to process {img_path}: {e}")
                failed_images.append(img_path)
    
    if failed_images:
        egg_counter.logger.warning(f"Failed to process {len(failed_images)} images")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Production Trout Egg Counter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python count_eggs_production.py -i /path/to/images -o /path/to/output
  python count_eggs_production.py -i /path/to/images -o /path/to/output --config config.json
  python count_eggs_production.py -i /path/to/images -o /path/to/output --max-workers 8
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Directory for output files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config",
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--csv-output",
        help="Path for CSV output file (default: output_dir/egg_count_results.csv)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of worker threads"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process images sequentially instead of in parallel"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting Trout Egg Counter")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory for annotated images
        images_output_dir = output_dir / "annotated_images"
        images_output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        config = EggCounterConfig(args.config)
        
        # Override config with command line arguments
        if args.max_workers:
            config.max_workers = args.max_workers
        if args.confidence:
            config.confidence_threshold = args.confidence
        
        logger.info(f"Configuration: {vars(config)}")
        
        # Initialize egg counter
        egg_counter = EggCounter(config, logger)
        
        # Get image paths
        image_paths = get_image_paths(args.input_dir)
        logger.info(f"Found {len(image_paths)} images to process")
        
        if not image_paths:
            logger.warning("No images found in input directory")
            return
        
        # Process images
        if args.sequential:
            logger.info("Processing images sequentially")
            results = []
            for img_path in tqdm(image_paths, desc="Processing images"):
                try:
                    result = egg_counter.count_eggs(img_path, str(images_output_dir))
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue
        else:
            logger.info(f"Processing images in parallel with {config.max_workers} workers")
            results = process_images_parallel(
                egg_counter, image_paths, str(images_output_dir), config.max_workers
            )
        
        # Combine results
        if results:
            logger.info("Combining results")
            final_df = pd.concat(results, ignore_index=False)
            
            # Save results
            csv_path = args.csv_output or str(output_dir / "egg_count_results.csv")
            final_df.to_csv(csv_path, index=True)
            logger.info(f"Results saved to: {csv_path}")
            
            # Print summary
            total_images = len(final_df)
            total_eggs = final_df['total_eggs'].sum()
            logger.info(f"Summary: {total_images} images processed, {total_eggs} total eggs detected")
            
            # Print detailed results
            print("\nDetection Summary:")
            print(final_df.to_string())
        else:
            logger.warning("No results to save")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
    
    logger.info("Egg counting completed successfully")


if __name__ == "__main__":
    main() 
