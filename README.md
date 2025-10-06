# Trout Egg Counter: Production-Ready Computer Vision for Aquaculture

A high-performance, production-grade computer vision system for automated detection and counting of trout eggs using deep learning. This system leverages the Roboflow inference engine to identify and classify different egg states (eyed, blank, dead) with configurable confidence thresholds and advanced QR code masking capabilities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Performance Optimization](#performance-optimization)
- [SLURM Deployment](#slurm-deployment)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

Manual counting of trout eggs represents a significant bottleneck in aquaculture operations, requiring extensive labor hours and introducing human error into hatchery management. This automated computer vision system addresses these challenges by providing rapid, accurate, and reproducible egg counting across large image datasets.

### Key Capabilities

The system processes images of trout eggs to detect three critical egg states:
- **Eyed eggs**: Fertilized eggs showing visible eye development
- **Blank eggs**: Unfertilized or early-stage eggs without eye development
- **Dead eggs**: Non-viable eggs requiring removal

### Technical Foundation

Built on the Roboflow object detection platform, the system employs a trained neural network model that achieves high accuracy through:
- Configurable confidence thresholding for precision-recall optimization
- Intelligent image slicing for processing high-resolution photographs
- Advanced QR code detection and masking to prevent counting artifacts
- Parallel processing architecture for high-throughput analysis

## Features

### Core Functionality

- **Multi-class Detection**: Simultaneously identifies and counts eyed, blank, and dead eggs
- **High-Resolution Processing**: Handles large images through intelligent slicing (default 640×640 patches)
- **QR Code Intelligence**: Automatically detects and masks QR codes with configurable expansion factors
- **Batch Processing**: Processes entire directories with parallel execution
- **Annotated Outputs**: Generates visualizations with bounding boxes for quality control
- **Flexible Configuration**: JSON-based configuration system for easy parameter tuning

### Production Features

- **SLURM Integration**: Designed for high-performance computing environments
- **Logging System**: Comprehensive logging with configurable verbosity
- **Error Handling**: Robust error recovery with detailed traceback reporting
- **Progress Tracking**: Real-time progress bars for long-running operations
- **CSV Export**: Structured data output for downstream analysis
- **Configurable Parallelism**: Adjustable worker threads based on available resources

### Quality Assurance

- **Detection Confidence Scores**: Per-detection confidence values for quality filtering
- **Visual Verification**: Annotated images enable manual validation of results
- **Reproducible Results**: Deterministic processing with documented parameters
- **Extensible Architecture**: Modular design supports easy customization

## System Architecture

### Processing Pipeline

The system implements a multi-stage processing pipeline:

1. **Image Loading**: Reads images from specified directory using OpenCV
2. **QR Code Processing**: Detects QR codes, extracts identifiers, creates expansion masks
3. **Image Masking**: Applies masks to prevent detection in QR code regions
4. **Inference Slicing**: Divides images into overlapping tiles for processing
5. **Detection Aggregation**: Combines detections across tiles with NMS
6. **Result Compilation**: Generates CSV summaries and annotated visualizations

### Mathematical Foundation

The detection system operates through a convolutional neural network trained on labeled trout egg images. Key mathematical operations include:

- **Confidence Thresholding**: Filters detections based on model confidence scores
- **IoU-based NMS**: Eliminates duplicate detections using intersection-over-union calculations
- **Coordinate Transformation**: Maps detections from slice coordinates to full image space
- **Geometric Scaling**: Expands QR code boundaries using configurable scaling factors

### Parallel Architecture

The system achieves high throughput through ThreadPoolExecutor-based parallelism:
- Concurrent image processing across multiple worker threads
- Progress tracking with tqdm for real-time monitoring
- Exception handling preserves partial results from failed images
- Automatic CPU count detection for optimal worker allocation

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for accelerated inference)
- Sufficient RAM for image processing (minimum 8GB recommended)

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trout-egg-counter.git
cd trout-egg-counter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### HPC/SLURM Installation

```bash
# Load required modules
module load python/3.8
module load cuda/11.8  # Optional, for GPU acceleration

# Create virtual environment in scratch space
python -m venv $SCRATCH/trout-egg-env
source $SCRATCH/trout-egg-env/bin/activate

# Install dependencies
pip install --no-cache-dir -r requirements.txt
```

### Dependencies

Core dependencies (see `requirements.txt` for complete list):

```
supervision>=0.16.0
inference>=0.9.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
tqdm>=4.65.0
```

## Quick Start

### Basic Usage

Process a directory of images with default settings:

```bash
python count_eggs_production.py \
  --input-dir /path/to/images \
  --output-dir /path/to/results
```

### With Custom Configuration

```bash
python count_eggs_production.py \
  --input-dir /path/to/images \
  --output-dir /path/to/results \
  --config config.json \
  --confidence 0.5 \
  --max-workers 8
```

### Sequential Processing

For debugging or memory-constrained environments:

```bash
python count_eggs_production.py \
  --input-dir /path/to/images \
  --output-dir /path/to/results \
  --sequential \
  --log-level DEBUG
```

## Usage

### Command-Line Interface

The system provides a comprehensive command-line interface:

```bash
python count_eggs_production.py [OPTIONS]
```

#### Required Arguments

- `-i, --input-dir PATH`: Directory containing input images
- `-o, --output-dir PATH`: Directory for output files

#### Optional Arguments

- `--config PATH`: Path to JSON configuration file
- `--csv-output PATH`: Custom path for CSV results (default: output_dir/egg_count_results.csv)
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging verbosity (default: INFO)
- `--log-file PATH`: Write logs to file
- `--max-workers N`: Maximum worker threads (default: CPU count)
- `--confidence FLOAT`: Detection confidence threshold (default: 0.45)
- `--sequential`: Process images sequentially instead of parallel

### Programmatic Usage

Import and use the system in Python code:

```python
from count_eggs_production import EggCounter, EggCounterConfig
import logging

# Configure system
config = EggCounterConfig()
config.confidence_threshold = 0.5
config.max_workers = 4

# Initialize
logger = logging.getLogger("egg_counter")
counter = EggCounter(config, logger)

# Process single image
results_df = counter.count_eggs(
    image_path="image.jpg",
    output_dir="./results"
)

print(f"Total eggs detected: {results_df['total_eggs'].values[0]}")
```

## Configuration

### JSON Configuration File

Create a `config.json` file to customize system behavior:

```json
{
  "model_id": "egg_training-bi/1",
  "api_key": "your_roboflow_api_key",
  "confidence_threshold": 0.45,
  "iou_threshold": 0.5,
  "slice_size": [640, 640],
  "qr_scale_x": 1.6,
  "qr_scale_y": 2.25,
  "output_image_size": [4000, 6000],
  "max_workers": 8
}
```

### Configuration Parameters

#### Model Configuration

- `model_id`: Roboflow model identifier (format: "workspace/version")
- `api_key`: Your Roboflow API key for model access
- `confidence_threshold`: Minimum detection confidence (0.0-1.0, default: 0.45)
- `iou_threshold`: IoU threshold for non-maximum suppression (default: 0.5)

#### Processing Configuration

- `slice_size`: Dimensions for image slicing as [width, height] (default: [640, 640])
- `max_workers`: Number of parallel worker threads (default: CPU count)
- `output_image_size`: Resize dimensions for annotated images (default: [4000, 6000])

#### QR Code Configuration

- `qr_scale_x`: Horizontal expansion factor for QR code masking (default: 1.6)
- `qr_scale_y`: Vertical expansion factor for QR code masking (default: 2.25)

### Optimizing Confidence Threshold

The confidence threshold balances precision and recall:

- **Higher values (0.6-0.8)**: Fewer false positives, may miss some eggs
- **Lower values (0.3-0.4)**: Captures more eggs, increased false positives
- **Recommended approach**: Start at 0.45, adjust based on validation results

### QR Code Masking

QR codes in images are automatically detected and masked to prevent false detections. The scaling factors control mask size:

- `qr_scale_x = 1.6`: Expands mask 60% beyond QR code horizontally
- `qr_scale_y = 2.25`: Expands mask 125% beyond QR code vertically

Adjust these values if QR codes contain useful counting areas or if masks are too aggressive.

## Output Format

### Directory Structure

Processing creates the following output structure:

```
output_dir/
├── egg_count_results.csv          # Detection summary
├── annotated_images/               # Visual outputs
│   ├── [identifier_1].png
│   ├── [identifier_2].png
│   └── ...
└── processing.log                  # Detailed logs (if --log-file specified)
```

### CSV Format

The results CSV contains detection counts by egg type:

```csv
,blank,dead,eyed,total_eggs
QR123,45,12,203,260
QR124,38,8,198,244
QR125,52,15,210,277
```

Columns:
- **Index**: Image identifier (from QR code or filename)
- **blank**: Count of blank/unfertilized eggs
- **dead**: Count of dead/non-viable eggs
- **eyed**: Count of eyed/fertilized eggs
- **total_eggs**: Total detections across all classes

### Annotated Images

Each annotated image shows:
- Bounding boxes around detected eggs
- Color-coded by egg type (if supported by supervision library)
- Original image with detection overlays
- Resized to configured output dimensions

## Performance Optimization

### Parallel Processing

Maximize throughput by tuning worker count:

```bash
# Use all available CPUs
python count_eggs_production.py -i images/ -o results/

# Limit to 4 workers for memory-constrained systems
python count_eggs_production.py -i images/ -o results/ --max-workers 4
```

### Memory Management

For large images or limited RAM:

1. **Reduce slice size**: Smaller slices reduce memory per operation
2. **Sequential processing**: Use `--sequential` flag to process one image at a time
3. **Adjust output size**: Smaller annotated images reduce memory requirements

```python
config = EggCounterConfig()
config.slice_size = (512, 512)  # Smaller slices
config.output_image_size = (2000, 3000)  # Smaller outputs
```

### GPU Acceleration

The Roboflow inference engine automatically uses GPU if available:

```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Process with GPU acceleration (automatic)
python count_eggs_production.py -i images/ -o results/
```

### Batch Processing Strategy

For very large datasets:

1. **Divide into batches**: Process 100-500 images per batch
2. **Monitor system resources**: Adjust worker count based on CPU/memory usage
3. **Checkpoint progress**: Process subdirectories separately for fault tolerance

## SLURM Deployment

### Example SLURM Script

Create `run_egg_counting.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=egg_count
#SBATCH --output=egg_count_%j.out
#SBATCH --error=egg_count_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=general

# Load modules
module load python/3.8
module load cuda/11.8

# Activate environment
source $SCRATCH/trout-egg-env/bin/activate

# Run processing
python count_eggs_production.py \
  --input-dir $SCRATCH/trout_images \
  --output-dir $SCRATCH/results \
  --log-file $SCRATCH/results/processing.log \
  --max-workers 16 \
  --confidence 0.45

# Copy results to permanent storage
cp -r $SCRATCH/results $HOME/trout_analysis/
```

### Submit Job

```bash
sbatch run_egg_counting.sh
```

### Monitor Progress

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f egg_count_JOBID.out

# Monitor resource usage
sstat -j JOBID --format=JobID,MaxRSS,AveCPU
```

## API Reference

### EggCounterConfig

Configuration container for system parameters.

```python
config = EggCounterConfig(config_path="config.json")
```

**Methods:**
- `load_from_file(config_path)`: Load configuration from JSON
- `save_to_file(config_path)`: Save configuration to JSON

**Attributes:**
- `model_id`: Roboflow model identifier
- `api_key`: Roboflow API key
- `confidence_threshold`: Detection confidence threshold
- `iou_threshold`: NMS IoU threshold
- `slice_size`: Image slice dimensions
- `qr_scale_x`: QR mask horizontal scaling
- `qr_scale_y`: QR mask vertical scaling
- `output_image_size`: Annotated image dimensions
- `max_workers`: Maximum parallel workers

### EggCounter

Main processing class for egg detection and counting.

```python
counter = EggCounter(config, logger)
```

**Methods:**

#### count_eggs(image_path, output_dir)

Process single image and return detection results.

**Parameters:**
- `image_path` (str): Path to input image
- `output_dir` (str): Directory for output files

**Returns:**
- `pandas.DataFrame`: Detection counts by egg type

**Example:**
```python
results = counter.count_eggs("image.jpg", "./output")
print(results)
```

### Utility Functions

#### setup_logging(log_level, log_file)

Configure logging system.

**Parameters:**
- `log_level` (str): Logging level (DEBUG/INFO/WARNING/ERROR)
- `log_file` (str, optional): Path to log file

**Returns:**
- `logging.Logger`: Configured logger instance

#### get_image_paths(input_dir, extensions)

Retrieve all image paths from directory.

**Parameters:**
- `input_dir` (str): Directory containing images
- `extensions` (list, optional): Image file extensions to include

**Returns:**
- `list`: Sorted list of image paths

#### process_images_parallel(egg_counter, image_paths, output_dir, max_workers)

Process multiple images in parallel.

**Parameters:**
- `egg_counter` (EggCounter): Configured counter instance
- `image_paths` (list): List of image paths
- `output_dir` (str): Output directory
- `max_workers` (int, optional): Number of workers

**Returns:**
- `list`: List of result DataFrames

## Troubleshooting

### Common Issues

#### Model Loading Failures

**Problem:** `Failed to load model: Authentication error`

**Solution:** Verify your Roboflow API key is correct:
```bash
python count_eggs_production.py --config config.json ...
```

Ensure `api_key` in configuration matches your Roboflow account.

#### Memory Errors

**Problem:** `MemoryError` or system slowdown during processing

**Solutions:**
1. Reduce worker count: `--max-workers 2`
2. Use sequential processing: `--sequential`
3. Reduce slice size in configuration
4. Process smaller batches of images

#### No QR Codes Detected

**Problem:** Images processed but QR codes not recognized

**Solutions:**
- Ensure QR codes have sufficient contrast and size
- Verify QR codes are not damaged or obscured
- Check that images contain valid QR code formats
- Image filenames will be used as identifiers if QR detection fails

#### Low Detection Accuracy

**Problem:** System misses eggs or reports too many false positives

**Solutions:**
1. Adjust confidence threshold:
   - Lower for more detections: `--confidence 0.35`
   - Higher for fewer false positives: `--confidence 0.55`
2. Verify image quality meets model requirements
3. Check that QR code masking is not too aggressive
4. Consider retraining model with additional examples

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
python count_eggs_production.py \
  --input-dir images/ \
  --output-dir results/ \
  --log-level DEBUG \
  --log-file debug.log
```

### Getting Help

If issues persist:

1. Check the log files for detailed error messages
2. Verify all dependencies are correctly installed
3. Test with a small subset of images first
4. Open an issue on GitHub with:
   - Complete error message
   - System configuration
   - Sample images (if possible)
   - Steps to reproduce

## Contributing

We welcome contributions to improve the Trout Egg Counter system.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/trout-egg-counter.git
cd trout-egg-counter

# Create development environment
python -m venv dev-env
source dev-env/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Add docstrings for all public functions and classes
- Maintain test coverage above 80%
- Run black formatter before committing

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with clear commit messages
4. Add tests for new functionality
5. Update documentation as needed
6. Push to your fork and submit a pull request

### Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=count_eggs_production tests/

# Run specific test file
pytest tests/test_egg_counter.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

Permission is granted to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to including the copyright notice and permission notice in all copies or substantial portions of the software.

The software is provided "as is", without warranty of any kind.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{trout_egg_counter,
  title={Trout Egg Counter: Production-Ready Computer Vision for Aquaculture},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/trout-egg-counter}
}
```

## Acknowledgments

This project builds upon several excellent open-source tools:

- **Roboflow**: Computer vision platform and inference engine
- **Supervision**: Detection utilities and visualization tools
- **OpenCV**: Computer vision and image processing
- **NumPy/Pandas**: Numerical computing and data analysis

Special thanks to the aquaculture research community for providing the biological context and validation datasets that made this system possible.

## Contact

For questions, suggestions, or collaborations:

- **Email**: your.email@institution.edu
- **GitHub Issues**: [Report issues](https://github.com/yourusername/trout-egg-counter/issues)
- **Documentation**: [Full documentation](https://yourusername.github.io/trout-egg-counter)

---

**Note**: This system is designed for research purposes. For production aquaculture deployment, consult with domain experts to validate accuracy for your specific trout species and imaging conditions.
