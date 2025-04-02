# LandSurvey - Drone Image Object Detection

A PyTorch-based project for detecting and measuring objects in drone aerial imagery.

## Project Overview

This project implements multiple object detection models specialized for drone imagery, with capabilities to:

1. Detect various objects in aerial imagery
2. Calculate physical dimensions of detected objects using GSD (Ground Sample Distance)
3. Handle perspective distortion challenges in aerial photography
4. Compare performance across different neural network architectures

## Features

- Multiple model architectures including Faster R-CNN, RetinaNet, and YOLO
- Ground sample distance (GSD) integration for physical measurements
- Visualization tools for detection results
- Training pipeline with data augmentation
- Evaluation metrics focused on aerial imagery challenges

## Project Structure

```
├── models/              # Different neural network model implementations
├── dataset.py           # Dataset handling and processing
├── transforms.py        # Image transformations and augmentations
├── engine.py            # Training and evaluation logic
├── utils.py             # Utility functions
├── main.py              # Training entry point
├── detect.py            # Inference script
└── requirements.txt     # Project dependencies
```

## Key Challenges

1. **Data Acquisition**: Obtaining appropriate sample images that match real-world scenarios and creating accurate annotations.
2. **Training Efficiency**: Determining optimal training parameters to achieve good loss convergence and accuracy.
3. **Perspective Distortion**: Addressing variations in object appearance due to drone altitude and angle.
4. **Scale Variance**: Objects appearing at different scales due to varying flight altitudes.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python main.py --data_path /path/to/dataset --model faster_rcnn --epochs 10
```

### Inference

```bash
python detect.py --image_path /path/to/image.jpg --model_path /path/to/model.pth --gsd 0.05
```

## Future Work

- Implement additional models for comparison
- Add perspective distortion correction
- Integrate with drone flight planning software
- Real-time detection capabilities
