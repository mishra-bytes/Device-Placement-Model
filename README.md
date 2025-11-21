# Device-Placement-Model

## Production Device Segmentation Pipeline

This repository contains a production-grade Deep Learning pipeline for semantic segmentation of medical devices (patches) on human skin. It features a hybrid architecture using YOLO for ROI localization (Smart Cropping) and Unet++ for high-precision segmentation.

**Version:** 1.0.0
**Status:** Production Ready

### Project Structure

The codebase is modularized to support distinct lifecycle stages: ETL, Training, and Inference.

```text
.
├── config.py           # Centralized configuration (Environment Variables supported)
├── dataset.py          # ETL Pipeline: Smart Cropping (YOLO) & Augmentation
├── model.py            # Model Architecture (Unet++) & Custom Loss Functions
├── train.py            # Training Loop, Validation, and MLOps Logging
├── inference.py        # ONNX Inference Engine (Standalone)
├── utils.py            # Logging, Seeding, and Metric Tracking
└── requirements.txt    # Dependency specifications