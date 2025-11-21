# Production Device Segmentation Pipeline

This repository contains a production-grade Deep Learning pipeline for semantic segmentation of medical devices (patches) on human skin. It features a hybrid architecture using YOLO for ROI localization (Smart Cropping) and Unet++ for high-precision segmentation.

**Version:** 1.0.0
**Status:** Production Ready

## 📂 Project Structure

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
````

## 🚀 Quick Start

### 1\. Installation

Ensure you have Python 3.8+ and a CUDA-capable GPU.

```bash
pip install -r requirements.txt
```

### 2\. Training

The training script handles data preprocessing, training, validation, and ONNX export automatically.

**Development Mode (Kaggle/Local):**
Runs using the default hardcoded paths defined in `config.py`.

```bash
python train.py
```

**Production Mode (Docker/Cloud):**
Override paths using environment variables.

```bash
export RAW_IMG_DIR="/data/images"
export RAW_JSON_PATH="/data/labels.json"
export WORK_DIR="/output"
python train.py
```

### 3\. Inference

Run the standalone inference engine on new images using the exported ONNX model.

```bash
python inference.py --test_dir /path/to/test/images --onnx_path /path/to/model.onnx
```

## 🧠 Algorithmic Methodology

### 1\. Smart Preprocessing (The "Zoom-In" Strategy)

To handle high-resolution inputs without losing fine-grained details, we employ a two-stage approach:

1.  **Person Detection:** A lightweight YOLO model (`yolo11n-seg`) detects the person in the frame.
2.  **ROI Calculation:** The algorithm identifies the "Head Top" coordinate to dynamically calculate a Region of Interest (ROI) centered on the upper body.
3.  **Crop & Pad:** The ROI is square-cropped and padded to preserve the aspect ratio before resizing to `512x512`.

### 2\. Segmentation Model

  * **Architecture:** Unet++ (Nested U-Net) for dense skip connections.
  * **Encoder:** ResNet34 (Pretrained on ImageNet) for feature extraction.
  * **Loss Function:** `ComboLoss` (50% Binary Cross Entropy + 50% Dice Loss).
  * **Optimization:** AdamW with Cosine Annealing Warm Restarts.

### 3\. Test Time Augmentation (TTA)

During inference and validation, prediction stability is improved by averaging the output of the original image and its horizontally flipped version. This reduces variance and edge errors.

$$P_{final} = \frac{Model(x) + Flip(Model(Flip(x)))}{2}$$

## ⚙️ Configuration

All hyperparameters are defined in `config.py`. You can override paths using environment variables for containerized deployment.

| Variable | Default (Kaggle) | Description |
| :--- | :--- | :--- |
| `RAW_IMG_DIR` | `/kaggle/input/...` | Directory containing raw images |
| `RAW_JSON_PATH` | `/kaggle/input/...` | Path to Label Studio JSON file |
| `WORK_DIR` | `/kaggle/working/...` | Output directory for logs/models |
| `BATCH_SIZE` | `12` | Training batch size |
| `INPUT_SIZE` | `512` | Input resolution (HxW) |
| `ARCH` | `UnetPlusPlus` | Segmentation architecture |
| `ENCODER` | `resnet34` | Backbone encoder |

## 📊 Metrics & Logging

The pipeline automatically tracks metrics in `WORK_DIR/logs/`:

  * **training.log**: System logs, error traces, and epoch summaries.
  * **metrics.json**: Structured JSON data containing Training Loss and Validation IoU per epoch.

**Performance Target:**

  * **Metric:** Jaccard Index (IoU)
  * **Target:** \> 0.90 IoU on Validation set.

## 🛠 Troubleshooting

**1. "CUDA out of memory"**

  * Reduce `BATCH_SIZE` in `config.py`.
  * Ensure `gc.collect()` is active in `dataset.py` (enabled by default).

**2. "No Patch Detected" in Inference**

  * The inference engine uses adaptive thresholding. If the model confidence is below 0.15, it will report no patch found. Check the lighting conditions of input images or model convergence.

**3. "Kernel Stopping" / Crash**

  * The preprocessing step writes intermediate cropped images to disk to conserve RAM. Ensure your `WORK_DIR` has sufficient write permissions and storage space.

## 📦 Deployment Notes

  * **Docker:** The code is environment-agnostic. Mount your data volumes to the paths defined in `config.py` via `os.getenv`.
  * **ONNX:** The training script automatically exports `device_segmentation.onnx`. This file allows deployment on Triton Inference Server or Edge devices without requiring the full PyTorch dependency tree.

<!-- end list -->


```