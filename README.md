````markdown
# Production Device Segmentation Pipeline

This repository contains a production-grade Deep Learning pipeline for semantic segmentation of medical devices (patches) on human skin. It features a modular architecture designed for reproducibility, incorporating a "Smart Cropping" stage (YOLO) followed by high-precision segmentation (Unet++).

**Status:** Production Ready
**Python Version:** 3.8+

---

## Project Structure

The codebase is strictly separated into Configuration, ETL, Modeling, Training, and Inference modules.

```text
.
├── config.py           # Central Configuration (Env Vars & Hyperparameters)
├── dataset.py          # ETL: YOLO Smart Cropping, caching to disk, & Augmentation
├── model.py            # Architecture Definition (Unet++) & ComboLoss
├── train.py            # Training orchestration, Validation, & MLOps
├── inference.py        # Standalone ONNX Inference Engine
├── utils.py            # Utilities: Logging, Seeding, Metric Tracking
└── requirements.txt    # Strict dependency versions
````

-----

## Quick Start

### 1\. Prerequisites & Installation

Ensure you have a CUDA-capable GPU.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2\. Training the Model

The training pipeline (`train.py`) performs the following steps automatically:

1.  **Sanity Checks:** Verifies model architecture and augmentations.
2.  **ETL:** Loads images/JSON, runs YOLO cropping, and **caches processed crops to disk** (to prevent RAM overflow).
3.  **Training:** Runs 5-Fold Cross-Validation with `UnetPlusPlus`.
4.  **Export:** Saves the best model as `device_segmentation.onnx`.

**Option A: Development (Kaggle/Local Defaults)**
Runs using the paths hardcoded as defaults in `config.py`.

```bash
python train.py
```

**Option B: Production (Docker/Cloud)**
Override specific paths using Environment Variables without changing the code.

```bash
# Example Overrides
export RAW_IMG_DIR="/mnt/data/raw_images"
export RAW_JSON_PATH="/mnt/data/annotations.json"
export WORK_DIR="/mnt/output"
export BATCH_SIZE="16"

python train.py
```

### 3\. Running Inference

The inference script (`inference.py`) is standalone. It uses the exported ONNX model and does not require the training libraries.

```bash
# Run inference on a specific folder of images
python inference.py \
  --onnx_path /kaggle/working/prod_pipeline_v1/device_segmentation.onnx \
  --test_dir /kaggle/input/sample3
```

-----

## Technical Methodology

### Stage 1: Smart Preprocessing (ETL)

  * **Problem:** High-resolution images make small devices hard to detect if resized directly.
  * **Solution:** We use **YOLOv11-seg** (`ultralytics`) to detect the person.
  * **Logic:**
    1.  Detect Person mask.
    2.  Calculate the "Head Top" and torso center.
    3.  Create a dynamic square ROI based on torso size.
    4.  Crop, Pad, and Resize to `512x512`.
    5.  **Memory Safety:** Crops are saved to `WORK_DIR/images` to keep RAM usage low.

### Stage 2: Segmentation Network

  * **Architecture:** `UnetPlusPlus` (Nested U-Net) which reduces the semantic gap between encoder and decoder.
  * **Backbone:** `resnet34` (Pretrained on ImageNet).
  * **Loss Function:** `ComboLoss`
      * 50% **Dice Loss** (Optimizes Overlap/IoU)
      * 50% **BCEWithLogitsLoss** (Optimizes Pixel Accuracy)
  * **Optimization:** `AdamW` optimizer with `CosineAnnealingWarmRestarts` scheduler.

### Stage 3: Inference Logic

  * **Engine:** `onnxruntime-gpu` for hardware-accelerated inference.
  * **TTA (Test Time Augmentation):** During validation, predictions are averaged: $(Pred(x) + Pred(HorizontalFlip(x))) / 2$.
  * **Adaptive Thresholding:** The inference engine dynamically scans thresholds (`0.5`, `0.3`, `0.15`) to maximize recall for difficult lighting conditions.

-----

## Configuration Reference (`config.py`)

You can control these parameters via Environment Variables or by editing the default values in `config.py`.

| Environment Variable | Description | Default Value |
| :--- | :--- | :--- |
| `RAW_JSON_PATH` | Path to Label Studio annotation file | *(Kaggle Input Path)* |
| `RAW_IMG_DIR` | Path to folder containing raw images | *(Kaggle Input Path)* |
| `WORK_DIR` | Root path for outputs (models/logs) | `/kaggle/working/prod_pipeline_v1` |
| `ARCH` | Segmentation Architecture | `UnetPlusPlus` |
| `ENCODER` | Backbone Encoder | `resnet34` |
| `INPUT_SIZE` | Image Input Resolution | `512` |
| `BATCH_SIZE` | Training Batch Size | `12` |
| `EPOCHS` | Total Training Epochs | `100` |

-----

## Output Artifacts

After running `train.py`, the `WORK_DIR` will contain:

1.  **`models/`**: Saved PyTorch weights (`.pth`) for each fold.
2.  **`images/` & `masks/`**: The processed, cropped datasets used for training.
3.  **`logs/`**:
      * `training.log`: Detailed system logs.
      * `metrics.json`: JSON formatted loss/IoU history for experiment tracking.
4.  **`device_segmentation.onnx`**: The final production model ready for deployment.

-----

## Troubleshooting

**1. Memory Errors (OOM)**

  * **Training:** Reduce `BATCH_SIZE` in `config.py`.
  * **Preprocessing:** The `dataset.py` includes explicit `del` and `gc.collect()` calls. If kernel dies, ensure you have enough disk space in `WORK_DIR` for the cached images.

**2. ONNX Export Failed**

  * Ensure `model.py` logic matches the weights being loaded. The `export_to_onnx` function in `train.py` expects the model structure to define `classes=1` and `in_channels=3`.

**3. Inference: "No Patch Detected"**

  * The logic strictly requires a confidence score \> 0.15. If the image is blurry or the patch is occluded, the adaptive thresholder will return a safe negative to avoid false positives.

<!-- end list -->

```
```