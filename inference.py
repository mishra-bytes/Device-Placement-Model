import os
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from ultralytics import YOLO

class PatchPredictor:
    def __init__(self, onnx_model_path, yolo_model_path='yolo11n-seg.pt'):
        print(f"Loading ONNX Model: {onnx_model_path}...")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(str(onnx_model_path), providers=providers)
        except Exception:
            print("CUDA not found. Falling back to CPU.")
            self.session = ort.InferenceSession(str(onnx_model_path), providers=['CPUExecutionProvider'])
            
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = 512
        
        print("Loading YOLO for Smart Cropping...")
        self.yolo = YOLO(yolo_model_path)

    def preprocess(self, image_path):
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None: raise ValueError(f"Image not found: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Smart Crop
        infer_img = img_rgb.copy()
        if max(h, w) > 1500:
            scale = 1500 / max(h, w)
            infer_img = cv2.resize(img_rgb, (0,0), fx=scale, fy=scale)
        
        results = self.yolo(infer_img, verbose=False, retina_masks=False)
        
        person_mask = np.zeros((h, w), dtype=np.uint8)
        if results[0].masks:
            boxes = results[0].boxes
            persons = boxes.cls == 0
            if persons.any():
                idx = results[0].boxes.conf[persons].argmax()
                real_idx = persons.nonzero(as_tuple=True)[0][idx]
                m = results[0].masks.data[real_idx].cpu().numpy()
                m = cv2.resize(m, (w, h))
                person_mask = (m > 0.5).astype(np.uint8)

        if np.sum(person_mask) > 0:
            rows = np.any(person_mask, axis=1)
            y_head_top = np.argmax(rows)
            cols = np.sum(person_mask, axis=0)
            center_x = int(np.dot(np.arange(w), cols) / (np.sum(cols)+1e-6))
        else:
            y_head_top = 0
            center_x = w // 2
        
        roi_height = h // 3
        square_dim = max(int(roi_height * 1.5), 256)
        center_y = y_head_top + (roi_height // 2)

        half = square_dim // 2
        x1, y1 = center_x - half, center_y - half
        x2, y2 = x1 + square_dim, y1 + square_dim

        pad_l, pad_t = max(0, -x1), max(0, -y1)
        pad_r, pad_b = max(0, x2 - w), max(0, y2 - h)

        padded_img = cv2.copyMakeBorder(img_rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0,0,0])
        
        cx1, cy1 = x1 + pad_l, y1 + pad_t
        cx2, cy2 = cx1 + square_dim, cy1 + square_dim

        crop_img = padded_img[cy1:cy2, cx1:cx2]
        input_img = cv2.resize(crop_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        
        # Normalization
        norm_img = input_img.astype(np.float32) / 255.0
        norm_img = (norm_img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        blob = np.transpose(norm_img, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        
        meta = {
            'padded_img': padded_img,
            'crop_coords': (cx1, cy1, cx2, cy2),
            'pad_info': (pad_t, pad_l, h, w)
        }
        
        return blob.astype(np.float32), meta

    def predict_raw(self, input_tensor):
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0][0, 0, :, :]
        probs = 1 / (1 + np.exp(-logits))
        return probs

    def visualize(self, image_path):
        try:
            input_tensor, meta = self.preprocess(image_path)
            
            # Get Probabilities
            probs = self.predict_raw(input_tensor)
            max_conf = np.max(probs)
            print(f"    Max Conf: {max_conf:.4f}")

            # Adaptive Thresholding
            thresholds = [0.5, 0.3, 0.15]
            mask = None
            final_thresh = 0.5
            
            for thresh in thresholds:
                temp_mask = (probs > thresh).astype(np.uint8)
                if np.sum(temp_mask) > 50: # Ensure at least 50 pixels are detected
                    mask = temp_mask
                    final_thresh = thresh
                    print(f"    Patch detected at threshold: {thresh}")
                    break
            
            padded_img = meta['padded_img']
            cx1, cy1, cx2, cy2 = meta['crop_coords']
            pad_t, pad_l, h, w = meta['pad_info']
            
            overlay = padded_img.copy()

            if mask is not None:
                # Resize mask to crop size
                crop_h, crop_w = cy2 - cy1, cx2 - cx1
                real_scale_mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                
                # Place in full image
                full_mask = np.zeros((padded_img.shape[0], padded_img.shape[1]), dtype=np.uint8)
                full_mask[cy1:cy2, cx1:cx2] = real_scale_mask
                
                # Draw Green Overlay
                color_mask = np.zeros_like(padded_img)
                color_mask[:, :] = [0, 255, 0] 
                
                overlay = np.where(full_mask[:, :, None] == 1, 
                                cv2.addWeighted(padded_img, 0.7, color_mask, 0.3, 0), 
                                padded_img)
                
                # Draw White Contour
                contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (255, 255, 255), 3)
            else:
                print("    No patch detected.")
                # Draw Text on image indicating failure
                cv2.putText(overlay, f"No Patch Detected (Max Conf: {max_conf:.2f})", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

            # Remove padding
            final_view = overlay[pad_t:pad_t+h, pad_l:pad_l+w]

            plt.figure(figsize=(12, 12))
            plt.imshow(final_view)
            plt.axis('off')
            plt.title(f"File: {image_path.name} | Conf: {max_conf:.2f}", fontsize=14)
            plt.show()
            
        except Exception as e:
            print(f"    Skipping {image_path.name}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Device Segmentation Inference")
    # Defaults set to your Original Paths
    parser.add_argument('--onnx_path', type=str, default="/kaggle/working/prod_pipeline_v1/device_segmentation.onnx", help="Path to ONNX model")
    parser.add_argument('--test_dir', type=str, default="/kaggle/input/sample3", help="Path to test images directory")
    
    args = parser.parse_args()
    
    ONNX_PATH = Path(args.onnx_path)
    RAW_DIR = Path(args.test_dir)
    
    if ONNX_PATH.exists():
        predictor = PatchPredictor(ONNX_PATH)
        
        # Gather all images
        all_images = sorted(list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")) + list(RAW_DIR.glob("*.jpeg")))
        
        if all_images:
            print(f"Found {len(all_images)} images in directory. Processing all...")
            for idx, test_img in enumerate(all_images):
                print(f"\n[{idx+1}/{len(all_images)}] Processing: {test_img.name}")
                predictor.visualize(test_img)
        else:
            print("No images found to test.")
    else:
        print("Please run training first or check --onnx_path.")