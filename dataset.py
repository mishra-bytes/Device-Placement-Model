import os
import json
import cv2
import numpy as np
import torch
import gc
import urllib.parse
import logging
from pathlib import Path
from tqdm.auto import tqdm
from ultralytics import YOLO
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

logger = logging.getLogger("TRAINER")

class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def process_data(self):
        existing_files = list(self.cfg.IMG_OUT.iterdir())
        if len(existing_files) > 5:
            logger.info(f"Data found ({len(existing_files)} images). Skipping generation.")
        else:
            logger.info(f"Starting Smart Crop Pipeline...")
            self._generate_images()
        return sorted([f.name for f in self.cfg.IMG_OUT.iterdir()])

    def _generate_images(self):
        logger.info("Loading YOLO for Smart Cropping...")
        yolo = YOLO('yolo11n-seg.pt')
        
        with open(self.cfg.RAW_JSON, 'r') as f: data = json.load(f)
        valid_items = [x for x in data if x.get('image')]
        
        success_count = 0
        for item in tqdm(valid_items, desc="Cropping"):
            try:
                url = item['image']
                if "?d=" in url: url = url.split("?d=")[1]
                clean_url = urllib.parse.unquote(url).replace("\\", "/")
                fname = Path(clean_url).name
                img_path = self.cfg.RAW_IMG_DIR / fname

                if not img_path.exists(): continue

                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None: continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]

                # Safety Resize
                infer_img = img_rgb.copy()
                if max(h, w) > 1500:
                    scale = 1500 / max(h, w)
                    infer_img = cv2.resize(img_rgb, (0,0), fx=scale, fy=scale)

                # --- Memory Safe Inference ---
                with torch.no_grad():
                    results = yolo(infer_img, verbose=False, retina_masks=False)
                
                person_mask = np.zeros((h, w), dtype=np.uint8)
                if results[0].masks:
                    boxes = results[0].boxes
                    persons = boxes.cls == 0
                    if persons.any():
                        idx = torch.argmax(boxes.xywh[persons, 2] * boxes.xywh[persons, 3])
                        real_idx = persons.nonzero(as_tuple=True)[0][idx]
                        m = results[0].masks.data[real_idx].cpu().numpy()
                        m = cv2.resize(m, (w, h))
                        person_mask = (m > 0.5).astype(np.uint8)
                
                # Free GPU memory immediately
                del results, infer_img
                # -----------------------------

                gt_mask_full = np.zeros((h, w), dtype=np.uint8)
                device_points = []
                if 'label' in item and item['label']:
                    lbl = item['label'][0]
                    pts = np.array(lbl['points'])
                    pts[:, 0] *= (lbl['original_width'] / 100.0)
                    pts[:, 1] *= (lbl['original_height'] / 100.0)
                    device_points = pts
                    cv2.fillPoly(gt_mask_full, [pts.astype(np.int32)], 1)

                if np.sum(person_mask) > 0:
                    rows = np.any(person_mask, axis=1)
                    y_head_top = np.argmax(rows)
                    cols = np.sum(person_mask, axis=0)
                    center_x = int(np.dot(np.arange(w), cols) / (np.sum(cols)+1e-6))
                else:
                    y_head_top = 0
                    center_x = w // 2
                
                y_device_bottom = int(np.max(device_points[:, 1])) if len(device_points) > 0 else h
                roi_height = y_device_bottom - y_head_top
                if roi_height < 50: roi_height = h // 3
                
                square_dim = max(int(roi_height * 1.5), 256)
                center_y = y_head_top + (roi_height // 2)

                half = square_dim // 2
                x1, y1 = center_x - half, center_y - half
                x2, y2 = x1 + square_dim, y1 + square_dim

                pad_l, pad_t = max(0, -x1), max(0, -y1)
                pad_r, pad_b = max(0, x2 - w), max(0, y2 - h)

                padded_img = cv2.copyMakeBorder(img_rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0,0,0])
                padded_mask = cv2.copyMakeBorder(gt_mask_full, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)

                cx1, cy1 = x1 + pad_l, y1 + pad_t
                cx2, cy2 = cx1 + square_dim, cy1 + square_dim

                fin_img = padded_img[cy1:cy2, cx1:cx2]
                fin_mask = padded_mask[cy1:cy2, cx1:cx2]

                fin_img = cv2.resize(fin_img, (self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE), interpolation=cv2.INTER_AREA)
                fin_mask = cv2.resize(fin_mask, (self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE), interpolation=cv2.INTER_NEAREST)

                out_name = Path(fname).stem + ".png"
                cv2.imwrite(str(self.cfg.IMG_OUT / out_name), cv2.cvtColor(fin_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(self.cfg.MASK_OUT / out_name), fin_mask * 255)
                success_count += 1
                
                # Prevent Kernel Stopping
                if success_count % 10 == 0: 
                    gc.collect()
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"Skipped image due to error: {e}")
                continue
        
        del yolo
        gc.collect()
        torch.cuda.empty_cache()

class SegmentationDataset(Dataset):
    def __init__(self, file_list, cfg, transform=None):
        self.files = file_list
        self.cfg = cfg
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = str(self.cfg.IMG_OUT / fname)
        mask_path = str(self.cfg.MASK_OUT / fname)

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None: raise ValueError(f"Failed to load image: {img_path}")
        if mask is None: raise ValueError(f"Failed to load mask: {mask_path}")
        if image.shape[:2] != mask.shape[:2]: raise ValueError(f"Shape mismatch: {fname}")

        if self.transform:
            res = self.transform(image=image, mask=mask)
            image, mask = res['image'], res['mask']
        
        return image, mask.unsqueeze(0).float() / 255.0

def get_transforms(phase):
    base = [albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    if phase == 'train':
        return albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.15, rotate_limit=15, shift_limit=0.1, border_mode=0, p=0.7),
            albu.RandomBrightnessContrast(p=0.4),
            albu.HueSaturationValue(p=0.3),
            albu.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        ] + base)
    return albu.Compose(base)