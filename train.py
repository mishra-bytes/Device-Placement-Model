import sys
import gc
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchmetrics
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Local Imports
from config import Config
from utils import setup_logger, seed_everything, MetricTracker
from dataset import Preprocessor, SegmentationDataset, get_transforms
from model import build_model, ComboLoss

CFG = Config()
LOGGER = setup_logger(CFG)
seed_everything(CFG.SEED)

def run_unit_tests(cfg):
    LOGGER.info("Running Pre-Flight Unit Tests...")
    try:
        model = build_model(cfg, weights=None)
        dummy_in = torch.randn(2, 3, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
        dummy_out = model(dummy_in)
        assert dummy_out.shape == (2, 1, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
        LOGGER.info("    [PASS] Model Architecture Output Shape")
    except Exception as e:
        LOGGER.error(f"    [FAIL] Model Architecture: {e}")
        raise e

    try:
        aug = get_transforms('train')
        dummy_img = np.random.randint(0, 255, (cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3), dtype=np.uint8)
        dummy_mask = np.random.randint(0, 1, (cfg.INPUT_SIZE, cfg.INPUT_SIZE), dtype=np.uint8)
        res = aug(image=dummy_img, mask=dummy_mask)
        assert res['image'].shape == (3, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
        LOGGER.info("    [PASS] Augmentation Pipeline")
    except Exception as e:
        LOGGER.error(f"    [FAIL] Augmentations: {e}")
        raise e
        
    LOGGER.info("All Unit Tests Passed. Proceeding to Pipeline.")

def train_fold(fold_idx, train_files, val_files, cfg, tracker):
    LOGGER.info(f"STARTING FOLD {fold_idx+1}/{cfg.FOLDS} (Train: {len(train_files)}, Val: {len(val_files)})")

    train_ds = SegmentationDataset(train_files, cfg, get_transforms('train'))
    val_ds = SegmentationDataset(val_files, cfg, get_transforms('val'))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(cfg, weights="imagenet").to(cfg.DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    criterion = ComboLoss()
    metric = torchmetrics.JaccardIndex(task="binary").to(cfg.DEVICE)
    scaler = GradScaler()
    
    best_iou = 0.0
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        model.train()
        t_loss = 0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(cfg.DEVICE), masks.to(cfg.DEVICE)
            with autocast():
                preds = model(imgs)
                loss = criterion(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            t_loss += loss.item()

        model.eval()
        metric.reset()
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(cfg.DEVICE), masks.to(cfg.DEVICE)
                with autocast():
                    preds = model(imgs)
                metric.update(preds.sigmoid() > 0.5, masks.int())
        
        val_iou = metric.compute().item()
        scheduler.step(epoch + val_iou)
        
        t_loss_avg = t_loss/len(train_loader)
        tracker.log(epoch, t_loss_avg, val_iou, fold_idx)

        if (epoch+1) % 10 == 0:
            LOGGER.info(f"    Ep {epoch+1}: Loss {t_loss_avg:.4f} | IoU {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), cfg.MODEL_DIR / f"model_fold_{fold_idx}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                LOGGER.info(f"    Early stopping at epoch {epoch+1}")
                break
    
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return best_iou

def export_to_onnx(cfg):
    LOGGER.info("Exporting Best Model to ONNX...")
    model = build_model(cfg, weights=None).to("cpu")
    weight_path = cfg.MODEL_DIR / "model_fold_0.pth"
    
    if not weight_path.exists():
        LOGGER.error("Model weights not found for export.")
        return
    
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    
    dummy_input = torch.randn(1, 3, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
    out_path = cfg.WORK_DIR / "device_segmentation.onnx"
    
    try:
        torch.onnx.export(
            model, dummy_input, out_path,
            input_names=["input"], output_names=["output"],
            opset_version=11,
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        LOGGER.info(f"Export successful: {out_path}")
    except Exception as e:
        LOGGER.error(f"Export failed: {e}")

def predict_with_tta(model, image_tensor):
    with torch.no_grad():
        p1 = model(image_tensor).sigmoid()
        p2 = torch.flip(model(torch.flip(image_tensor, [3])).sigmoid(), [3])
    return (p1 + p2) / 2.0

if __name__ == "__main__":
    try:
        run_unit_tests(CFG)
        
        prep = Preprocessor(CFG)
        all_files = np.array(prep.process_data())
        
        tracker = MetricTracker(CFG.LOG_DIR)
        
        kf = KFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
        fold_scores = []
        
        LOGGER.info(f"Starting Training: {CFG.FOLDS} Folds | {CFG.ARCH} | {CFG.ENCODER}")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_files)):
            score = train_fold(fold, all_files[train_idx], all_files[val_idx], CFG, tracker)
            fold_scores.append(score)
            LOGGER.info(f"Fold {fold+1} Result: {score:.4f}")
            
        tracker.save()
        LOGGER.info(f"AVERAGE IoU: {np.mean(fold_scores):.4f}")
        LOGGER.info(f"Metrics saved to {CFG.LOG_DIR}/metrics.json")
        
        export_to_onnx(CFG)
        
        LOGGER.info("Generating Visual Report...")
        model = build_model(CFG, weights=None).to(CFG.DEVICE)
        model.load_state_dict(torch.load(CFG.MODEL_DIR / "model_fold_0.pth"))
        model.eval()
        
        _, val_idx = next(kf.split(all_files))
        val_ds = SegmentationDataset(all_files[val_idx], CFG, get_transforms('val'))
        val_loader = DataLoader(val_ds, batch_size=3, shuffle=True)
        
        imgs, masks = next(iter(val_loader))
        imgs = imgs.to(CFG.DEVICE)
        preds = predict_with_tta(model, imgs)
        
        imgs = imgs.cpu().numpy()
        masks = masks.cpu().numpy()
        preds = preds.cpu().numpy()
        
        mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        
        fig, axes = plt.subplots(len(imgs), 3, figsize=(10, 3*len(imgs)))
        if len(imgs) == 1: axes = np.array([axes])
        
        for i in range(len(imgs)):
            viz_img = np.clip((imgs[i] * std + mean).transpose(1,2,0), 0, 1)
            axes[i,0].imshow(viz_img); axes[i,0].set_title("Input")
            axes[i,1].imshow(masks[i].squeeze(), cmap='gray'); axes[i,1].set_title("Truth")
            axes[i,2].imshow(preds[i].squeeze() > 0.5, cmap='jet'); axes[i,2].set_title("Pred (TTA)")
            for ax in axes[i]: ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        LOGGER.info("Pipeline Completed Successfully.")

    except Exception as e:
        LOGGER.critical(f"Critical Pipeline Failure: {e}")
        raise e