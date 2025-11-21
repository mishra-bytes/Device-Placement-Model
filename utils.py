import os
import sys
import json
import random
import logging
import numpy as np
import torch
from datetime import datetime

def setup_logger(cfg):
    """Configures a file logger and a console logger"""
    log_file = cfg.LOG_DIR / "training.log"
    
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("TRAINER")

def seed_everything(seed=42):
    logger = logging.getLogger("SETUP")
    logger.info(f"Seeding with {seed}...")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class MetricTracker:
    """Simulates a remote experiment tracker"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.data = []

    def log(self, epoch, train_loss, val_iou, fold):
        self.data.append({
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_iou": val_iou,
            "timestamp": datetime.now().isoformat()
        })
    
    def save(self):
        with open(self.log_dir / "metrics.json", "w") as f:
            json.dump(self.data, f, indent=4)