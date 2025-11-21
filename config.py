import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import torch

@dataclass
class Config:
    # Data Paths (Production: Env Vars | Development: Your Hardcoded Defaults)
    RAW_JSON: Path = Path(os.getenv("RAW_JSON_PATH", "/kaggle/input/patch-placement/Work/project-4-at-2025-11-19-03-05-8b131c6a.json"))
    RAW_IMG_DIR: Path = Path(os.getenv("RAW_IMG_DIR", "/kaggle/input/patch-placement/Work/InnerGize/Datasets/Device_Placement"))
    WORK_DIR: Path = Path(os.getenv("WORK_DIR", "/kaggle/working/prod_pipeline_v1"))
    
    # Model Params
    ARCH: str = 'UnetPlusPlus'
    ENCODER: str = 'resnet34'
    INPUT_SIZE: int = 512
    
    # Training Hyperparams
    FOLDS: int = 5
    BATCH_SIZE: int = 12
    LR: float = 1e-4
    EPOCHS: int = 100
    PATIENCE: int = 15
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED: int = 42
    
    # MLOps
    EXP_NAME: str = field(default_factory=lambda: os.getenv("EXP_NAME", f"exp_{datetime.now().strftime('%Y%m%d_%H%M')}"))
    
    # Derived Output Paths
    IMG_OUT: Path = field(init=False)
    MASK_OUT: Path = field(init=False)
    MODEL_DIR: Path = field(init=False)
    LOG_DIR: Path = field(init=False)

    def __post_init__(self):
        self.IMG_OUT = self.WORK_DIR / "images"
        self.MASK_OUT = self.WORK_DIR / "masks"
        self.MODEL_DIR = self.WORK_DIR / "models"
        self.LOG_DIR = self.WORK_DIR / "logs" / self.EXP_NAME
        
        os.makedirs(self.IMG_OUT, exist_ok=True)
        os.makedirs(self.MASK_OUT, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)