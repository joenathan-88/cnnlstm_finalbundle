from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

BASE_DIR = Path(__file__).resolve().parent
def pick_model_path() -> Path:
    savedmodel = BASE_DIR / "models" / "best_model_savedmodel"
    legacy_keras = BASE_DIR / "models" / "best_model.keras"
    if savedmodel.exists() and savedmodel.is_dir():
        return savedmodel
    return legacy_keras

MODEL_PATH = pick_model_path()
SCALER_PATH = BASE_DIR / "artifacts" / "minmax_scaler.pkl"

print(f"[INFO] Python     : {sys.version.split()[0]}")
print(f"[INFO] Executable : {sys.executable}")
print(f"[INFO] Model path : {MODEL_PATH}")
print(f"[INFO] Scaler path: {SCALER_PATH}")

import tensorflow as tf
print(f"[INFO] TensorFlow : {tf.__version__}")
print(f"[INFO] tf module  : {tf.__file__}")

from model_utils import REQUIRED_FEATURES, WINDOW_SIZE, get_model, get_scaler

scaler = get_scaler(SCALER_PATH)
print(f"[OK] scaler loaded: {type(scaler).__name__}")
model = get_model(MODEL_PATH)
print(f"[OK] model loaded : {type(model).__name__}")

import numpy as np
x = np.zeros((1, WINDOW_SIZE, len(REQUIRED_FEATURES)), dtype=np.float32)
y = model.predict(x, verbose=0)
print(f"[OK] dummy predict: shape={getattr(y, 'shape', None)}")
