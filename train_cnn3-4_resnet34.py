import os
import sys
import json
import glob
import yaml
import boto3
import random
import warnings
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
)
from time import perf_counter
from sklearn.model_selection import StratifiedKFold  # NO-GROUP

# Reduce shared memory usage in DataLoader workers
try:
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")
except Exception:
    pass


def format_hms(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------
# Utils: Config / S3 / Seeds
# ---------------------------

def load_config(path="config.yaml") -> dict:
    if not os.path.exists(path):
        print("ERROR: config.yaml not found.", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tr = cfg.get("training", {})
    params = {
        "epochs": tr.get("epochs", 10),
        "batch_size": tr.get("batch_size", 64),
        "learning_rate": tr.get("learning_rate", 5e-4),
        "weight_decay": tr.get("weight_decay", 1e-4),
        "num_workers": tr.get("num_workers", 4),
        "image_size": tr.get("image_size", 128),
        "seed": tr.get("seed", 42),
        "early_stopping_patience": tr.get("early_stopping_patience", 7),
        "mixed_precision": bool(tr.get("mixed_precision", True)),
        "train_val_test_split": tr.get("train_val_test_split", [0.8, 0.1, 0.1]),
    }

    bands_order = cfg.get("bands_order", ["B02", "B03", "B04", "B08"])

    s3cfg = cfg.get("s3", {})
    bucket = s3cfg.get("bucket")
    if not bucket:
        print("s3.bucket is required.", file=sys.stderr)
        sys.exit(1)

    results_prefix = s3cfg.get("results_prefix", "results/")
    cv_dataset_prefix = s3cfg.get("cv_dataset_prefix", "training-patches-ship-type-opensea/")

    def norm(p): return p if p.endswith("/") else p + "/"

    return {
        "training": params,
        "bands_order": bands_order,
        "s3": {
            "bucket": bucket,
            "results_prefix": norm(results_prefix),
            "cv_dataset_prefix": norm(cv_dataset_prefix),
        }
    }


def set_determinism(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def s3_client():
    return boto3.client("s3")


def s3_list_keys(bucket: str, prefix: str) -> List[str]:
    cli = s3_client()
    paginator = cli.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith("/"):
                continue
            if "/." in k or ".ipynb_checkpoints" in k:
                continue
            base = os.path.basename(k).lower()
            if "tci" in base:
                continue  # skip TCI
            if not (k.lower().endswith(".tif") or k.lower().endswith(".tiff")):
                continue
            keys.append(k)
    return keys


def s3_download_to_tree(bucket: str, keys: List[str], src_prefix: str, dst_root: str):
    os.makedirs(dst_root, exist_ok=True)
    cli = s3_client()
    for k in keys:
        rel = k[len(src_prefix):]
        local_path = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        cli.download_file(bucket, k, local_path)


def s3_upload_dir(bucket: str, local_dir: str, dest_prefix: str):
    cli = s3_client()
    for root, _, files in os.walk(local_dir):
        for f in files:
            local_path = os.path.join(root, f)
            rel = os.path.relpath(local_path, local_dir).replace("\\", "/")
            dest_key = f"{dest_prefix}{rel}"
            cli.upload_file(local_path, bucket, dest_key)


# ---------------------------
# Labels (MERGED)
# ---------------------------

# Final 4-class list with merges:
# - Sailing + Pleasure -> Leisure
# - Cargo + Tanker     -> CargoTanker
CLASSES = ["CargoTanker", "Fishing", "Passenger", "Leisure"]

LEISURE_ALIASES = {"Sailing", "Pleasure", "Leisure"}
CARGOTANKER_ALIASES = {"Cargo", "Tanker", "CargoTanker"}


def parse_label_from_path(p: str) -> str:
    """
    Parse label from path segments.

    Merges (backward-compatible with existing directory names):
    - Sailing / Pleasure  -> Leisure
    - Cargo / Tanker      -> CargoTanker
    """
    parts = set(p.replace("\\", "/").split("/"))

    # merged aliases
    if LEISURE_ALIASES & parts:
        return "Leisure"
    if CARGOTANKER_ALIASES & parts:
        return "CargoTanker"

    # direct hit (already merged)
    for c in CLASSES:
        if c in parts:
            return c

    # fallback: parent directory
    parent = p.replace("\\", "/").split("/")[-2]
    if parent in LEISURE_ALIASES:
        return "Leisure"
    if parent in CARGOTANKER_ALIASES:
        return "CargoTanker"
    return parent


# ---------------------------
# I/O helpers
# ---------------------------

def _to_chw(arr: np.ndarray, expected_channels: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected TIFF shape: {arr.shape}")

    axes = list(arr.shape)
    if expected_channels in axes:
        ch_axis = axes.index(expected_channels)
    elif any(s in (3, 4, 8, 12) for s in axes):
        if axes[2] in (3, 4, 8, 12):
            ch_axis = 2
        elif axes[0] in (3, 4, 8, 12):
            ch_axis = 0
        else:
            ch_axis = int(np.argmin(axes))
    else:
        ch_axis = int(np.argmin(axes))

    if ch_axis == 0:
        chw = arr
    elif ch_axis == 2:
        chw = np.transpose(arr, (2, 0, 1))
    elif ch_axis == 1:
        chw = np.transpose(arr, (1, 0, 2))
    else:
        perm = [ch_axis] + [i for i in range(3) if i != ch_axis]
        chw = np.transpose(arr, perm)
    return chw


def read_tif(path: str, expected_channels: int) -> np.ndarray:
    arr = tiff.imread(path)
    chw = _to_chw(arr, expected_channels)
    if chw.shape[0] > expected_channels and expected_channels > 0:
        chw = chw[:expected_channels, ...]
    chw = chw.astype(np.float32)
    chw = np.clip(chw, 0, None)
    return chw


def resize_to_square(arr: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2
        C, H, W = arr.shape
        out = np.zeros((C, size, size), dtype=arr.dtype)
        for c in range(C):
            out[c] = cv2.resize(arr[c], (size, size), interpolation=cv2.INTER_NEAREST)
        return out
    except Exception:
        C, H, W = arr.shape
        if H == size and W == size:
            return arr
        out = np.zeros((C, size, size), dtype=arr.dtype)
        y0 = max(0, (H - size) // 2)
        x0 = max(0, (W - size) // 2)
        y1 = min(H, y0 + size)
        x1 = min(W, x0 + size)
        crop = arr[:, y0:y1, x0:x1]
        out[:, :crop.shape[1], :crop.shape[2]] = crop
        return out


class S2Dataset(Dataset):
    """Dataset reading pre-downloaded TIFFs; applies augmentation and normalization."""
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        image_size: int,
        norm_stats: Dict[str, np.ndarray],
        expected_channels: int,
        train: bool = True
    ):
        self.samples = samples
        self.image_size = image_size
        self.train = train
        self.low = norm_stats["low"].astype(np.float32)
        self.high = norm_stats["high"].astype(np.float32)
        self.expected_channels = expected_channels

    def __len__(self):
        return len(self.samples)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            x = x[:, :, ::-1]  # H flip
        if random.random() < 0.5:
            x = x[:, ::-1, :]  # V flip
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            x = np.rot90(x, k, axes=(1, 2)).copy()
        if random.random() < 0.5:
            alpha = 1.0 + random.uniform(-0.1, 0.1)   # contrast
            beta = random.uniform(-0.03, 0.03)        # brightness
            x = np.clip(x * alpha + beta, 0.0, None)
        return x

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        c = x.shape[0]
        low = self.low[:c, None, None]
        high = self.high[:c, None, None]
        x = np.clip(x, low, high)
        denom = np.maximum(high - low, 1e-6)
        x = (x - low) / denom
        return x

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = read_tif(path, expected_channels=self.expected_channels)
        x = resize_to_square(x, self.image_size)
        if self.train:
            x = self._augment(x)
        x = self._normalize(x)
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ---------------------------
# Model: ResNet-34-ish
# ---------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class SmallResNet34(nn.Module):
    """
    ResNet-34-ish backbone adapted to N-channel input.
    Block config: [3, 4, 6, 3]
    """
    def __init__(self, in_ch=4, num_classes=4):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------------------------
# Metrics / Reporting
# ---------------------------

def plot_curves(history: Dict[str, List[float]], out_png: str):
    plt.figure()
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_confusion_matrix(y_true, y_pred, classes, out_png: str, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-6, None)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (norm)" if normalize else ""))
    plt.colorbar()
    tick = np.arange(len(classes))
    plt.xticks(tick, classes, rotation=45, ha="right")
    plt.yticks(tick, classes)

    fmt = ".2f" if normalize else "d"
    thr = cm.max() / 2.0 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thr else "black"
            )

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def classification_report_to_df_dict(y_true, y_pred, classes):
    return classification_report(
        y_true, y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )


# ---------------------------
# Normalization stats
# ---------------------------

def compute_percentile_stats(
    samples: List[Tuple[str, int]],
    image_size: int,
    per_class_limit: int = 200,
    expected_channels: int = 4,
) -> Dict[str, np.ndarray]:
    by_class = defaultdict(list)
    for p, y in samples:
        by_class[y].append(p)

    all_arrays = []
    for y, paths in by_class.items():
        random.shuffle(paths)
        for p in paths[:per_class_limit]:
            arr = read_tif(p, expected_channels=expected_channels)
            arr = resize_to_square(arr, image_size)
            all_arrays.append(arr)

    if not all_arrays:
        raise RuntimeError("No samples to compute normalization stats.")

    C = all_arrays[0].shape[0]
    lows, highs = [], []
    for c in range(C):
        vals = np.concatenate([a[c].ravel() for a in all_arrays])
        low = np.percentile(vals, 2.0)
        high = np.percentile(vals, 98.0)
        if high <= low:
            high = low + 1e-3
        lows.append(low)
        highs.append(high)

    return {
        "low": np.array(lows, dtype=np.float32),
        "high": np.array(highs, dtype=np.float32)
    }


# ---------------------------
# Training & Evaluation
# ---------------------------

def make_loaders(
    X_train, X_val, X_test,
    image_size, norm_stats,
    batch_size, num_workers,
    expected_channels, device
):
    # Safer defaults for CPU to avoid shm errors
    if device.type != "cuda":
        num_workers = 0
        pin_memory = False
        persistent_workers = False
    else:
        pin_memory = True
        persistent_workers = (num_workers > 0)

    ds_train = S2Dataset(X_train, image_size, norm_stats, expected_channels, train=True)
    ds_val = S2Dataset(X_val, image_size, norm_stats, expected_channels, train=False)
    ds_test = S2Dataset(X_test, image_size, norm_stats, expected_channels, train=False)

    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    return dl_train, dl_val, dl_test


def train_one_fold(
    fold_dir: str,
    model: nn.Module,
    device: torch.device,
    dl_train, dl_val, dl_test,
    epochs: int,
    lr: float,
    weight_decay: float,
    early_stopping_patience: int,
    mixed_precision: bool,
    class_weights: torch.Tensor
) -> Dict[str, Any]:
    os.makedirs(fold_dir, exist_ok=True)

    mp_enabled = mixed_precision and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=mp_enabled)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    patience = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        n_train = 0

        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=mp_enabled):
                logits = model(x)
                loss = criterion(logits, y)

            if mp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * x.size(0)
            n_train += x.size(0)

        train_loss = running_loss / max(1, n_train)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        n_val = 0
        correct = 0

        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=mp_enabled):
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                n_val += x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()

        val_loss = val_loss / max(1, n_val)
        val_acc = correct / max(1, n_val)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ---- Early stopping ----
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    plot_curves(history, os.path.join(fold_dir, "curves.png"))

    # ---- Load best and evaluate ----
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    y_true_val, y_pred_val = [], []
    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            y_true_val.extend(y.numpy().tolist())
            y_pred_val.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    y_true_test, y_pred_test, y_prob_test = [], [], []
    with torch.no_grad():
        for x, y in dl_test:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            y_true_test.extend(y.numpy().tolist())
            y_pred_test.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            y_prob_test.extend(probs.cpu().numpy().tolist())

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "y_true_val": y_true_val,
        "y_pred_val": y_pred_val,
        "y_true_test": y_true_test,
        "y_pred_test": y_pred_test,
        "y_prob_test": y_prob_test,
        "best_state": best_state,
    }


# ---------------------------
# Main CV pipeline (NO GROUPS)
# ---------------------------

def main():
    cfg = load_config()
    tr = cfg["training"]
    bands_order = cfg["bands_order"]
    s3cfg = cfg["s3"]

    set_determinism(tr["seed"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join("results", f"cv_{ts}")
    os.makedirs(out_root, exist_ok=True)

    # 1) Download dataset from S3 to local cache (skip if already cached)
    bucket = s3cfg["bucket"]
    src_prefix = s3cfg["cv_dataset_prefix"]

    print(f"Listing S3: s3://{bucket}/{src_prefix}")
    keys = s3_list_keys(bucket, src_prefix)
    if not keys:
        raise RuntimeError("No .tif files found under cv_dataset_prefix.")

    local_cache = "dataset"
    os.makedirs(local_cache, exist_ok=True)

    missing_keys = []
    for k in keys:
        rel = k[len(src_prefix):]
        local_path = os.path.join(local_cache, rel)
        if not os.path.exists(local_path):
            missing_keys.append(k)

    if missing_keys:
        print(f"Downloading {len(missing_keys)}/{len(keys)} missing files to ./{local_cache} ...")
        s3_download_to_tree(bucket, missing_keys, src_prefix, local_cache)
    else:
        print(f"Local cache is complete ({len(keys)} files). Skipping download.")

    # 2) Build manifest (path, label) with merged labels
    files = glob.glob(os.path.join(local_cache, "**", "*.tif"), recursive=True) + \
            glob.glob(os.path.join(local_cache, "**", "*.tiff"), recursive=True)

    samples_all = []
    for p in files:
        fname = os.path.basename(p).lower()
        if "tci" in fname:
            continue
        label_name = parse_label_from_path(p)
        if label_name not in CLASSES:
            continue
        label_idx = CLASSES.index(label_name)
        samples_all.append((p, label_idx))

    if not samples_all:
        raise RuntimeError("No valid samples collected.")

    print(f"Collected {len(samples_all)} samples across classes: {dict(Counter([s[1] for s in samples_all]))}")

    # Infer channel count from one sample
    test_arr = read_tif(samples_all[0][0], expected_channels=len(bands_order))
    in_ch = test_arr.shape[0]
    if len(bands_order) != in_ch:
        warnings.warn(
            f"bands_order length ({len(bands_order)}) != image channels ({in_ch}). "
            f"Proceeding with {in_ch} channels (will use first {in_ch})."
        )

    num_classes = len(CLASSES)  # 4

    # 3) Stratified K-Fold (NO GROUPS)
    train_frac, val_frac, test_frac = tr["train_val_test_split"]
    if abs(val_frac - test_frac) < 1e-6 and val_frac > 0:
        K = int(round(1.0 / val_frac))
    else:
        K = 5
    K = max(3, min(20, K))
    print(f"K-fold setup (no-group): K={K} (train/val/test target ~ {train_frac}/{val_frac}/{test_frac})")

    X = np.arange(len(samples_all))
    y = np.array([s[1] for s in samples_all])

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=tr["seed"])
    fold_test_indices = [test_idx for _, test_idx in skf.split(X, y)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cv_results = []
    agg_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    total_train_start = perf_counter()

    for r in range(K):
        fold_start = perf_counter()

        test_idx = fold_test_indices[r]
        val_idx = fold_test_indices[(r + 1) % K]
        train_idx = np.setdiff1d(np.arange(len(samples_all)), np.concatenate([test_idx, val_idx]))

        X_train = [(samples_all[i][0], samples_all[i][1]) for i in train_idx]
        X_val = [(samples_all[i][0], samples_all[i][1]) for i in val_idx]
        X_test = [(samples_all[i][0], samples_all[i][1]) for i in test_idx]

        # class weights from train distribution
        train_labels = [lbl for _, lbl in X_train]
        cnt = Counter(train_labels)
        total = sum(cnt.values())
        class_weights = torch.tensor(
            [total / max(1, cnt.get(c, 1)) for c in range(num_classes)],
            dtype=torch.float32
        )

        # normalization stats from train
        norm_stats = compute_percentile_stats(
            X_train,
            image_size=tr["image_size"],
            per_class_limit=200,
            expected_channels=in_ch
        )

        dl_train, dl_val, dl_test = make_loaders(
            X_train, X_val, X_test,
            tr["image_size"], norm_stats,
            tr["batch_size"], tr["num_workers"],
            expected_channels=in_ch,
            device=device
        )

        # ✅ ResNet34 here
        model = SmallResNet34(in_ch=in_ch, num_classes=num_classes).to(device)

        fold_dir = os.path.join(out_root, f"fold_{r:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        out = train_one_fold(
            fold_dir, model, device, dl_train, dl_val, dl_test,
            epochs=tr["epochs"],
            lr=tr["learning_rate"],
            weight_decay=tr["weight_decay"],
            early_stopping_patience=tr["early_stopping_patience"],
            mixed_precision=tr["mixed_precision"],
            class_weights=class_weights
        )

        fold_elapsed = perf_counter() - fold_start
        print(f"[Fold {r}] Train wall-time = {format_hms(fold_elapsed)} ({fold_elapsed:.1f}s)")

        if out["best_state"] is not None:
            torch.save(out["best_state"], os.path.join(fold_dir, "best_model.pth"))

        y_true = np.array(out["y_true_test"])
        y_pred = np.array(out["y_pred_test"])

        acc = accuracy_score(y_true, y_pred)

        # top-2 only makes sense if >=2 classes (we have 4)
        top2 = top_k_accuracy_score(
            y_true,
            np.array(out["y_prob_test"]),
            k=2,
            labels=list(range(num_classes))
        )

        rep_dict = classification_report_to_df_dict(y_true, y_pred, CLASSES)

        with open(os.path.join(fold_dir, "classification_report.json"), "w", encoding="utf-8") as f:
            json.dump(rep_dict, f, indent=2)

        with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "fold": r,
                "test_accuracy": float(acc),
                "test_top2_accuracy": float(top2),
                "best_val_loss": float(out["best_val_loss"])
            }, f, indent=2)

        save_confusion_matrix(
            y_true, y_pred, CLASSES,
            os.path.join(fold_dir, "confusion_matrix.png"),
            normalize=False
        )
        save_confusion_matrix(
            y_true, y_pred, CLASSES,
            os.path.join(fold_dir, "confusion_matrix_norm.png"),
            normalize=True
        )

        agg_confusion += confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

        cv_results.append({
            "fold": r,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "train_time_sec": round(fold_elapsed, 2),
            "train_time_hms": format_hms(fold_elapsed),
            "acc": float(acc),
            "top2": float(top2),
            "macro_f1": float(rep_dict["macro avg"]["f1-score"]),
            "weighted_f1": float(rep_dict["weighted avg"]["f1-score"]),
        })

    # ---- CV summary ----
    df_cv = pd.DataFrame(cv_results)
    df_cv.to_csv(os.path.join(out_root, "cv_summary.csv"), index=False)

    total_train_elapsed = perf_counter() - total_train_start
    print(f"[Total] Training wall-time = {format_hms(total_train_elapsed)} ({total_train_elapsed:.1f}s)")

    cv_stats = {
        "acc_mean": float(df_cv["acc"].mean()),
        "acc_std": float(df_cv["acc"].std(ddof=0)),
        "macro_f1_mean": float(df_cv["macro_f1"].mean()),
        "macro_f1_std": float(df_cv["macro_f1"].std(ddof=0)),
        "weighted_f1_mean": float(df_cv["weighted_f1"].mean()),
        "weighted_f1_std": float(df_cv["weighted_f1"].std(ddof=0)),
        "top2_mean": float(df_cv["top2"].mean()),
        "top2_std": float(df_cv["top2"].std(ddof=0)),
        "folds": len(cv_results),
        "classes": CLASSES,
        # ✅ merged info corrected
        "merge_info": "Sailing + Pleasure -> Leisure; Cargo + Tanker -> CargoTanker",
        "total_train_time_sec": float(round(total_train_elapsed, 2)),
        "total_train_time_hms": format_hms(total_train_elapsed),
    }

    with open(os.path.join(out_root, "cv_stats.json"), "w", encoding="utf-8") as f:
        json.dump(cv_stats, f, indent=2)

    # Append timing info to summary file
    try:
        with open(os.path.join(out_root, "summary.txt"), "a", encoding="utf-8") as f:
            f.write(f"[Total] Training time: {format_hms(total_train_elapsed)} ({total_train_elapsed:.1f}s)\n")
            for rec in cv_results:
                f.write(f"Fold {rec.get('fold')}: {rec.get('train_time_hms')} ({rec.get('train_time_sec')}s)\n")
    except Exception as _e:
        print("WARN: could not write summary.txt:", _e)

    # Aggregate confusion
    plt.figure(figsize=(6, 5))
    plt.imshow(agg_confusion, interpolation="nearest")
    plt.title("Aggregate Confusion Matrix")
    plt.colorbar()

    tick = np.arange(len(CLASSES))
    plt.xticks(tick, CLASSES, rotation=45, ha="right")
    plt.yticks(tick, CLASSES)

    thr = agg_confusion.max() / 2 if agg_confusion.size else 0
    for i in range(agg_confusion.shape[0]):
        for j in range(agg_confusion.shape[1]):
            plt.text(
                j, i, str(agg_confusion[i, j]),
                ha="center", va="center",
                color="white" if agg_confusion[i, j] > thr else "black"
            )

    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "aggregate_confusion.png"), dpi=160)
    plt.close()

    # Upload results to S3
    dest_prefix = f"{s3cfg['results_prefix']}cv_{ts}/"
    print(f"Uploading results to s3://{s3cfg['bucket']}/{dest_prefix} ...")
    s3_upload_dir(s3cfg["bucket"], out_root, dest_prefix)
    print("Done.")


if __name__ == "__main__":
    main()
