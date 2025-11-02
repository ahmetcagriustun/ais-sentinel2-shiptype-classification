"""
Unified ResNet training script (ResNet-18/34/50) with:
- S3 download that skips files already present locally (size check)
- Configurable K-fold cross-validation (stratified) and fold index logging (JSON)
- User-selectable subset of classes + on-the-fly class merging rules
- Resize → Normalize order (fix): images are resized first, then percentile-normalized
- Optional class balancing via class-weights or weighted sampler
- Rich outputs (opt-in via flags):
  * Per-fold predictions CSV + overall OOF CSV
  * Confusion matrix PNGs, ROC/PR curves, training curves
  * Additional metrics: Balanced Acc., Kappa, MCC, Log Loss, Brier Score
  * Calibration curve + Expected Calibration Error (ECE)
  * Fold class distributions (CSV), list of unreadable files (JSON)
  * Environment info (JSON) and inference benchmark (JSON)
  * Optional ONNX / TorchScript export for deployment

Usage examples
--------------
# Merge Sailing+Pleasure → Leisure, train 5-fold with class-weights and save plots & preds
python train_resnet_model.py \
  --arch resnet50 \
  --local-data-dir ./dataset \
  --classes Cargo,Fishing,Passenger,Sailing,Pleasure,Tanker \
  --merge "Sailing+Pleasure=Leisure" \
  --kfolds 5 --epochs 30 --mixed-precision --use-class-weights \
  --save-preds --save-plots --save-calibration --benchmark --export onnx

Notes
-----
- Directory structure is expected as <local-data-dir>/<ClassName>/*.tif (or .tiff). Class names must
  be the original source folder names (before merging). The final label space after merging is logged.
- TCI or non-4-band files are automatically skipped. Only 4-band GeoTIFFs are used.
- Requires: torch, torchvision, numpy, scikit-learn, rasterio, boto3, pillow, matplotlib
"""
from __future__ import annotations
import os
import sys
import io
import time
import json
import argparse
import logging
import random
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import boto3
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, log_loss
)

import rasterio
import rasterio.errors

import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class SmallResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5, in_ch=4):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1]*(num_blocks-1)
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_resnet(arch: str, num_classes: int, in_ch: int = 4) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        return SmallResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_ch=in_ch)
    elif arch == "resnet34":
        return SmallResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_ch=in_ch)
    elif arch == "resnet50":
        return SmallResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_ch=in_ch)
    else:
        raise ValueError("Unsupported arch. Choose from: resnet18, resnet34, resnet50")

def s3_sync_prefix(bucket: str, prefix: str, local_dir: str, region: Optional[str] = None) -> Tuple[int, int]:
    session_kwargs = {}
    if region:
        session_kwargs["region_name"] = region
    s3 = boto3.client("s3", **session_kwargs)

    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    paginator = s3.get_paginator("list_objects_v2")
    downloaded = 0
    skipped = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel_path = key[len(prefix):] if key.startswith(prefix) else key
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            base = os.path.basename(local_path).lower()
            if (base.endswith(".tif") or base.endswith(".tiff")) and "tci" in base:
                continue

            s3_size = obj.get("Size", None)
            if os.path.exists(local_path) and s3_size is not None:
                try:
                    local_size = os.path.getsize(local_path)
                    if local_size == s3_size:
                        skipped += 1
                        continue
                except OSError:
                    pass

            s3.download_file(bucket, key, local_path)
            downloaded += 1

    logger.info(f"S3 sync finished. downloaded={downloaded}, skipped={skipped}")
    return downloaded, skipped

def parse_merge_rules(rules: Optional[str]) -> List[Tuple[List[str], str]]:
    """Parse merge rules like: "Sailing+Pleasure=Leisure; ClassA+ClassB=New".
    Returns a list of (sources, target).
    """
    if not rules:
        return []
    out = []
    for chunk in rules.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '=' not in chunk:
            raise ValueError(f"Invalid merge rule: {chunk}")
        left, right = chunk.split('=', 1)
        sources = [s.strip() for s in left.split('+') if s.strip()]
        target = right.strip()
        if not sources or not target:
            raise ValueError(f"Invalid merge rule: {chunk}")
        out.append((sources, target))
    return out


def build_class_mapping(source_classes: List[str], merges: List[Tuple[List[str], str]]):
    src_to_tgt = {c: c for c in source_classes}
    targets = []
    merged_sources = set()
    for sources, target in merges:
        targets.append(target)
        for s in sources:
            src_to_tgt[s] = target
            merged_sources.add(s)
    final_classes = [c for c in source_classes if c not in merged_sources]
    for t in targets:
        if t not in final_classes:
            final_classes.append(t)
    return final_classes, src_to_tgt

_UNREADABLE_FILES: List[Dict[str,str]] = []

@dataclass
class Sample:
    path: str
    label_idx: int
    target_class: str


def find_tiff_samples(root: str, source_classes: List[str], final_classes: List[str], src_to_tgt: Dict[str,str]) -> List[Sample]:
    samples: List[Sample] = []
    tgt_to_idx = {c: i for i, c in enumerate(final_classes)}

    for src in source_classes:
        cls_dir = os.path.join(root, src)
        if not os.path.isdir(cls_dir):
            logger.warning(f"Class folder missing: {cls_dir} (skipping)")
            continue
        tgt = src_to_tgt.get(src, src)
        label_idx = tgt_to_idx[tgt]
        for dirpath, _, filenames in os.walk(cls_dir):
            for fn in filenames:
                low = fn.lower()
                if (low.endswith('.tif') or low.endswith('.tiff')) and ('tci' not in low):
                    samples.append(Sample(path=os.path.join(dirpath, fn), label_idx=label_idx, target_class=tgt))

    if not samples:
        logger.error("No valid GeoTIFF samples found. Ensure directory structure and file extensions.")
    else:
        logger.info(f"Found {len(samples)} samples across {len(final_classes)} final classes (from {len(source_classes)} sources).")
    return samples


def percentile_normalize(img: np.ndarray, p1: float = 2, p2: float = 98) -> np.ndarray:
    """Per-image, per-channel percentile normalization to [0,1]. Apply AFTER resizing."""
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        band = img[c]
        lo, hi = np.percentile(band, [p1, p2])
        if hi <= lo:
            out[c] = np.clip((band - lo), 0, None)
            mx = np.max(out[c])
            if mx > 0:
                out[c] /= (mx + 1e-6)
        else:
            out[c] = (band - lo) / (hi - lo + 1e-6)
            out[c] = np.clip(out[c], 0.0, 1.0)
    return out


class GeoTiffDataset(Dataset):
    def __init__(self, samples: List[Sample], image_size: int = 128, augment: bool = True):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _read_geotiff_4band(self, path: str) -> Optional[np.ndarray]:
        try:
            with rasterio.open(path) as ds:
                if ds.count < 4:
                    raise ValueError("Less than 4 bands")
                arr = ds.read(indexes=[1, 2, 3, 4]).astype(np.float32)  # (C,H,W)
                return arr
        except Exception as e:
            _UNREADABLE_FILES.append({"path": path, "error": str(e)})
            logger.warning(f"Failed to read {path}: {e}")
            return None

    def _resize(self, img: np.ndarray, size: int) -> np.ndarray:
        c, h, w = img.shape
        out = np.zeros((c, size, size), dtype=np.float32)
        for i in range(c):
            # ensure non-negative for PIL
            band = Image.fromarray(np.clip(img[i], 0, None))
            band = band.resize((size, size), resample=Image.BILINEAR)
            out[i] = np.asarray(band).astype(np.float32)
        return out

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
        if random.random() < 0.5:
            img = torch.flip(img, dims=[1])
        k = random.randint(0, 3)
        img = torch.rot90(img, k, dims=[1, 2])
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        arr = self._read_geotiff_4band(sample.path)
        if arr is None:
            arr = np.zeros((4, self.image_size, self.image_size), dtype=np.float32)
        else:
            arr = self._resize(arr, self.image_size)
            arr = percentile_normalize(arr, 2, 98)
        tensor = torch.from_numpy(arr)
        if self.augment:
            tensor = self._augment(tensor)
        return tensor, sample.label_idx, sample.path

@dataclass
class TrainConfig:
    arch: str
    source_classes: List[str]
    merge_rules: Optional[str]
    local_data_dir: str
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    s3_region: Optional[str] = None
    image_size: int = 128
    epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    kfolds: int = 5
    seed: int = 42
    num_workers: int = 4
    mixed_precision: bool = False
    early_stop_patience: int = 7
    out_dir: str = "results"
    use_class_weights: bool = False
    weighted_sampler: bool = False
    save_preds: bool = False
    save_plots: bool = False
    save_calibration: bool = False
    export: Optional[str] = None  # None|"onnx"|"torchscript"
    benchmark: bool = False


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.array(labels), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (counts * num_classes)
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels: List[int], num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(np.array(labels), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    class_weights = counts.sum() / (counts * num_classes)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double), num_samples=len(labels), replacement=True)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / (exp.sum(axis=1, keepdims=True) + 1e-12)


def brier_score_multi(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    Y = np.zeros_like(probs)
    Y[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - Y)**2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> Tuple[float, Dict]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    bin_data = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        mask = (confidences > lo) & (confidences <= hi) if b>0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_data.append({"bin": [float(lo), float(hi)], "count": 0, "acc": None, "conf": None})
            continue
        acc = float(accuracies[mask].mean())
        conf = float(confidences[mask].mean())
        ece += (mask.mean()) * abs(acc - conf)
        bin_data.append({"bin": [float(lo), float(hi)], "count": int(mask.sum()), "acc": acc, "conf": conf})
    return float(ece), {"bins": bin_data}


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], path: str, normalize: bool = False):
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix (normalized)' if normalize else 'Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, format(val, fmt), horizontalalignment="center",
                     color="white" if val > thresh else "black", fontsize=8)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration_curve(bin_info: Dict, path: str):
    plt.figure(figsize=(5,5))
    xs, ys = [], []
    for b in bin_info["bins"]:
        if b["acc"] is not None and b["conf"] is not None:
            xs.append(b["conf"]); ys.append(b["acc"])
    plt.plot([0,1],[0,1], linestyle='--')
    plt.scatter(xs, ys)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration curve')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def train_one_fold(model, train_loader, val_loader, device, cfg: TrainConfig, class_weights: Optional[torch.Tensor]):
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    history = {"epoch": [], "train_loss": [], "val_loss": [], "top1": [], "top2": []}

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        n = 0
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        train_loss = running_loss / max(1, n)

        model.eval()
        val_loss = 0.0
        vn = 0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels_t = torch.tensor(labels, dtype=torch.long, device=device)
                with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                    logits = model(imgs)
                    loss = criterion(logits, labels_t)
                val_loss += loss.item() * imgs.size(0)
                vn += imgs.size(0)
                all_logits.append(logits.cpu())
                all_labels.extend(labels)
        val_loss = val_loss / max(1, vn)
        scheduler.step()

        if all_logits:
            all_logits_np = torch.cat(all_logits, dim=0).numpy()
            y_true = np.array(all_labels)
            y_pred = all_logits_np.argmax(axis=1)
            top1 = accuracy_score(y_true, y_pred)
            top2 = top_k_accuracy_score(y_true, all_logits_np, k=2)
        else:
            top1 = top2 = 0.0

        history["epoch"].append(epoch+1)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["top1"].append(float(top1))
        history["top2"].append(float(top2))

        logger.info(f"Epoch {epoch+1:02d}/{cfg.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} top1={top1:.4f} top2={top2:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                logger.info("Early stopping triggered.")
                break

    return best_state, best_val_loss, history


def save_training_curves(history: Dict[str, List[float]], out_path_prefix: str):
    os.makedirs(os.path.dirname(out_path_prefix), exist_ok=True)
    # Loss curve
    plt.figure()
    plt.plot(history['epoch'], history['train_loss'], label='train_loss')
    plt.plot(history['epoch'], history['val_loss'], label='val_loss')
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training/Validation Loss')
    plt.savefig(out_path_prefix + '_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    # Accuracy curve
    plt.figure()
    plt.plot(history['epoch'], history['top1'], label='top1')
    plt.plot(history['epoch'], history['top2'], label='top2')
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Top-1/Top-2 Accuracy')
    plt.savefig(out_path_prefix + '_acc.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_training(cfg: TrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.seed)

    # Optional: S3 sync
    if cfg.s3_bucket and cfg.s3_prefix:
        logger.info(f"Sync S3 s3://{cfg.s3_bucket}/{cfg.s3_prefix} -> {cfg.local_data_dir}")
        os.makedirs(cfg.local_data_dir, exist_ok=True)
        s3_sync_prefix(cfg.s3_bucket, cfg.s3_prefix, cfg.local_data_dir, region=cfg.s3_region)

    # Parse merges and build class mapping
    merges = parse_merge_rules(cfg.merge_rules)
    final_classes, src_to_tgt = build_class_mapping(cfg.source_classes, merges)
    cfg.final_classes = final_classes  # attach dynamically
    logger.info(f"Final classes after merges: {final_classes}")

    # Collect samples with target labels
    samples = find_tiff_samples(cfg.local_data_dir, cfg.source_classes, final_classes, src_to_tgt)
    if not samples:
        raise SystemExit(1)

    # Save unreadable list if any
    if _UNREADABLE_FILES:
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, 'unreadable_files.json'), 'w') as f:
            json.dump(_UNREADABLE_FILES, f, indent=2)

    # Build arrays for StratifiedKFold using TARGET labels
    y = np.array([s.label_idx for s in samples])
    idxs = np.arange(len(samples))

    skf = StratifiedKFold(n_splits=cfg.kfolds, shuffle=True, random_state=cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    plots_dir = os.path.join(cfg.out_dir, 'plots')
    fold_summaries = []
    fold_indices_log = []
    oof_rows = []  # accumulate OOF predictions rows
    fold_dist_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(idxs, y), start=1):
        logger.info(f"===== Fold {fold}/{cfg.kfolds} =====")
        train_samples = [samples[i] for i in tr_idx]
        val_samples   = [samples[i] for i in va_idx]

        # Fold distributions
        def dist(labels, n):
            c = np.bincount(np.array(labels), minlength=n)
            return c.tolist(), (c / max(1, c.sum())).round(4).tolist()
        train_labels = [s.label_idx for s in train_samples]
        val_labels   = [s.label_idx for s in val_samples]
        train_cnt, train_pct = dist(train_labels, len(final_classes))
        val_cnt, val_pct     = dist(val_labels, len(final_classes))
        fold_dist_rows.append({
            'fold': fold,
            **{f'train_count_{final_classes[i]}': int(train_cnt[i]) for i in range(len(final_classes))},
            **{f'train_pct_{final_classes[i]}': float(train_pct[i]) for i in range(len(final_classes))},
            **{f'val_count_{final_classes[i]}': int(val_cnt[i]) for i in range(len(final_classes))},
            **{f'val_pct_{final_classes[i]}': float(val_pct[i]) for i in range(len(final_classes))},
        })

        # Balancing
        class_weights = None
        sampler = None
        if cfg.weighted_sampler:
            sampler = make_weighted_sampler(np.array(train_labels), num_classes=len(final_classes))
            logger.info("Using WeightedRandomSampler for class balancing.")
        elif cfg.use_class_weights:
            class_weights = compute_class_weights(train_labels, num_classes=len(final_classes))
            logger.info(f"Using class weights: {class_weights.tolist()}")

        # Datasets / loaders
        train_ds = GeoTiffDataset(train_samples, image_size=cfg.image_size, augment=True)
        val_ds   = GeoTiffDataset(val_samples,   image_size=cfg.image_size, augment=False)

        if sampler is not None:
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

        model = build_resnet(cfg.arch, num_classes=len(final_classes), in_ch=4).to(device)

        best_state, best_val_loss, history = train_one_fold(model, train_loader, val_loader, device, cfg, class_weights)

        ckpt_name = f"best_model_{cfg.arch}_fold{fold}.pth"
        ckpt_path = os.path.join(cfg.out_dir, ckpt_name)
        if best_state is None:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({
            'model_state': best_state,
            'arch': cfg.arch,
            'final_classes': final_classes,
            'image_size': cfg.image_size,
        }, ckpt_path)
        logger.info(f"Saved {ckpt_path}")

        if cfg.save_plots:
            save_training_curves(history, os.path.join(plots_dir, f"fold{fold}_train"))

        model.load_state_dict(best_state)
        model.eval()
        all_logits = []
        y_true = []
        paths = []
        with torch.no_grad():
            for imgs, labels, img_paths in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                logits = model(imgs).cpu()
                all_logits.append(logits)
                y_true.extend(labels)
                paths.extend(img_paths)
        all_logits = torch.cat(all_logits, dim=0).numpy()
        y_true = np.array(y_true)
        probs = softmax_np(all_logits)
        y_pred = probs.argmax(axis=1)

        # Metrics
        top1 = accuracy_score(y_true, y_pred)
        top2 = top_k_accuracy_score(y_true, probs, k=2)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        ce = log_loss(y_true, probs, labels=list(range(len(final_classes))))
        brier = brier_score_multi(y_true, probs, len(final_classes))

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=final_classes, output_dict=True)

        if cfg.save_plots:
            plot_confusion_matrix(cm, final_classes, os.path.join(plots_dir, f"fold{fold}_cm_raw.png"), normalize=False)
            plot_confusion_matrix(cm, final_classes, os.path.join(plots_dir, f"fold{fold}_cm_norm.png"), normalize=True)

        ece_val = None; cal_bins = None
        if cfg.save_calibration:
            ece_val, cal_bins = expected_calibration_error(y_true, probs, n_bins=15)
            plot_calibration_curve(cal_bins, os.path.join(plots_dir, f"fold{fold}_calibration.png"))

        fold_summary = {
            'fold': fold,
            'val_loss': float(best_val_loss),
            'top1': float(top1),
            'top2': float(top2),
            'balanced_accuracy': float(bal_acc),
            'kappa': float(kappa),
            'mcc': float(mcc),
            'log_loss': float(ce),
            'brier_score': float(brier),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'ece': float(ece_val) if ece_val is not None else None,
        }
        fold_summaries.append(fold_summary)


        with open(os.path.join(cfg.out_dir, f"metrics_{cfg.arch}_fold{fold}.json"), 'w') as f:
            json.dump(fold_summary, f, indent=2)
        logger.info(f"Fold {fold} metrics saved.")

        if cfg.save_preds:
            import csv
            prob_cols = [f"prob_{c}" for c in final_classes]
            rows = []
            for pth, yt, yp, pr in zip(paths, y_true, y_pred, probs):
                row = {
                    'path': pth,
                    'true_idx': int(yt), 'true_name': final_classes[int(yt)],
                    'pred_idx': int(yp), 'pred_name': final_classes[int(yp)],
                    'top1_prob': float(pr[int(yp)])
                }
                for i, c in enumerate(final_classes):
                    row[prob_cols[i]] = float(pr[i])
                rows.append(row)
                oof_rows.append(row)
            csv_path = os.path.join(cfg.out_dir, f"predictions_{cfg.arch}_fold{fold}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader(); writer.writerows(rows)
            logger.info(f"Saved predictions CSV: {csv_path}")

        fold_indices_log.append({
            'fold': fold,
            'train_paths': [s.path for s in train_samples],
            'train_labels': train_labels,
            'val_paths': [s.path for s in val_samples],
            'val_labels': [s.label_idx for s in val_samples],
        })


    cv_path = os.path.join(cfg.out_dir, f"cv_summary_{cfg.arch}.json")
    with open(cv_path, 'w') as f:
        json.dump({'final_classes': final_classes, 'folds': fold_summaries}, f, indent=2)
    logger.info(f"Cross-validation summary saved to {cv_path}")

    folds_idx_path = os.path.join(cfg.out_dir, f"fold_indices_{cfg.arch}.json")
    with open(folds_idx_path, 'w') as f:
        json.dump(fold_indices_log, f, indent=2)
    logger.info(f"Fold indices saved to {folds_idx_path}")


    import csv
    if fold_dist_rows:
        dist_csv = os.path.join(cfg.out_dir, f"fold_distributions_{cfg.arch}.csv")
        with open(dist_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(fold_dist_rows[0].keys()))
            writer.writeheader(); writer.writerows(fold_dist_rows)
        logger.info(f"Fold distribution saved to {dist_csv}")


    if cfg.save_preds and oof_rows:
        oof_csv = os.path.join(cfg.out_dir, f"oof_predictions_{cfg.arch}.csv")
        with open(oof_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(oof_rows[0].keys()))
            writer.writeheader(); writer.writerows(oof_rows)
        logger.info(f"OOF predictions saved to {oof_csv}")

    env = {
        'python': sys.version,
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': str(device),
        'seed': cfg.seed,
        'args': vars(cfg),
        'git_commit': None,
    }
    try:
        env['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    with open(os.path.join(cfg.out_dir, 'env_info.json'), 'w') as f:
        json.dump(env, f, indent=2)

    if cfg.export in ("onnx", "torchscript"):
        export_dir = os.path.join(cfg.out_dir, 'export')
        os.makedirs(export_dir, exist_ok=True)
        dummy = torch.zeros(1, 4, cfg.image_size, cfg.image_size, device=device)
        model = build_resnet(cfg.arch, num_classes=len(final_classes), in_ch=4).to(device).eval()
        if cfg.export == 'onnx':
            onnx_path = os.path.join(export_dir, f"model_{cfg.arch}.onnx")
            torch.onnx.export(model, dummy, onnx_path, input_names=['input'], output_names=['logits'], opset_version=12)
            logger.info(f"Exported ONNX: {onnx_path}")
        else:
            ts = torch.jit.trace(model, dummy)
            ts_path = os.path.join(export_dir, f"model_{cfg.arch}.pt")
            ts.save(ts_path)
            logger.info(f"Exported TorchScript: {ts_path}")

    if cfg.benchmark:
        model = build_resnet(cfg.arch, num_classes=len(final_classes), in_ch=4).to(device).eval()
        dummy = torch.randn(1, 4, cfg.image_size, cfg.image_size, device=device)
        # warmup
        for _ in range(10):
            _ = model(dummy)
        iters = 100
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        for _ in range(iters):
            _ = model(dummy)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = (time.time() - t0) / iters
        bench = {"arch": cfg.arch, "image_size": cfg.image_size, "ms_per_image": dt*1000.0}
        with open(os.path.join(cfg.out_dir, f"inference_benchmark_{cfg.arch}.json"), 'w') as f:
            json.dump(bench, f, indent=2)
        logger.info(f"Inference benchmark saved: {bench}")

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train ResNet models on 4-band GeoTIFF patches with K-fold CV, class merging, and rich outputs.")
    p.add_argument('--arch', type=str, required=True, choices=['resnet18','resnet34','resnet50'], help='Backbone architecture')
    p.add_argument('--classes', type=str, required=True, help='Comma-separated SOURCE class names (folder names)')
    p.add_argument('--merge', type=str, default=None, help='Merge rules, e.g., "Sailing+Pleasure=Leisure;A+B=New"')
    p.add_argument('--local-data-dir', type=str, required=True, help='Local dataset root directory')

    # Optional S3 sync
    p.add_argument('--s3-bucket', type=str, default=None, help='S3 bucket name (optional)')
    p.add_argument('--s3-prefix', type=str, default=None, help='S3 prefix (folder) to sync (optional)')
    p.add_argument('--s3-region', type=str, default=None, help='S3 region name (optional)')

    # Training settings
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--kfolds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--mixed-precision', action='store_true')
    p.add_argument('--early-stop-patience', type=int, default=7)
    p.add_argument('--out-dir', type=str, default='results')

    # Class balancing options
    p.add_argument('--use-class-weights', action='store_true', help='Use inverse-frequency class weights in CE loss')
    p.add_argument('--weighted-sampler', action='store_true', help='Use WeightedRandomSampler on the train fold')

    # Outputs
    p.add_argument('--save-preds', action='store_true', help='Save per-fold and OOF predictions CSVs')
    p.add_argument('--save-plots', action='store_true', help='Save plots (CMs, training curves)')
    p.add_argument('--save-calibration', action='store_true', help='Save calibration curve and ECE JSON')
    p.add_argument('--export', type=str, default=None, choices=['onnx','torchscript'], help='Export trained backbone (random weights)')
    p.add_argument('--benchmark', action='store_true', help='Run a simple inference benchmark')

    args = p.parse_args()

    source_classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    if len(source_classes) < 2:
        raise SystemExit("Provide at least two source classes via --classes.")

    cfg = TrainConfig(
        arch=args.arch,
        source_classes=source_classes,
        merge_rules=args.merge,
        local_data_dir=args.local_data_dir,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_region=args.s3_region,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        kfolds=args.kfolds,
        seed=args.seed,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        early_stop_patience=args.early_stop_patience,
        out_dir=args.out_dir,
        use_class_weights=args.use_class_weights,
        weighted_sampler=args.weighted_sampler,
        save_preds=args.save_preds,
        save_plots=args.save_plots,
        save_calibration=args.save_calibration,
        export=args.export,
        benchmark=args.benchmark,
    )
    return cfg


if __name__ == '__main__':
    cfg = parse_args()
    run_training(cfg)
