import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision import models

from lib.loss import CIoU_Loss
from lib.plotting import plot_single_loss_curve
from lib.weather_dataset import WeatherDetectionDataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(PROJECT_ROOT) == "src":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

DEFAULT_ROOT = os.path.join(PROJECT_ROOT, "data", "detection_dataset_weather_like")
TRAIN_IMG_DIR = os.path.join(DEFAULT_ROOT, "train", "images")
TRAIN_GT_PATH = os.path.join(DEFAULT_ROOT, "train", "gt.txt")
VAL_IMG_DIR = os.path.join(DEFAULT_ROOT, "val", "images")
VAL_GT_PATH = os.path.join(DEFAULT_ROOT, "val", "gt.txt")
SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "RESNET_CNN.pt")
PLOT_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PREFIX = "RESNET_CNN"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def infer_num_classes(*gt_paths):
    max_class = -1
    for gt_path in gt_paths:
        if not os.path.isfile(gt_path):
            continue
        with open(gt_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split(";")
                if len(parts) < 6:
                    continue
                try:
                    max_class = max(max_class, int(parts[5]))
                except ValueError:
                    continue
    if max_class < 0:
        raise ValueError(f"could not infer any classes from {gt_paths}")
    return max_class + 1


def split_eval_indices(total_size, tune_size, test_size, seed):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(total_size, generator=generator).tolist()
    tune_size = min(tune_size, total_size)
    test_size = min(test_size, max(total_size - tune_size, 0))
    if test_size == 0 and total_size > tune_size:
        test_size = total_size - tune_size
    return order[:tune_size], order[tune_size:tune_size + test_size]


def normalize_run_name(name):
    return name if name.startswith(f"{MODEL_PREFIX}_") else f"{MODEL_PREFIX}_{name}"


def build_save_path(save_path, run_name):
    root, ext = os.path.splitext(save_path)
    base_name = os.path.basename(root)
    if base_name == run_name:
        return save_path
    if run_name.startswith(f"{base_name}_"):
        return os.path.join(os.path.dirname(save_path), f"{run_name}{ext}")
    return f"{root}_{run_name}{ext}"


@dataclass
class TrainConfig:
    name: str = "resnet18_detection_dataset"
    img_size: int = 256
    S: int = 8
    batch: int = 32
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    use_cache: bool = True
    seed: int = 42
    val_tune_size: int = 1000
    val_test_size: int = 500
    lambda_box: float = 3.0
    lambda_noobj: float = 2.0
    lambda_class: float = 1.0
    num_classes: int = 0
    backbone: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False


class ResnetCNNDetector(nn.Module):
    def __init__(self, num_classes, backbone="resnet18", pretrained=True, freeze_backbone=False):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"unsupported backbone: {backbone}")

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 5 + num_classes, 1),
        )

    def forward(self, x):
        y = self.backbone(x)
        y = self.head(y)
        return y.permute(0, 2, 3, 1).contiguous()


def make_dataset(img_dir, gt_path, cfg):
    return WeatherDetectionDataset(
        img_dir=img_dir,
        gt_path=gt_path,
        img_size=cfg.img_size,
        S=cfg.S,
        cache=cfg.use_cache,
        include_classes=True,
    )


def make_loaders(cfg):
    train_ds = make_dataset(TRAIN_IMG_DIR, TRAIN_GT_PATH, cfg)
    val_ds = make_dataset(VAL_IMG_DIR, VAL_GT_PATH, cfg)
    tune_indices, test_indices = split_eval_indices(
        len(val_ds),
        cfg.val_tune_size,
        cfg.val_test_size,
        cfg.seed,
    )
    pin_memory = DEVICE == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(Subset(val_ds, tune_indices), batch_size=cfg.batch, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(Subset(val_ds, test_indices), batch_size=cfg.batch, shuffle=False, pin_memory=pin_memory)
    split_sizes = {
        "train": len(train_ds),
        "val_tune": len(tune_indices),
        "val_test": len(test_indices),
    }
    return train_loader, val_loader, test_loader, split_sizes


def compute_loss(cfg, pred, target):
    return CIoU_Loss(
        pred,
        target,
        lambda_box=cfg.lambda_box,
        lambda_noobj=cfg.lambda_noobj,
        lambda_class=cfg.lambda_class,
    )


def run_loader(model, loader, cfg, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        loss, _ = compute_loss(cfg, model(images), targets)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def build_checkpoint(model, cfg, run_name, epoch, best_val, test_loss):
    return {
        "model": model.state_dict(),
        "S": cfg.S,
        "img_size": cfg.img_size,
        "config": {
            **cfg.__dict__,
            "model_type": MODEL_PREFIX,
            "run_name": run_name,
        },
        "loss_fn_name": CIoU_Loss.__name__,
        "num_classes": cfg.num_classes,
        "best_val": best_val,
        "test_loss": test_loss,
        "epoch": epoch,
    }


def train_model(cfg):
    set_seed(cfg.seed)
    if cfg.num_classes <= 0:
        cfg.num_classes = infer_num_classes(TRAIN_GT_PATH, VAL_GT_PATH)

    run_name = normalize_run_name(cfg.name)
    save_path = build_save_path(SAVE_PATH, run_name)
    model = ResnetCNNDetector(
        num_classes=cfg.num_classes,
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
    ).to(DEVICE)
    params = (parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_loader, val_loader, test_loader, split_sizes = make_loaders(cfg)

    print(
        f"Device: {DEVICE} | run: {run_name} | seed: {cfg.seed} | "
        f"classes={cfg.num_classes} | lambda_box={cfg.lambda_box} | lambda_noobj={cfg.lambda_noobj}"
    )
    print(
        f"Split sizes: train={split_sizes['train']} | "
        f"val_tune={split_sizes['val_tune']} | val_test={split_sizes['val_test']}"
    )

    best_val = float("inf")
    best_test = None
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(1, cfg.epochs + 1), desc=f"train {run_name}"):
        train_loss = run_loader(model, train_loader, cfg, optimizer)
        val_loss = run_loader(model, val_loader, cfg)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"{run_name} | Epoch {epoch:02d}/{cfg.epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | lr {cfg.lr:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_test = run_loader(model, test_loader, cfg)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(build_checkpoint(model, cfg, run_name, epoch, best_val, best_test), save_path)
            print("Saved:", save_path)

    result = {
        "name": run_name,
        "best_val": best_val,
        "test_loss": best_test,
        "save_path": save_path,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "plot_dir": PLOT_DIR,
        "num_classes": cfg.num_classes,
        "lambda_box": cfg.lambda_box,
        "lambda_noobj": cfg.lambda_noobj,
    }
    plot_single_loss_curve(result)
    return result


def main():
    cfg = TrainConfig(
        name="resnet18_detection_dataset",
        lr=1e-4,
        batch=32,
        epochs=30,
        use_cache=True,
        pretrained=True,
        freeze_backbone=False,
    )
    result = train_model(cfg)
    test_msg = "n/a" if result["test_loss"] is None else f"{result['test_loss']:.4f}"
    print(
        f"{result['name']}: best_val={result['best_val']:.4f} | test={test_msg} | "
        f"classes={result['num_classes']} | "
        f"lambdas=({result['lambda_box']}, {result['lambda_noobj']}) | {result['save_path']}"
    )


if __name__ == "__main__":
    main()
