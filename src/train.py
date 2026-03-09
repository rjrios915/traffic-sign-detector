import os
import glob
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from tqdm import tqdm

from loss import CIoU_Loss

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(PROJECT_ROOT) == "src":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class TrainConfig:
    name: str = "baseline"
    root: str = os.path.join(PROJECT_ROOT, "data", "detection_dataset_resized")
    img_size: int = 256
    S: int = 8
    batch: int = 32
    epochs: int = 30
    lr: float = 1e-3
    use_cache: bool = True
    weight_decay: float = 1e-4
    save_path: str = os.path.join(PROJECT_ROOT, "models", "detector_cnn.pt")
    loss_fn: callable = CIoU_Loss
    plot_dir: str = os.path.join(PROJECT_ROOT, "results")


class dataset(TorchDataset):
    def __init__(self, img_dir, label_dir, img_size=256, S=8, cache=True):
        self.img_paths = sorted(
            p for p in glob.glob(os.path.join(img_dir, "*")) if p.lower().endswith(VALID_EXTS)
        )
        self.label_dir = label_dir
        self.img_size = img_size
        self.S = S
        self.cache = cache

        self._cache_data = None
        if self.cache:
            self._cache_data = [
                (self._load_image_tensor(img_path), self._make_target_from_label(img_path))
                for img_path in tqdm(self.img_paths, desc=f"cache {os.path.basename(img_dir)}")
            ]

    def __len__(self):
        return len(self.img_paths)

    def _label_path(self, img_path):
        base = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.label_dir, base + ".txt")

    def _load_image_tensor(self, img_path):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if img.size != (self.img_size, self.img_size):
                img = img.resize((self.img_size, self.img_size))
            return TF.to_tensor(img)

    def _make_target_from_label(self, img_path):
        target = torch.zeros((self.S, self.S, 5), dtype=torch.float32)
        label_path = self._label_path(img_path)
        if not os.path.exists(label_path):
            return target

        best_area = torch.zeros((self.S, self.S), dtype=torch.float32)
        with open(label_path, "r") as f:
            for ln in f:
                parts = ln.strip().split()
                if len(parts) < 5:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                col = max(0, min(self.S - 1, int(xc * self.S)))
                row = max(0, min(self.S - 1, int(yc * self.S)))
                area = w * h
                if area <= best_area[row, col]:
                    continue
                best_area[row, col] = area
                target[row, col, 0] = 1.0
                target[row, col, 1] = xc * self.S - col
                target[row, col, 2] = yc * self.S - row
                target[row, col, 3] = w
                target[row, col, 4] = h
        return target

    def __getitem__(self, idx):
        if self.cache:
            return self._cache_data[idx]
        img_path = self.img_paths[idx]
        return self._load_image_tensor(img_path), self._make_target_from_label(img_path)

class detector(nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.backbone = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            block(256, 256),
        )
        self.head = nn.Conv2d(256, 5, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).permute(0, 2, 3, 1).contiguous()

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = DEVICE
        self.pin = self.device == "cuda"
        self.model = detector().to(self.device)
        self.opt = optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.train_loader, self.val_loader = self._make_loaders()

    def _make_ds(self, split):
        return dataset(
            img_dir=os.path.join(self.cfg.root, "images", split),
            label_dir=os.path.join(self.cfg.root, "labels", split),
            img_size=self.cfg.img_size,
            S=self.cfg.S,
            cache=self.cfg.use_cache,
        )

    def _make_loaders(self):
        train_ds, val_ds = self._make_ds("train"), self._make_ds("val")
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch,
            shuffle=True,
            pin_memory=self.pin,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch,
            shuffle=False,
            pin_memory=self.pin,
        )
        return train_loader, val_loader

    def train_epoch(self):
        self.model.train()
        running = 0.0
        for x, t in self.train_loader:
            x, t = x.to(self.device), t.to(self.device)
            pred = self.model(x)
            loss, _ = self.cfg.loss_fn(pred, t)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            running += loss.item()
        return running / len(self.train_loader)

    def eval_epoch(self):
        self.model.eval()
        running = 0.0
        with torch.no_grad():
            for x, t in self.val_loader:
                x, t = x.to(self.device), t.to(self.device)
                pred = self.model(x)
                loss, _ = self.cfg.loss_fn(pred, t)
                running += loss.item()
        return running / len(self.val_loader)

    def fit(self):
        print(f"Device: {self.device} | run: {self.cfg.name}")
        best_val = float("inf")
        train_losses = []
        val_losses = []
        best_path = self.cfg.save_path
        if best_path.endswith(".pt"):
            best_path = best_path.replace(".pt", f"_{self.cfg.name}.pt")

        for epoch in tqdm(range(1, self.cfg.epochs + 1), desc=f"train {self.cfg.name}"):
            train_loss = self.train_epoch()
            val_loss = self.eval_epoch()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"{self.cfg.name} | Epoch {epoch:02d}/{self.cfg.epochs} | train {train_loss:.4f} | val {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "S": self.cfg.S,
                        "img_size": self.cfg.img_size,
                        "config": {k: v for k, v in self.cfg.__dict__.items() if k != "loss_fn"},
                        "loss_fn_name": getattr(self.cfg.loss_fn, "__name__", str(self.cfg.loss_fn)),
                        "best_val": best_val,
                    },
                    best_path,
                )
                print("Saved:", best_path)
        return {
            "name": self.cfg.name,
            "best_val": best_val,
            "save_path": best_path,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "plot_dir": self.cfg.plot_dir,
        }


def plot_single_loss_curve(result):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping plots")
        return

    plot_dir = result.get("plot_dir", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    epochs = range(1, len(result["train_losses"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, result["train_losses"], label="train")
    plt.plot(epochs, result["val_losses"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves: {result['name']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(plot_dir, f"loss_{result['name']}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print("Saved plot:", out_path)


def plot_val_comparison(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping plots")
        return

    if not results:
        return

    plot_dir = results[0].get("plot_dir", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for r in results:
        epochs = range(1, len(r["val_losses"]) + 1)
        plt.plot(epochs, r["val_losses"], label=r["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title("Validation Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    comp_path = os.path.join(plot_dir, "val_loss_comparison.png")
    plt.tight_layout()
    plt.savefig(comp_path, dpi=140)
    plt.close()
    print("Saved plot:", comp_path)

def run_experiments(configs):
    results = []
    for cfg in configs:
        result = Trainer(cfg).fit()
        results.append(result)
        plot_single_loss_curve(result)
    plot_val_comparison(results)
    print("Results:")
    for r in sorted(results, key=lambda x: x["best_val"]):
        print(f"{r['name']}: best_val={r['best_val']:.4f} | {r['save_path']}")
    return results


def main():
    configs = [
        TrainConfig(name="lr5e2", lr=1e-3, batch=32, epochs=10, use_cache=True),
        TrainConfig(name="lr1e2", lr=1e-3, batch=32, epochs=10, use_cache=True),
        TrainConfig(name="lr5e3", lr=1e-3, batch=32, epochs=10, use_cache=True)
    ]
    run_experiments(configs)


if __name__ == "__main__":
    main()
