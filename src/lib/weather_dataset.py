import glob
import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm")

class WeatherDetectionDataset(Dataset):
    def __init__(self, img_dir, gt_path, img_size=256, S=8, cache=True, include_classes=False):
        self.img_paths = sorted(
            path for path in glob.glob(os.path.join(img_dir, "*")) if path.lower().endswith(VALID_EXTS)
        )
        self.gt_path = gt_path
        self.img_size = img_size
        self.S = S
        self.cache = cache
        self.include_classes = include_classes
        self.annotations = self._load_annotations()
        self.img_paths = [path for path in self.img_paths if os.path.basename(path) in self.annotations]
        if not self.img_paths:
            raise ValueError(f"no labeled images found in {img_dir} using {gt_path}")

        self._cache_data = None
        if self.cache:
            self._cache_data = [
                self._load_item(img_path)
                for img_path in tqdm(self.img_paths, desc=f"cache {os.path.basename(img_dir)}")
            ]

    def __len__(self):
        return len(self.img_paths)

    def _load_annotations(self):
        if not os.path.exists(self.gt_path):
            raise FileNotFoundError(f"missing gt file: {self.gt_path}")

        annotations = {}
        with open(self.gt_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(";")
                if len(parts) < 5:
                    continue
                fname = parts[0].strip()
                try:
                    x1 = float(parts[1])
                    y1 = float(parts[2])
                    x2 = float(parts[3])
                    y2 = float(parts[4])
                except ValueError:
                    continue

                box = [x1, y1, x2, y2]
                if self.include_classes:
                    if len(parts) < 6:
                        continue
                    try:
                        box.append(int(parts[5]))
                    except ValueError:
                        continue
                annotations.setdefault(fname, []).append(tuple(box))
        return annotations

    def _load_image_tensor(self, img_path):
        with Image.open(img_path) as image:
            image = image.convert("RGB")
            orig_w, orig_h = image.size
            if image.size != (self.img_size, self.img_size):
                image = image.resize((self.img_size, self.img_size))
            return TF.to_tensor(image), orig_w, orig_h

    def _make_target_from_label(self, img_path, orig_w, orig_h):
        depth = 6 if self.include_classes else 5
        target = torch.zeros((self.S, self.S, depth), dtype=torch.float32)
        if orig_w <= 0 or orig_h <= 0:
            return target

        best_area = torch.zeros((self.S, self.S), dtype=torch.float32)
        for annotation in self.annotations.get(os.path.basename(img_path), []):
            x1, y1, x2, y2 = annotation[:4]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 0.0 or bh <= 0.0:
                continue
            xc = ((x1 + x2) * 0.5) / orig_w
            yc = ((y1 + y2) * 0.5) / orig_h
            w = bw / orig_w
            h = bh / orig_h
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
            if self.include_classes:
                target[row, col, 5] = float(annotation[4])
        return target

    def _load_item(self, img_path):
        image, orig_w, orig_h = self._load_image_tensor(img_path)
        return image, self._make_target_from_label(img_path, orig_w, orig_h)

    def __getitem__(self, idx):
        return self._cache_data[idx] if self.cache else self._load_item(self.img_paths[idx])
