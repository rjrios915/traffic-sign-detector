import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import models
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(PROJECT_ROOT) == "src":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LegacyScratchDetector(nn.Module):
    def __init__(self, S, backbone_out=128, out_channels=5):
        super().__init__()
        self.S = S

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        blocks = [block(3, 32), block(32, 64), block(64, 128)]
        if backbone_out == 256:
            blocks.extend([block(128, 256), block(256, 256)])

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Conv2d(backbone_out, out_channels, 1)

    def forward(self, x):
        y = self.head(self.backbone(x))
        return y.permute(0, 2, 3, 1).contiguous()

class ScratchCNNDetector(nn.Module):
    def __init__(self, num_classes=0):
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
            nn.Dropout2d(0.2),
            block(128, 256),
            nn.Dropout2d(0.3),
            block(256, 256),
        )
        self.head = nn.Sequential(
            nn.Dropout2d(0.3),
            nn.Conv2d(256, 5 + num_classes, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).permute(0, 2, 3, 1).contiguous()

class ResnetDetector(nn.Module):
    def __init__(self, S=8, backbone="resnet18", num_classes=0):
        super().__init__()
        self.S = S

        if backbone == "resnet18":
            base = models.resnet18(weights=None)
            channels = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights=None)
            channels = 2048
        else:
            raise ValueError(f"unsupported backbone: {backbone}")

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
        self.head = nn.Sequential(
            nn.Conv2d(channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 5 + num_classes, 1),
        )

    def forward(self, x):
        y = self.backbone(x)
        y = self.head(y)
        return y.permute(0, 2, 3, 1).contiguous()

def build_rcnn_model(model_name, num_classes, config):
    common_kwargs = {
        "weights": None,
        "weights_backbone": None,
        "min_size": config.get("image_min_size", 640),
        "max_size": config.get("image_max_size", 1024),
        "rpn_pre_nms_top_n_test": config.get("rpn_pre_nms_top_n_test", 1000),
        "rpn_post_nms_top_n_test": config.get("rpn_post_nms_top_n_test", 300),
        "box_detections_per_img": config.get("box_detections_per_img", 100),
        "box_score_thresh": config.get("box_score_thresh", 0.01),
    }
    if model_name == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(
            trainable_backbone_layers=5 if config.get("train_backbone", True) else 0,
            **common_kwargs,
        )
    elif model_name == "fasterrcnn_mobilenet_v3_large_320_fpn":
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            trainable_backbone_layers=6 if config.get("train_backbone", True) else 0,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported RCNN model: {model_name}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def nms(boxes, scores, iou_th):
    if not boxes:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx = order[0]
        keep.append(idx)
        xx1 = np.maximum(x1[idx], x1[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        overlap = inter / (areas[idx] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(overlap <= iou_th)[0] + 1]
    return keep

def load_model(weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("model_state", checkpoint))
    config = checkpoint.get("config", {})
    S = checkpoint.get("S", 8)
    img_size = checkpoint.get("img_size", 256)

    if any(key.startswith("roi_heads.") or key.startswith("rpn.") for key in state_dict):
        model_name = config.get("model_name", "fasterrcnn_resnet50_fpn")
        num_classes = checkpoint.get("num_classes", 44)
        model = build_rcnn_model(model_name, num_classes, config).to(DEVICE)
        model_type = config.get("model_type", "R_CNN")
    elif "head.3.weight" in state_dict and "backbone.7.0.conv1.weight" in state_dict:
        backbone = config.get("backbone", "resnet18")
        out_channels = state_dict["head.3.weight"].shape[0]
        num_classes = max(0, out_channels - 5)
        model = ResnetDetector(S=S, backbone=backbone, num_classes=num_classes).to(DEVICE)
        model_type = config.get("model_type", "RESNET_CNN")
    elif "head.1.weight" in state_dict:
        out_channels = state_dict["head.1.weight"].shape[0]
        num_classes = max(0, out_channels - 5)
        model = ScratchCNNDetector(num_classes=num_classes).to(DEVICE)
        model_type = config.get("model_type", "SCRATCH_CNN")
    elif "head.weight" in state_dict:
        out_channels = state_dict["head.weight"].shape[0]
        backbone_out = state_dict["head.weight"].shape[1]
        model = LegacyScratchDetector(S=S, backbone_out=backbone_out, out_channels=out_channels).to(DEVICE)
        model_type = config.get("model_type", "SCRATCH_CNN")
    elif "head.0.weight" in state_dict:
        backbone = config.get("backbone", "resnet18")
        out_channels = state_dict["head.3.weight"].shape[0] if "head.3.weight" in state_dict else 5
        num_classes = max(0, out_channels - 5)
        model = ResnetDetector(S=S, backbone=backbone, num_classes=num_classes).to(DEVICE)
        model_type = config.get("model_type", "RESNET_CNN")
    else:
        sample_keys = list(state_dict.keys())[:10]
        raise RuntimeError(f"Unsupported checkpoint format for {weights_path}. Sample keys: {sample_keys}")

    model.load_state_dict(state_dict)
    model.eval()
    return model, S, img_size, model_type, config

def predict_grid_boxes(model, S, img_size, image_path, conf_th, nms_iou):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    height, width = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (img_size, img_size))
    x = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(DEVICE)

    with torch.no_grad():
        pred = model(x)[0].detach().cpu().numpy()

    detections = []
    for row in range(S):
        for col in range(S):
            obj = 1.0 / (1.0 + np.exp(-pred[row, col, 0]))
            if obj < conf_th:
                continue
            x_off = 1.0 / (1.0 + np.exp(-pred[row, col, 1]))
            y_off = 1.0 / (1.0 + np.exp(-pred[row, col, 2]))
            w_n = 1.0 / (1.0 + np.exp(-pred[row, col, 3]))
            h_n = 1.0 / (1.0 + np.exp(-pred[row, col, 4]))
            xc = (col + x_off) / S
            yc = (row + y_off) / S
            box = ((xc - w_n / 2) * width, (yc - h_n / 2) * height, (xc + w_n / 2) * width, (yc + h_n / 2) * height)
            label = None
            if pred.shape[-1] > 5:
                label = int(np.argmax(pred[row, col, 5:]))
            detections.append({"box": box, "score": float(obj), "label": label})

    keep = nms([item["box"] for item in detections], [item["score"] for item in detections], nms_iou)
    return [detections[idx] for idx in keep]

def predict_rcnn_boxes(model, image_path, conf_th):
    with Image.open(image_path) as img:
        x = F.to_tensor(img.convert("RGB")).to(DEVICE)
    with torch.no_grad():
        pred = model([x])[0]
    boxes = pred["boxes"].detach().cpu().numpy().tolist()
    scores = pred["scores"].detach().cpu().numpy().tolist()
    labels = pred["labels"].detach().cpu().numpy().tolist()
    filtered = []
    for box, score, label in zip(boxes, scores, labels):
        if score >= conf_th:
            filtered.append({"box": tuple(box), "score": float(score), "label": int(label)})
    return filtered

def predict_image_boxes(model, model_type, S, img_size, image_path, conf_th=0.5, nms_iou=0.4):
    if model_type == "R_CNN":
        return predict_rcnn_boxes(model, image_path, conf_th)
    return predict_grid_boxes(model, S, img_size, image_path, conf_th, nms_iou)

def list_images(img_dir):
    return sorted(
        os.path.join(img_dir, name)
        for name in os.listdir(img_dir)
        if name.lower().endswith(VALID_EXTS)
    )

def predict_directory(weights_path, img_dir, out_path, conf_th=0.5, nms_iou=0.4, progress_desc="Predicting images", limit=None):
    model, S, img_size, model_type, config = load_model(weights_path)
    image_paths = list_images(img_dir)
    if limit is not None:
        image_paths = image_paths[:limit]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for image_path in tqdm(image_paths, desc=progress_desc):
            detections = predict_image_boxes(model, model_type, S, img_size, image_path, conf_th=conf_th, nms_iou=nms_iou)
            name = os.path.basename(image_path)
            for item in detections:
                x1, y1, x2, y2 = item["box"]
                if item["label"] is None:
                    f.write(f"{name};{x1:.2f};{y1:.2f};{x2:.2f};{y2:.2f}\n")
                else:
                    f.write(f"{name};{x1:.2f};{y1:.2f};{x2:.2f};{y2:.2f};{item['label']}\n")
                written += 1

    return {
        "device": DEVICE,
        "weights": weights_path,
        "model_type": model_type,
        "config": config,
        "images_processed": len(image_paths),
        "boxes_written": written,
        "out_path": out_path,
    }
