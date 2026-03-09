import os
import torch
import torch.nn as nn
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(PROJECT_ROOT) == "src":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

WEIGHTS = os.path.join(PROJECT_ROOT, "models", "detector_cnn_lr1e2.pt")
IMG_PATHS = [os.path.join(PROJECT_ROOT, "data", "detection_dataset", "test", f"d{i}.png") for i in [7, 8, 29, 39, 45, 49]]
CONF_TH = 0.9
NMS_IOU = 0.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TinyDetector(nn.Module):
    def __init__(self, S, backbone_out=128, out_channels=5):
        super().__init__()
        self.S = S

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        blocks = [block(3, 32), block(32, 64), block(64, 128)]
        if backbone_out == 256:
            blocks.extend([block(128, 256), block(256, 256)])

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Conv2d(backbone_out, out_channels, 1)

    def forward(self, x):
        y = self.head(self.backbone(x))
        return y.permute(0,2,3,1).contiguous()

def nms(boxes, scores, iou_th):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_th)[0]
        order = order[inds+1]
    return keep

def main():
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    S = ckpt["S"]
    img_size = ckpt["img_size"]
    out_channels = ckpt["model"]["head.weight"].shape[0]
    backbone_out = ckpt["model"]["head.weight"].shape[1]

    model = TinyDetector(S=S, backbone_out=backbone_out, out_channels=out_channels).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for IMG_PATH in IMG_PATHS:
        print(f"Processing {IMG_PATH}")

        img_bgr = cv2.imread(IMG_PATH)
        if img_bgr is None:
            raise RuntimeError("Could not read image")

        H, W = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (img_size, img_size))
        x = torch.from_numpy(resized).float().permute(2,0,1) / 255.0
        x = x.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(x)[0].cpu().numpy()  # (S,S,5) or (S,S,5+C)

        boxes, scores = [], []
        for r in range(S):
            for c in range(S):
                obj_logit = pred[r,c,0]
                obj = 1 / (1 + np.exp(-obj_logit))
                if obj < CONF_TH:
                    continue

                # offsets + wh
                x_off = 1 / (1 + np.exp(-pred[r,c,1]))
                y_off = 1 / (1 + np.exp(-pred[r,c,2]))
                w_n   = 1 / (1 + np.exp(-pred[r,c,3]))
                h_n   = 1 / (1 + np.exp(-pred[r,c,4]))

                score = obj

                # convert to normalized center in image
                xc = (c + x_off) / S
                yc = (r + y_off) / S

                # box corners in original image coords
                x1 = (xc - w_n/2) * W
                y1 = (yc - h_n/2) * H
                x2 = (xc + w_n/2) * W
                y2 = (yc + h_n/2) * H

                boxes.append([x1,y1,x2,y2])
                scores.append(score)

        keep = nms(boxes, scores, NMS_IOU)

        for i in keep:
            x1,y1,x2,y2 = [int(v) for v in boxes[i]]
            cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img_bgr, f"sign {scores[i]:.2f}", (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("detections", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
