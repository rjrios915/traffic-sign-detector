import math
import torch
import torch.nn as nn

LAMBDA_BOX = 10.0
LAMBDA_NOOBJ = 0.5

# Distance‑IoU Loss: Faster and Better Learning for Bounding Box Regression
def CIoU_Loss(pred, target, S=8):
    obj_t = target[..., 0]
    x_t, y_t, w_t, h_t = target[..., 1], target[..., 2], target[..., 3], target[..., 4]

    obj_p = pred[..., 0]
    box_p = pred[..., 1:5]

    bce = nn.BCEWithLogitsLoss(reduction="none")

    noobj_mask = (obj_t == 0)
    obj_mask = (obj_t == 1)

    # Calculate for false negative and false positives
    obj_loss_all = bce(obj_p, obj_t)
    obj_loss = obj_loss_all[obj_mask].mean() if obj_mask.any() else obj_loss_all.mean()
    noobj_loss = (
        obj_loss_all[noobj_mask].mean()
        if noobj_mask.any()
        else torch.tensor(0.0, device=pred.device)
    )

    if obj_mask.any():
        B, Sy, Sx, _ = pred.shape
        device = pred.device

        # predicted box params
        xy_p = torch.sigmoid(box_p[..., 0:2])
        wh_p = torch.sigmoid(box_p[..., 2:4])
        rows = torch.arange(Sy, device=device).view(1, Sy, 1).expand(B, Sy, Sx)
        cols = torch.arange(Sx, device=device).view(1, 1, Sx).expand(B, Sy, Sx)

        # predicted centers in normalized image coords
        xc_p = (cols + xy_p[..., 0]) / Sx
        yc_p = (rows + xy_p[..., 1]) / Sy
        w_p = wh_p[..., 0]
        h_p = wh_p[..., 1]

        # target centers in normalized image coords
        xc_t, yc_t = (cols + x_t) / Sx, (rows + y_t) / Sy
        w_gt, h_gt = w_t, h_t

        # convert from center to corners
        px1, py1, px2, py2 = xc_p - w_p / 2, yc_p - h_p / 2, xc_p + w_p / 2, yc_p + h_p / 2
        tx1, ty1, tx2, ty2 = xc_t - w_gt / 2, yc_t - h_gt / 2, xc_t + w_gt / 2, yc_t + h_gt / 2

        # keep only cells with objects
        px1,py1, px2, py2 = px1[obj_mask], py1[obj_mask], px2[obj_mask], py2[obj_mask]
        tx1, ty1, tx2, ty2 = tx1[obj_mask], ty1[obj_mask], tx2[obj_mask], ty2[obj_mask]

        ix1, iy1, ix2, iy2 = torch.max(px1, tx1), torch.max(py1, ty1), torch.min(px2, tx2), torch.min(py2, ty2)
        inter_w, inter_h = (ix2 - ix1).clamp(min=0), (iy2 - iy1).clamp(min=0)
        inter = inter_w * inter_h

        # calculate areas
        area_p, area_t = (px2 - px1).clamp(min=1e-6) * (py2 - py1).clamp(min=1e-6), (tx2 - tx1).clamp(min=1e-6) * (ty2 - ty1).clamp(min=1e-6)
        area_t = (tx2 - tx1).clamp(min=1e-6) * (ty2 - ty1).clamp(min=1e-6)
        union = area_p + area_t - inter + 1e-7
        iou = inter / union

        # calculate center distance
        pcx, pcy, tcx, tcy = (px1 + px2) / 2, (py1 + py2) / 2, (tx1 + tx2) / 2, (ty1 + ty2) / 2
        rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

        # calculate enclosing box diagonal squared
        cx1, cy1, cx2, cy2 = torch.min(px1, tx1), torch.min(py1, ty1), torch.max(px2, tx2), torch.max(py2, ty2)
        c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + 1e-7

        # calculate aspect ratio term
        pw, ph, tw, th = (px2 - px1).clamp(min=1e-6), (py2 - py1).clamp(min=1e-6), (tx2 - tx1).clamp(min=1e-6), (ty2 - ty1).clamp(min=1e-6)
        v = (4.0 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2

        with torch.no_grad():
            alpha = v / (1.0 - iou + v + 1e-7)

        ciou = iou - (rho2 / c2) - alpha * v
        box_loss = (1.0 - ciou).mean()

    else:
        box_loss = torch.tensor(0.0, device=pred.device)

    total = (LAMBDA_BOX * box_loss) + obj_loss + (LAMBDA_NOOBJ * noobj_loss)

    return total, {
        "box": float(box_loss.item()),
        "obj": float(obj_loss.item()),
        "noobj": float(noobj_loss.item()),
    }

# YOLO Loss: You Only Look Once for Object Detection
def YOLO_Loss(pred, target):
    obj_t = target[..., 0]
    x_t, y_t, w_t, h_t = target[..., 1], target[..., 2], target[..., 3], target[..., 4]

    obj_p = pred[..., 0]
    box_p = pred[..., 1:5]

    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    obj_loss = bce(obj_p, obj_t)
    noobj_mask = (obj_t == 0)
    obj_mask = (obj_t == 1)

    obj_loss = obj_loss[obj_mask].mean() if obj_mask.any() else obj_loss.mean()
    noobj_loss = bce(obj_p[noobj_mask], obj_t[noobj_mask]).mean() if noobj_mask.any() else torch.tensor(0.0, device=pred.device)

    if obj_mask.any():
        xyp = torch.sigmoid(box_p[..., 0:2])
        whp = torch.sigmoid(box_p[..., 2:4])
        box_loss_xy = mse(xyp[obj_mask], torch.stack([x_t, y_t], dim=-1)[obj_mask]).mean()
        box_loss_wh = mse(torch.sqrt(whp[obj_mask].clamp(min=1e-6)),
                          torch.sqrt(torch.stack([w_t, h_t], dim=-1)[obj_mask].clamp(min=1e-6))).mean()
        box_loss = box_loss_xy + box_loss_wh

    else:
        box_loss = torch.tensor(0.0, device=pred.device)

    total = (LAMBDA_BOX * box_loss) + obj_loss + (LAMBDA_NOOBJ * noobj_loss)
    return total, {"box": box_loss.item(), "obj": obj_loss.item(), "noobj": float(noobj_loss.item())}
