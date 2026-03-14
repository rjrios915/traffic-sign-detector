import argparse
import os
from collections import defaultdict

import numpy as np

from lib.weather_inference import predict_directory


RESULTS_DIR = os.path.join("results")


def parse_boxes(path, has_class, pred_class_offset=0):
    boxes = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split(";")]
            if not parts or len(parts) < 5:
                continue
            try:
                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])
            except ValueError:
                continue

            label = None
            if has_class:
                if len(parts) < 6:
                    continue
                try:
                    label = int(parts[5])
                except ValueError:
                    continue
            elif len(parts) >= 6:
                try:
                    label = int(parts[5]) + pred_class_offset
                except ValueError:
                    label = None

            boxes[parts[0]].append((x1, y1, x2, y2, label))
    return boxes


def list_classes_from_gt(gt_path):
    classes = set()
    with open(gt_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split(";")]
            if len(parts) < 6:
                continue
            try:
                classes.add(int(parts[5]))
            except ValueError:
                continue
    return sorted(classes)


def split_augmented_name(name):
    stem, ext = os.path.splitext(name)
    for suffix in ("darker", "foggy", "rainy", "snowy"):
        if stem.endswith(f"_{suffix}"):
            return stem[:-(len(suffix) + 1)] + ext, suffix
    return name, "original"


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def evaluate(gt_path, pred_path, iou_thresh, by_condition=False, print_class=False, pred_class_offset=0):
    gt = parse_boxes(gt_path, has_class=True)
    pred = parse_boxes(pred_path, has_class=False, pred_class_offset=pred_class_offset)

    box_metrics = {"tp": 0, "fp": 0, "fn": 0}
    matched_ious = []
    class_metrics = {"correct": 0, "wrong": 0, "total": 0, "has_pred_labels": False}
    image_confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    by_condition_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    image_fp_metrics = {"fp_images": 0, "clean_images": 0}
    image_fp_by_condition = defaultdict(lambda: {"fp_images": 0, "clean_images": 0})
    class_confusion = defaultdict(lambda: defaultdict(int))
    class_records = []

    all_names = set(gt.keys()) | set(pred.keys())
    for pred_name in all_names:
        gt_name, condition = split_augmented_name(pred_name) if by_condition else (pred_name, "overall")
        gt_boxes = gt.get(gt_name, [])
        pred_boxes = pred.get(pred_name, [])
        used_gt = [False] * len(gt_boxes)
        has_unmatched_prediction = False

        for pred_box in pred_boxes:
            if pred_box[4] is not None:
                class_metrics["has_pred_labels"] = True

            best_iou = 0.0
            best_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if used_gt[gt_idx]:
                    continue
                score = iou(pred_box[:4], gt_box[:4])
                if score > best_iou:
                    best_iou = score
                    best_idx = gt_idx

            if best_idx >= 0 and best_iou >= iou_thresh:
                used_gt[best_idx] = True
                box_metrics["tp"] += 1
                matched_ious.append(best_iou)

                pred_cls = pred_box[4]
                gt_cls = gt_boxes[best_idx][4]
                if pred_cls is not None:
                    class_metrics["total"] += 1
                    class_confusion[gt_cls][pred_cls] += 1
                    class_records.append(
                        {
                            "image": pred_name,
                            "condition": condition,
                            "gt_class": gt_cls,
                            "pred_class": pred_cls,
                            "iou": best_iou,
                        }
                    )
                    if print_class:
                        print(
                            f"{pred_name} | gt={gt_cls} | pred={pred_cls} | "
                            f"iou={best_iou:.3f} | condition={condition}"
                        )
                    if pred_cls == gt_cls:
                        class_metrics["correct"] += 1
                    else:
                        class_metrics["wrong"] += 1
            else:
                box_metrics["fp"] += 1
                has_unmatched_prediction = True

        box_metrics["fn"] += sum(1 for used in used_gt if not used)

        has_gt = bool(gt_boxes)
        matched_any = any(used_gt)
        has_pred = bool(pred_boxes)
        bucket = by_condition_counts[condition]
        if has_gt and matched_any:
            image_confusion["tp"] += 1
            bucket["tp"] += 1
        elif has_gt:
            image_confusion["fn"] += 1
            bucket["fn"] += 1
        elif has_pred:
            image_confusion["fp"] += 1
            bucket["fp"] += 1
        else:
            image_confusion["tn"] += 1
            bucket["tn"] += 1

        fp_bucket = image_fp_by_condition[condition]
        if has_unmatched_prediction:
            image_fp_metrics["fp_images"] += 1
            fp_bucket["fp_images"] += 1
        else:
            image_fp_metrics["clean_images"] += 1
            fp_bucket["clean_images"] += 1

    tp = box_metrics["tp"]
    fp = box_metrics["fp"]
    fn = box_metrics["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0

    return {
        "box_metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "mean_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        },
        "class_metrics": {
            "correct": class_metrics["correct"],
            "wrong": class_metrics["wrong"],
            "total": class_metrics["total"],
            "has_pred_labels": class_metrics["has_pred_labels"],
            "accuracy": class_metrics["correct"] / class_metrics["total"] if class_metrics["total"] else None,
            "error": class_metrics["wrong"] / class_metrics["total"] if class_metrics["total"] else None,
        },
        "confusion": image_confusion,
        "image_fp_metrics": image_fp_metrics,
        "by_condition": dict(by_condition_counts),
        "image_fp_by_condition": dict(image_fp_by_condition),
        "class_confusion": {gt_cls: dict(preds) for gt_cls, preds in class_confusion.items()},
        "class_records": class_records,
    }


def write_example_submission(gt_path, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(gt_path, "r", encoding="utf-8", errors="ignore") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = [part.strip() for part in line.strip().split(";")]
            if len(parts) < 5:
                continue
            fout.write(";".join(parts[:5]) + "\n")
            written += 1
    return written


def visualize_submission(pred_path, img_dir, out_dir, gt_path=None, limit=100):
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("visualize requires opencv-python (`cv2`) to be installed") from exc

    pred = parse_boxes(pred_path, has_class=False)
    gt = parse_boxes(gt_path, has_class=True) if gt_path else {}

    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for name in sorted(pred.keys())[:limit]:
        img_path = os.path.join(img_dir, name)
        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        for x1, y1, x2, y2, _ in pred.get(name, []):
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        gt_name, _ = split_augmented_name(name)
        for x1, y1, x2, y2, _ in gt.get(gt_name, []):
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        stem, _ = os.path.splitext(name)
        cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), image)
        written += 1

    return written


def print_report(result, by_condition=False):
    metrics = result["box_metrics"]
    class_metrics = result["class_metrics"]
    matrix = result["confusion"]
    image_fp_metrics = result["image_fp_metrics"]

    print(f"TP: {metrics['tp']}")
    print(f"FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"Accuracy (TP/(TP+FP+FN)): {metrics['accuracy']:.2f}")
    print(f"Mean IoU (matched boxes): {metrics['mean_iou']:.3f}")

    if not class_metrics["has_pred_labels"]:
        print("Classification Accuracy/Error: n/a (prediction file has no class labels)")
    elif class_metrics["total"] == 0:
        print("Classification Accuracy/Error: n/a (no detections matched GT at this IoU)")
    else:
        print(f"Classification Accuracy (matched boxes): {class_metrics['accuracy']:.2f}")
        print(f"Classification Error (matched boxes): {class_metrics['error']:.2f}")

    print("Confusion Matrix (image-level):")
    print(f"  TP: {matrix['tp']} | FP: {matrix['fp']}")
    print(f"  FN: {matrix['fn']} | TN: {matrix['tn']}")
    print(
        "Image FP (contains at least one unmatched predicted box): "
        f"{image_fp_metrics['fp_images']} | Clean images: {image_fp_metrics['clean_images']}"
    )

    if by_condition:
        print("By condition:")
        for condition in sorted(result["by_condition"]):
            item = result["by_condition"][condition]
            print(f"  {condition}: TP={item['tp']} FP={item['fp']} FN={item['fn']} TN={item['tn']}")
        print("By condition image FP:")
        for condition in sorted(result["image_fp_by_condition"]):
            item = result["image_fp_by_condition"][condition]
            print(f"  {condition}: FP_images={item['fp_images']} Clean_images={item['clean_images']}")


def save_box_confusion_matrix_plot(result, out_path, model_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping confusion matrix plot")
        return

    metrics = result["box_metrics"]
    values = [[metrics["tp"], metrics["fn"]], [metrics["fp"], 0]]
    labels = [["TP", "FN"], ["FP", "TN"]]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(values, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred Sign", "Pred No Sign"])
    ax.set_yticks([0, 1], labels=["Actual Sign", "Actual No Sign"])
    ax.set_title(f"Confusion Matrix: {model_name}")
    for row in range(2):
        for col in range(2):
            text = "TN\nn/a" if (row, col) == (1, 1) else f"{labels[row][col]}\n{values[row][col]}"
            ax.text(col, row, text, ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved confusion matrix plot: {out_path}")


def save_class_confusion_heatmap(result, out_path, model_name, class_ids):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping class confusion heatmap")
        return

    confusion_rows = result["class_confusion"]
    if not confusion_rows:
        print("No class confusion data available, skipping class confusion heatmap")
        return

    classes = [cls for cls in range(min(class_ids), max(class_ids) + 1) if cls != 0]
    values = np.zeros((len(classes), len(classes)), dtype=np.int32)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for gt_cls, row in confusion_rows.items():
        if gt_cls not in class_to_idx:
            continue
        for pred_cls, count in row.items():
            if pred_cls not in class_to_idx:
                continue
            values[class_to_idx[gt_cls], class_to_idx[pred_cls]] = count

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    size = max(8, min(18, len(classes) * 0.3 + 4))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(values, cmap="magma")
    ax.set_xticks(range(len(classes)), labels=[str(cls) for cls in classes])
    ax.set_yticks(range(len(classes)), labels=[str(cls) for cls in classes])
    ax.set_xlabel("Predicted sign class")
    ax.set_ylabel("Actual sign class")
    ax.set_title(f"Class Confusion Heatmap: {model_name}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved class confusion heatmap: {out_path}")


def save_class_log(result, out_path):
    records = result["class_records"]
    if not records:
        print("No matched classified detections available, skipping class log")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("image\tcondition\tgt_class\tpred_class\tiou\n")
        for item in records:
            handle.write(
                f"{item['image']}\t{item['condition']}\t{item['gt_class']}\t"
                f"{item['pred_class']}\t{item['iou']:.6f}\n"
            )
    print(f"Saved class log: {out_path}")


def predict_if_needed(args):
    if args.pred:
        return args.pred

    if not args.weights or not args.img_dir or not args.out:
        raise RuntimeError("Need either --pred or all of --weights, --img-dir, and --out")

    result = predict_directory(
        weights_path=args.weights,
        img_dir=args.img_dir,
        out_path=args.out,
        conf_th=args.conf,
        nms_iou=args.nms_iou,
        limit=args.limit,
    )
    print(f"Generated predictions with {result['model_type']} -> {result['out_path']}")
    return result["out_path"]


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate detector predictions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    make_ex = sub.add_parser("make-ex")
    make_ex.add_argument("--gt", required=True)
    make_ex.add_argument("--out", required=True)

    predict = sub.add_parser("predict")
    predict.add_argument("--weights", required=True)
    predict.add_argument("--img-dir", required=True)
    predict.add_argument("--out", required=True)
    predict.add_argument("--conf", type=float, default=0.5)
    predict.add_argument("--nms-iou", type=float, default=0.4)
    predict.add_argument("--limit", type=int)

    eval_cmd = sub.add_parser("eval")
    eval_cmd.add_argument("--gt", required=True)
    eval_cmd.add_argument("--pred")
    eval_cmd.add_argument("--weights")
    eval_cmd.add_argument("--img-dir")
    eval_cmd.add_argument("--out")
    eval_cmd.add_argument("--conf", type=float, default=0.5)
    eval_cmd.add_argument("--nms-iou", type=float, default=0.4)
    eval_cmd.add_argument("--iou", type=float, default=0.5)
    eval_cmd.add_argument("--by-condition", action="store_true")
    eval_cmd.add_argument("--print-class", action="store_true")
    eval_cmd.add_argument("--obo", action="store_true")
    eval_cmd.add_argument("--confusion-viz")
    eval_cmd.add_argument("--class-heatmap-viz")
    eval_cmd.add_argument("--class-log")
    eval_cmd.add_argument("--limit", type=int)

    visualize = sub.add_parser("visualize")
    visualize.add_argument("--pred", required=True)
    visualize.add_argument("--img-dir", required=True)
    visualize.add_argument("--out-dir", required=True)
    visualize.add_argument("--gt")
    visualize.add_argument("--limit", type=int, default=100)

    return parser


def main():
    args = build_parser().parse_args()

    if args.cmd == "make-ex":
        print(f"Wrote {write_example_submission(args.gt, args.out)} rows to {args.out}")
        return

    if args.cmd == "predict":
        result = predict_directory(
            weights_path=args.weights,
            img_dir=args.img_dir,
            out_path=args.out,
            conf_th=args.conf,
            nms_iou=args.nms_iou,
            limit=args.limit,
        )
        print(f"Device: {result['device']}")
        print(f"Weights: {result['weights']}")
        print(f"Model type: {result['model_type']}")
        print(f"Images processed: {result['images_processed']}")
        print(f"Boxes written: {result['boxes_written']}")
        print(f"Predictions saved to {result['out_path']}")
        return

    if args.cmd == "visualize":
        written = visualize_submission(args.pred, args.img_dir, args.out_dir, gt_path=args.gt, limit=args.limit)
        print(f"Saved {written} visualizations to {args.out_dir}")
        return

    pred_path = predict_if_needed(args)
    result = evaluate(
        args.gt,
        pred_path,
        args.iou,
        by_condition=args.by_condition,
        print_class=args.print_class,
        pred_class_offset=-1 if args.obo else 0,
    )
    print_report(result, by_condition=args.by_condition)

    model_name = os.path.splitext(os.path.basename(args.weights or pred_path))[0]
    if args.confusion_viz:
        save_box_confusion_matrix_plot(result, args.confusion_viz, model_name)
    if args.class_heatmap_viz:
        save_class_confusion_heatmap(result, args.class_heatmap_viz, model_name, list_classes_from_gt(args.gt))
    if args.class_log:
        save_class_log(result, args.class_log)


if __name__ == "__main__":
    main()
