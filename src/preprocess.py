import os
import glob
import shutil
import gc
import sys
import time
import resource
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(PROJECT_ROOT) == "src":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

SOURCE_ROOT = os.path.join(PROJECT_ROOT, "data", "detection_dataset")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "detection_dataset_resized")
IMG_SIZE = 256
SPLITS = ["train", "val"]

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BATCH_SIZE = 200
WORKERS = max(1, (os.cpu_count() or 4) - 1)
OVERWRITE = False


def get_rss_mb():
    # macOS ru_maxrss is bytes, Linux is KB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def resize_image(in_path, out_path, img_size):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with Image.open(in_path) as img:
        resized = img.convert("RGB").resize((img_size, img_size))
        resized.save(out_path)
        resized.close()


def copy_label_if_exists(in_img_path, img_dir, label_dir, out_label_dir):
    rel = os.path.relpath(in_img_path, img_dir)
    label_rel = os.path.splitext(rel)[0] + ".txt"

    src_label = os.path.join(label_dir, label_rel)
    dst_label = os.path.join(out_label_dir, label_rel)

    if os.path.exists(src_label):
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)
        shutil.copy2(src_label, dst_label)


def process_one(args):
    in_img_path, img_dir, label_dir, out_img_dir, out_label_dir, img_size, overwrite = args
    rel = os.path.relpath(in_img_path, img_dir)
    out_img_path = os.path.join(out_img_dir, rel)

    wrote = 0
    skipped = 0
    if (not overwrite) and os.path.exists(out_img_path):
        skipped = 1
    else:
        resize_image(in_img_path, out_img_path, img_size)
        wrote = 1

    copy_label_if_exists(in_img_path, img_dir, label_dir, out_label_dir)
    return wrote, skipped


def preprocess_split(split):
    img_dir = os.path.join(SOURCE_ROOT, "images", split)
    label_dir = os.path.join(SOURCE_ROOT, "labels", split)

    out_img_dir = os.path.join(OUTPUT_ROOT, "images", split)
    out_label_dir = os.path.join(OUTPUT_ROOT, "labels", split)

    pattern = os.path.join(img_dir, "**", "*")
    total = sum(
        1 for p in glob.iglob(pattern, recursive=True)
        if p.lower().endswith(VALID_EXTS)
    )
    if total == 0:
        print(f"[{split}] no images found")
        return

    img_paths = [
        p for p in glob.iglob(pattern, recursive=True)
        if p.lower().endswith(VALID_EXTS)
    ]

    processed = 0
    resized = 0
    skipped = 0
    split_start = time.perf_counter()
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for batch_idx in tqdm(
            range(num_batches),
            total=num_batches,
            desc=f"{split} batches",
            unit="batch",
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, total)
            batch_paths = img_paths[start:end]
            batch_start = time.perf_counter()
            jobs = [
                (p, img_dir, label_dir, out_img_dir, out_label_dir, IMG_SIZE, OVERWRITE)
                for p in batch_paths
            ]

            for wrote, was_skipped in executor.map(process_one, jobs, chunksize=16):
                resized += wrote
                skipped += was_skipped
                processed += 1

            gc.collect()

            now = time.perf_counter()
            batch_s = max(now - batch_start, 1e-9)
            total_s = max(now - split_start, 1e-9)
            batch_rate = len(batch_paths) / batch_s
            avg_rate = processed / total_s
            print(
                f"[{split}] batch {batch_idx + 1}/{num_batches} | "
                f"{processed}/{total} | "
                f"batch {batch_rate:.2f} img/s | "
                f"avg {avg_rate:.2f} img/s | "
                f"resized {resized} | skipped {skipped} | "
                f"rss {get_rss_mb():.1f} MB"
            )

    gc.collect()
    split_total_s = max(time.perf_counter() - split_start, 1e-9)
    print(
        f"[{split}] processed {processed} images in {split_total_s:.1f}s "
        f"({processed / split_total_s:.2f} img/s), resized {resized}, skipped {skipped}, "
        f"rss {get_rss_mb():.1f} MB"
    )


def main():
    print(f"source: {SOURCE_ROOT}")
    print(f"output: {OUTPUT_ROOT}")
    print(f"size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"workers: {WORKERS}")
    print(f"overwrite: {OVERWRITE}")

    for split in SPLITS:
        preprocess_split(split)

    print("done")


if __name__ == "__main__":
    main()
