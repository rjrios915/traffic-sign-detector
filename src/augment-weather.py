import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(PROJECT_ROOT) == "src":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm")
INPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "weather", "TestIJCNN2013", "TestIJCNN2013Download"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "weather", "test_augmented")
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def slightly_darker(img):
    return ImageEnhance.Brightness(img).enhance(0.45)

def foggy(img):
    width, height = img.size
    base = np.asarray(img).astype(np.float32)
    haze = np.full((height, width, 3), 210, dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=18.0, size=(height, width, 1)).astype(np.float32)
    fog = np.clip(haze + noise, 0, 255)
    mixed = np.clip((0.55 * base) + (0.45 * fog), 0, 255).astype(np.uint8)
    return Image.fromarray(mixed).filter(ImageFilter.GaussianBlur(radius=2.2))

def rainy(img):
    width, height = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    streak_count = max(260, (width * height) // 1100)

    for _ in range(streak_count):
        x1 = random.randint(0, width - 1)
        y1 = random.randint(-height // 8, height - 1)
        length = random.randint(max(12, height // 24), max(18, height // 9))
        dx = random.randint(-5, 5)
        x2 = min(width - 1, max(0, x1 + dx))
        y2 = min(height - 1, max(0, y1 + length))
        alpha = random.randint(65, 125)
        draw.line((x1, y1, x2, y2), fill=(210, 220, 255, alpha), width=2)

    rainy_img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    rainy_img = rainy_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    rainy_img = ImageEnhance.Brightness(rainy_img).enhance(0.76)
    rainy_img = ImageEnhance.Contrast(rainy_img).enhance(0.88)
    return rainy_img

def snowy(img):
    width, height = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    flake_count = max(220, (width * height) // 1500)

    for _ in range(flake_count):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        radius = random.randint(1, 4)
        alpha = random.randint(170, 245)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 255, alpha))

    snowy_img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    snowy_img = ImageEnhance.Brightness(snowy_img).enhance(1.1)
    return snowy_img.filter(ImageFilter.GaussianBlur(radius=0.5))

def save_variants(input_path, output_dir, seed):
    set_seed(seed)
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        variants = {
            "darker": slightly_darker(img),
            "foggy": foggy(img),
            "rainy": rainy(img),
            "snowy": snowy(img),
        }

    os.makedirs(output_dir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(input_path))
    written = []
    for name, variant in variants.items():
        out_path = os.path.join(output_dir, f"{stem}_{name}{ext or '.png'}")
        variant.save(out_path)
        written.append(out_path)
    return written

def list_images(path):
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.lower().endswith(VALID_EXTS)
        )
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(f"missing path: {path}")

def main():
    images = list_images(INPUT_PATH)
    total_written = 0
    for idx, image_path in enumerate(images):
        written = save_variants(image_path, OUTPUT_DIR, SEED + idx)
        total_written += len(written)
        print(f"{image_path} -> {len(written)} variants")

    print(f"Processed {len(images)} images and wrote {total_written} files to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
