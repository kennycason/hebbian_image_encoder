# download_tiny_imagenet.py

from datasets import load_dataset
from PIL import Image
from pathlib import Path
import argparse
import shutil

# === Config ===
SAVE_DIR = Path("tiny_imagenet")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def download_tiny_imagenet(n_images=1000, split="train"):
    print(f"[INFO] Downloading {n_images} images from {split} split...")
    dataset = load_dataset("slegroux/tiny-imagenet-200-clean", split=split)
    dataset = dataset.shuffle(seed=42).select(range(n_images))

    image_dir = SAVE_DIR / split
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(dataset):
        img: Image.Image = item["image"]
        label = item["label"]
        label_dir = image_dir / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        img.save(label_dir / f"img_{i:04d}.png")
    print(f"[DONE] Saved {n_images} images to {image_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Number of images to download")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/validation/test)")
    args = parser.parse_args()
    download_tiny_imagenet(n_images=args.n, split=args.split)
