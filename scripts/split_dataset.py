#!/usr/bin/env python3
"""
Split dataset into train/val with CLI.

Usage:
python scripts/split_dataset.py --src dataset/aligned --dst dataset --train_ratio 0.8
"""

import os, shutil, random
import argparse

def split_dataset(src, dst, train_ratio=0.8):
    # Create output directories
    train_root = os.path.join(dst, "train")
    val_root = os.path.join(dst, "val")

    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

    print(f"\n Source folder: {src}")
    print(f" Destination dataset root: {dst}")
    print(f" Train ratio: {train_ratio}\n")

    for cls in sorted(os.listdir(src)):
        cls_path = os.path.join(src, cls)
        if not os.path.isdir(cls_path):
            continue

        # get image list
        imgs = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(imgs) == 0:
            print(f"⚠ No images in class {cls}, skipping...")
            continue

        random.shuffle(imgs)
        split = int(len(imgs) * train_ratio)

        train_imgs = imgs[:split]
        val_imgs = imgs[split:]

        # class folders
        train_class_dir = os.path.join(train_root, cls)
        val_class_dir   = os.path.join(val_root, cls)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir,   exist_ok=True)

        # Copy train images
        for f in train_imgs:
            shutil.copy(os.path.join(cls_path, f), os.path.join(train_class_dir, f))

        # Copy val images
        for f in val_imgs:
            shutil.copy(os.path.join(cls_path, f), os.path.join(val_class_dir, f))

        print(f"✔ {cls}: train {len(train_imgs)}, val {len(val_imgs)}")

    print("\n Dataset split complete!")


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val sets.")
    parser.add_argument("--src", required=True, help="Source dataset folder (e.g., dataset/aligned)")
    parser.add_argument("--dst", required=True, help="Destination dataset root (e.g., dataset)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training images (default 0.8)")

    args = parser.parse_args()

    split_dataset(args.src, args.dst, args.train_ratio)
