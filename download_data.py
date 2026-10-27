"""
download_data.py
----------------
Helper script to manage, inspect, and organize vehicle datasets for Vehicle Type Classification.
Classes: Car, Bus, Truck, Motorcycle

Usage:
    python download_data.py --status         # Check current dataset folder status (train, val, test)
    python download_data.py --create-sample   # Create or update starter dataset (train, val, test)
    python download_data.py --organize <dir> # Automatically split an unorganized folder into train/val/test
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Ensure utf-8 stdout encoding if possible
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CLASSES = ["Car", "Bus", "Truck", "Motorcycle"]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
TEST_SAMPLES_DIR = BASE_DIR / "test_samples"


def ensure_directories():
    """Ensure all required train, val, test, and test_samples directories exist."""
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in CLASSES:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)
    TEST_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def count_images(directory: Path) -> dict:
    """Count valid images (.jpg, .jpeg, .png, .bmp, .webp) in class subdirectories."""
    counts = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for cls in CLASSES:
        cls_dir = directory / cls
        if cls_dir.exists():
            files = [f for f in cls_dir.iterdir() if f.suffix.lower() in valid_exts]
            counts[cls] = len(files)
        else:
            counts[cls] = 0
    return counts


def check_status():
    """Display count of images currently present in train, val, and test splits."""
    ensure_directories()
    print("=" * 65)
    print("[*] VEHICLE DATASET STATUS REPORT")
    print("=" * 65)

    train_counts = count_images(TRAIN_DIR)
    val_counts = count_images(VAL_DIR)
    test_counts = count_images(TEST_DIR)

    print(f"\n1. Training set path:   {TRAIN_DIR}")
    for cls in CLASSES:
        print(f"   - {cls:<12}: {train_counts[cls]} images")
    print(f"   Total Training Images:   {sum(train_counts.values())}")

    print(f"\n2. Validation set path: {VAL_DIR}")
    for cls in CLASSES:
        print(f"   - {cls:<12}: {val_counts[cls]} images")
    print(f"   Total Validation Images: {sum(val_counts.values())}")

    print(f"\n3. Test set path:       {TEST_DIR}")
    for cls in CLASSES:
        print(f"   - {cls:<12}: {test_counts[cls]} images")
    print(f"   Total Test Images:       {sum(test_counts.values())}")

    total_imgs = sum(train_counts.values()) + sum(val_counts.values()) + sum(test_counts.values())
    print("\n" + "-" * 65)
    print(f"Total dataset images across all splits: {total_imgs}")
    print("-" * 65)

    print("\n[RECOMMENDED PUBLIC DATASETS]")
    print("For real-world performance, download a public 4-class vehicle dataset:")
    print("  1. Kaggle: 'Vehicle Type Recognition'")
    print("     URL: https://www.kaggle.com/datasets/iamsouravbanerjee/vehicle-type-recognition")
    print("     Contains ~400 images each of Car, Bus, Truck, and Motorcycle.")
    print("  2. Kaggle: 'Vehicle Type Recognition (brj090)'")
    print("     URL: https://www.kaggle.com/datasets/brj090/vehicle-type-recognition")
    print("\nAfter downloading, place images into data/train, data/val, and data/test,")
    print("or use: python download_data.py --organize <downloaded_folder_path>")
    print("=" * 65)


def create_sample_vehicle_image(cls_name: str, index: int, size=(224, 224)) -> Image.Image:
    """Creates a distinct synthetic vehicle silhouette test image."""
    colors = {
        "Car": (41, 128, 185),        # Blue
        "Bus": (241, 196, 15),        # Yellow
        "Truck": (231, 76, 60),       # Red
        "Motorcycle": (46, 204, 113)  # Green
    }
    bg_color = (245, 245, 245)
    main_color = colors.get(cls_name, (100, 100, 100))

    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)

    rng = np.random.RandomState(index * 17 + hash(cls_name) % 1000)
    noise_offset = rng.randint(-8, 8)

    if cls_name == "Car":
        draw.rectangle([30, 110 + noise_offset, 194, 160 + noise_offset], fill=main_color)
        draw.polygon([(60, 110 + noise_offset), (90, 70 + noise_offset),
                      (140, 70 + noise_offset), (170, 110 + noise_offset)], fill=main_color)
        draw.ellipse([50, 150 + noise_offset, 80, 180 + noise_offset], fill=(40, 40, 40))
        draw.ellipse([144, 150 + noise_offset, 174, 180 + noise_offset], fill=(40, 40, 40))

    elif cls_name == "Bus":
        draw.rectangle([25, 60 + noise_offset, 199, 165 + noise_offset], fill=main_color)
        for wx in range(35, 180, 35):
            draw.rectangle([wx, 75 + noise_offset, wx + 25, 105 + noise_offset], fill=(220, 240, 255))
        draw.ellipse([50, 155 + noise_offset, 80, 185 + noise_offset], fill=(40, 40, 40))
        draw.ellipse([144, 155 + noise_offset, 174, 185 + noise_offset], fill=(40, 40, 40))

    elif cls_name == "Truck":
        draw.rectangle([25, 70 + noise_offset, 140, 160 + noise_offset], fill=(120, 120, 120))
        draw.rectangle([140, 95 + noise_offset, 195, 160 + noise_offset], fill=main_color)
        draw.ellipse([45, 150 + noise_offset, 75, 180 + noise_offset], fill=(40, 40, 40))
        draw.ellipse([85, 150 + noise_offset, 115, 180 + noise_offset], fill=(40, 40, 40))
        draw.ellipse([155, 150 + noise_offset, 185, 180 + noise_offset], fill=(40, 40, 40))

    elif cls_name == "Motorcycle":
        draw.ellipse([30, 130 + noise_offset, 75, 175 + noise_offset], fill=(40, 40, 40), outline=(20, 20, 20), width=4)
        draw.ellipse([149, 130 + noise_offset, 194, 175 + noise_offset], fill=(40, 40, 40), outline=(20, 20, 20), width=4)
        draw.line([(52, 152 + noise_offset), (105, 110 + noise_offset), (172, 152 + noise_offset)], fill=main_color, width=8)
        draw.line([(105, 110 + noise_offset), (130, 90 + noise_offset)], fill=main_color, width=6)
        draw.line([(120, 90 + noise_offset), (145, 90 + noise_offset)], fill=(40, 40, 40), width=4)

    draw.text((10, 10), f"{cls_name} #{index}", fill=(60, 60, 60))
    return img


def create_sample_dataset(samples_train=30, samples_val=10, samples_test=10):
    """Creates starter sample images for train, val, and test splits."""
    ensure_directories()
    print("[*] Generating starter sample images for train, val, and test splits...")

    for cls in CLASSES:
        # Train
        cls_dir = TRAIN_DIR / cls
        for i in range(samples_train):
            img = create_sample_vehicle_image(cls, index=i + 1)
            img.save(cls_dir / f"{cls.lower()}_{i+1:03d}.png")
        print(f"  [+] Created {samples_train} train images for '{cls}'")

        # Val
        cls_dir = VAL_DIR / cls
        for i in range(samples_val):
            img = create_sample_vehicle_image(cls, index=100 + i + 1)
            img.save(cls_dir / f"{cls.lower()}_val_{i+1:03d}.png")
        print(f"  [+] Created {samples_val} val images for '{cls}'")

        # Test
        cls_dir = TEST_DIR / cls
        for i in range(samples_test):
            img = create_sample_vehicle_image(cls, index=200 + i + 1)
            img.save(cls_dir / f"{cls.lower()}_test_{i+1:03d}.png")
        print(f"  [+] Created {samples_test} test images for '{cls}'")

        # Sample for Streamlit app
        sample_img = create_sample_vehicle_image(cls, index=999)
        sample_img.save(TEST_SAMPLES_DIR / f"test_{cls.lower()}.png")

    print("\n[+] Starter dataset successfully updated with train, val, and test sets!")


def organize_dataset(source_path_str: str, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Takes an unorganized directory or .zip file containing folders for Car, Bus, Truck, Motorcycle,
    and automatically splits real images into data/train, data/val, and data/test.
    """
    import zipfile
    import tempfile

    src = Path(source_path_str)
    if not src.exists():
        print(f"[ERROR] Source file or folder does not exist: {src}")
        return

    # If the user passed a .zip file, extract it to a temporary directory
    temp_dir_obj = None
    if src.is_file() and src.suffix.lower() == ".zip":
        print(f"[*] Extracting ZIP archive: {src} ...")
        temp_dir_obj = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(temp_dir_obj.name)
        search_root = Path(temp_dir_obj.name)
    else:
        search_root = src

    ensure_directories()
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Define synonyms for folder matching
    synonyms = {
        "Car": ["car", "cars", "automobile"],
        "Bus": ["bus", "buses"],
        "Truck": ["truck", "trucks"],
        "Motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes", "bike", "bikes"]
    }

    # Find candidate directories recursively
    all_subdirs = [d for d in search_root.rglob("*") if d.is_dir()]

    # Clear existing synthetic starter images if transitioning to real images
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                for f in cls_dir.iterdir():
                    if f.name.startswith(cls.lower()) and f.suffix.lower() == ".png":
                        try:
                            f.unlink()
                        except Exception:
                            pass

    print(f"[*] Organizing images from: {src}")
    for cls in CLASSES:
        cls_keywords = synonyms.get(cls, [cls.lower()])
        matching_dirs = [
            d for d in all_subdirs
            if any(kw == d.name.lower() or f"{kw}s" == d.name.lower() or kw in d.name.lower() for kw in cls_keywords)
        ]

        images = []
        for cdir in matching_dirs:
            images.extend([f for f in cdir.glob("*") if f.suffix.lower() in valid_exts])

        images = list(set(images))
        np.random.seed(42)
        np.random.shuffle(images)

        n_total = len(images)
        if n_total == 0:
            print(f"  [!] Notice: No images found matching class '{cls}'")
            continue

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img_path in train_imgs:
            shutil.copy2(img_path, TRAIN_DIR / cls / img_path.name)
        for img_path in val_imgs:
            shutil.copy2(img_path, VAL_DIR / cls / img_path.name)
        for img_path in test_imgs:
            shutil.copy2(img_path, TEST_DIR / cls / img_path.name)

        print(f"  [+] {cls:<12}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test (Total: {n_total})")

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    print("\n[+] Dataset organization complete!")
    check_status()


def main():
    parser = argparse.ArgumentParser(description="Vehicle Type Classification Dataset Manager")
    parser.add_argument("--status", action="store_true", help="Check status of train/val/test folders")
    parser.add_argument("--create-sample", action="store_true", help="Generate starter images for pipeline testing")
    parser.add_argument("--organize", type=str, help="Path to raw folder to automatically split into train/val/test")
    args = parser.parse_args()

    if args.create_sample:
        create_sample_dataset()
    elif args.organize:
        organize_dataset(args.organize)
    else:
        check_status()


if __name__ == "__main__":
    main()
