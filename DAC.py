#!/usr/bin/env python3
"""
Dynamic approximate classification demo script (batched version).

- Draw multiple random images from a directory.
- For each image:
    * Look up its distance label from metadata.json.
    * Use a simple threshold rule on distance to choose between:
        - MobileNetV3-Small (fast)
        - MobileNetV3-Large (bigger, more accurate)
    * Run the chosen model and print the label distribution and confidences.

Key points:
- We only load each model ONCE (no re-training / re-loading per image).
- Larger distance means the object is CLOSER (more urgent).
  => For distance >= threshold, we use the SMALL (fast) model.
- All configuration is in parameters.py.
"""

import json
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms  # noqa: F401 (import kept for clarity)

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import urllib.request

import parameters as cfg  # <-- all config flags live here


# -----------------------------
# Utility: load metadata & image list
# -----------------------------

def load_metadata(metadata_path: str):
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    return meta


def get_valid_image_paths(image_dir: str, metadata: dict) -> list[Path]:
    """
    Return all image paths in image_dir whose filenames appear as keys in metadata.
    """
    image_dir_path = Path(image_dir)
    valid_files = [
        image_dir_path / fname
        for fname in metadata.keys()
        if (image_dir_path / fname).is_file()
    ]

    if not valid_files:
        raise RuntimeError(
            f"No images in {image_dir_path} match any keys in metadata.json. "
            "Make sure your image filenames line up with the keys in metadata."
        )

    return valid_files


# -----------------------------
# Model loading & preprocessing
# -----------------------------

def load_mobilenet_v3_small(img_size: int):
    """
    Load MobileNetV3-Small with ImageNet weights and its transforms.
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights).eval()  # CPU by default
    transform = weights.transforms(crop_size=img_size)
    categories = weights.meta.get("categories")
    return model, transform, categories


def load_mobilenet_v3_large():
    """
    Load MobileNetV3-Large (from timm) with ImageNet weights and its transforms.
    """
    model = timm.create_model("mobilenetv3_large_100", pretrained=True)
    model.eval()

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    # Get ImageNet class labels (cached locally after first download)
    classes_file = "imagenet_classes.txt"
    if not os.path.exists(classes_file):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        print(f"Downloading ImageNet class names from {url} ...")
        urllib.request.urlretrieve(url, classes_file)

    with open(classes_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    return model, transform, categories


# -----------------------------
# Inference helpers
# -----------------------------

@torch.inference_mode()
def run_model(
    model,
    transform,
    categories,
    image_path: Path,
    topk: int = 5,
):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # [1,3,H,W]

    start = time.perf_counter()
    out = model(x)
    elapsed = time.perf_counter() - start

    # Convert logits to probabilities
    probs = torch.softmax(out[0], dim=0)  # [num_classes]

    # Clamp topk to number of classes
    topk = min(topk, probs.shape[0])
    top_vals, top_idxs = torch.topk(probs, k=topk)

    # Build nice structure: list of (label, confidence)
    results = []
    for idx, val in zip(top_idxs.tolist(), top_vals.tolist()):
        label = categories[idx] if categories is not None else str(idx)
        results.append((label, float(val)))

    return results, float(elapsed), probs, categories


# -----------------------------
# Model selection based on distance
# -----------------------------

def select_model_by_distance(distance: float, threshold: float) -> str:
    """
    Simple threshold model for the Vision Model Selector.

    Interpretation:
      - distance is (size of largest bounding box) / (size of image).
      - Larger distance  -> object appears closer -> more urgent.
      - More urgent      -> we want a faster (smaller) model.

    Rule:
      - If distance >= threshold -> use 'small'
      - Else                     -> use 'large'
    """
    if distance >= threshold:
        return "small"
    else:
        return "large"


# -----------------------------
# Main orchestration
# -----------------------------

def main():
    # Global threading config
    torch.set_num_threads(cfg.NUM_THREADS)

    # Seed for reproducibility of random image sampling
    if cfg.RANDOM_SEED is not None:
        random.seed(cfg.RANDOM_SEED)

    # Load metadata describing distances & bounding boxes
    metadata = load_metadata(cfg.METADATA_PATH)

    # All candidate images
    valid_paths = get_valid_image_paths(cfg.IMAGE_DIR, metadata)

    # How many images to sample this run
    num_samples = min(cfg.NUM_SAMPLES, len(valid_paths))
    sampled_paths = random.sample(valid_paths, k=num_samples)

    print("=== Dynamic Approximate Classification (Batch Demo) ===")
    print(f"Image directory        : {cfg.IMAGE_DIR}")
    print(f"Metadata path          : {cfg.METADATA_PATH}")
    print(f"Distance threshold     : {cfg.DISTANCE_THRESHOLD:.6f}")
    print(f"Number of images (pool): {len(valid_paths)}")
    print(f"Number of samples      : {num_samples}")
    print(f"Num threads (torch)    : {cfg.NUM_THREADS}")
    print("--------------------------------------------------------")

    # Load both models ONCE
    print("Loading MobileNetV3-Small...")
    small_model, small_transform, small_categories = load_mobilenet_v3_small(
        cfg.IMG_SIZE
    )
    print("Loading MobileNetV3-Large...")
    large_model, large_transform, large_categories = load_mobilenet_v3_large()
    print("Models loaded. Starting per-image inference...\n")

    stats = {
        "small": 0,
        "large": 0,
    }
    total_latency_small = 0.0
    total_latency_large = 0.0

    for idx, img_path in enumerate(sampled_paths, start=1):
        img_name = img_path.name

        if img_name not in metadata:
            # Should not happen given how we built valid_paths, but just in case.
            print(f"[WARN] {img_name} not in metadata; skipping.")
            continue

        distance = metadata[img_name]["distance"]
        model_choice = select_model_by_distance(distance, cfg.DISTANCE_THRESHOLD)

        if model_choice == "small":
            model, transform, categories = small_model, small_transform, small_categories
        else:
            model, transform, categories = large_model, large_transform, large_categories

        stats[model_choice] += 1

        topk_results, latency, full_probs, _ = run_model(
            model=model,
            transform=transform,
            categories=categories,
            image_path=img_path,
            topk=cfg.TOPK,
        )

        if model_choice == "small":
            total_latency_small += latency
        else:
            total_latency_large += latency

        # Per-image output
        print(f"Image {idx}/{num_samples}: {img_name}")
        print(f"  Distance label     : {distance:.6f} (larger = closer)")
        print(f"  Selected model     : MobileNetV3-{model_choice.upper()}")
        print(f"  Inference latency  : {latency * 1000.0:.2f} ms")
        print(f"  Top-{cfg.TOPK} label distribution (label, confidence):")
        for rank, (label, prob) in enumerate(topk_results, start=1):
            print(f"    {rank}. {label:30s}  {prob * 100.0:6.2f}%")
        print("--------------------------------------------------------")

    # Summary
    print("\n=== Summary over sampled images ===")
    print(f"Total images processed   : {num_samples}")
    print(f"Images using SMALL model : {stats['small']}")
    print(f"Images using LARGE model : {stats['large']}")

    if stats["small"] > 0:
        avg_small = (total_latency_small / stats["small"]) * 1000.0
        print(f"Avg SMALL model latency : {avg_small:.2f} ms")
    if stats["large"] > 0:
        avg_large = (total_latency_large / stats["large"]) * 1000.0
        print(f"Avg LARGE model latency : {avg_large:.2f} ms")


if __name__ == "__main__":
    main()
