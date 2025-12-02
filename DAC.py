#!/usr/bin/env python3
"""
Dynamic approximate classification demo script (batched version).

- Draw multiple images from a directory.
- For each image:
    * Look up its distance from metadata.json.
    * Use a threshold rule on distance to choose between:
        - MobileNetV3-Small (fast, fine-tuned)
        - MobileNetV3-Large (bigger, more accurate, fine-tuned)
    * Run the chosen model and print:
        - Ground-truth label (inferred from metadata)
        - Predicted top-1 label (class_id)
        - Whether prediction is correct
        - Top-k label distribution and confidences

Key points:
- We only load each model ONCE.
- Larger distance means the object is CLOSER (more urgent).
  => For distance >= threshold, we use the SMALL (fast) model.
- Ground-truth labels are inferred from bounding_boxes[*].class_id
  via majority vote.
"""

import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms  # noqa: F401
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import parameters as cfg  # all config flags live here


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


def infer_image_label_from_metadata(image_meta):
    """
    Given a single image's metadata entry:

        {
            "bounding_boxes": [
                {"xmin": ..., "xmax": ..., "ymin": ..., "ymax": ..., "class_id": ...},
                ...
            ],
            "distance": ...
        }

    Return the class_id of the *largest* bounding box (by area).
    """

    boxes = image_meta.get("bounding_boxes", [])
    if not boxes:
        return None  # or raise an error, depending on your pipeline

    def box_area(box):
        width = max(0, box["xmax"] - box["xmin"])
        height = max(0, box["ymax"] - box["ymin"])
        return width * height

    # Pick the bounding box with the largest area
    largest_box = max(boxes, key=box_area)

    # Return its class_id (e.g., 1..5)
    return largest_box["class_id"]



def sample_image_paths_with_label_diversity(
    valid_paths: list[Path],
    metadata: dict,
    num_samples: int,
) -> list[Path]:
    """
    Sample image paths while encouraging diversity across inferred ground-truth labels.

    We infer an image-level label using infer_image_label_from_metadata(meta)
    (majority class_id over bounding_boxes) and then do round-robin sampling
    across labels so we don't end up with all images from the same class.

    Falls back to uniform random sampling if labels cannot be inferred.
    """
    # Map filename -> Path
    filename_to_path = {p.name: p for p in valid_paths}

    # Build label -> list[Path]
    label_to_paths: dict = {}
    for fname, meta in metadata.items():
        if fname not in filename_to_path:
            continue
        label = infer_image_label_from_metadata(meta)
        if label is None:
            continue
        label_to_paths.setdefault(label, []).append(filename_to_path[fname])

    # If we failed to assign any labels, just random sample
    if not label_to_paths:
        return random.sample(valid_paths, k=num_samples)

    # Shuffle each label bucket so we don't always take same examples
    for paths in label_to_paths.values():
        random.shuffle(paths)

    # Round-robin across labels
    sampled: list[Path] = []
    labels = list(label_to_paths.keys())
    label_idx = 0

    while len(sampled) < num_samples and any(label_to_paths.values()):
        label = labels[label_idx % len(labels)]
        bucket = label_to_paths[label]
        if bucket:
            sampled.append(bucket.pop())
        label_idx += 1

    # If still need more images, fill from remaining pool
    if len(sampled) < num_samples:
        remaining = [p for paths in label_to_paths.values() for p in paths]
        # Also allow any other valid_paths that haven't been used yet
        remaining += [p for p in valid_paths if p not in sampled]

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for p in remaining:
            if p not in seen:
                seen.add(p)
                deduped.append(p)

        needed = num_samples - len(sampled)
        if len(deduped) > needed:
            sampled.extend(random.sample(deduped, k=needed))
        else:
            sampled.extend(deduped)

    return sampled[:num_samples]


# -----------------------------
# Checkpoint helper
# -----------------------------


def _extract_state_dict(ckpt):
    """
    Handle different checkpoint formats.

    Supports:
      - {"model_state_dict": ...}
      - {"model_state": ...}
      - {"state_dict": ...}
      - or a bare state_dict
    """
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            if key in ckpt:
                return ckpt[key]
        # Fall back: maybe the dict itself is already a state_dict
        return ckpt
    # Non-dict -> assume it's already a state_dict
    return ckpt


# -----------------------------
# Model loading & preprocessing (fine-tuned)
# -----------------------------


def load_mobilenet_v3_small(img_size: int, ckpt_path: str):
    """
    Load fine-tuned MobileNetV3-Sall and its transforms.

    - Architecture: torchvision mobilenet_v3_small
    - Head: replaced with NUM_CLASSES output
    - Weights: loaded from fine-tuned checkpoint
    """
    # Reuse ImageNet transforms (same as used in finetuning script)
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    transform = weights.transforms(crop_size=img_size)

    # Build model skeleton (no pretrained weights; checkpoint will provide them)
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, cfg.NUM_CLASSES)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    model.load_state_dict(state_dict)

    # Use id2name from checkpoint to build label list in index order
    if isinstance(ckpt, dict):
        id2name = ckpt.get(
            "id2name",
            {i + 1: str(i + 1) for i in range(cfg.NUM_CLASSES)},
        )
    else:
        id2name = {i + 1: str(i + 1) for i in range(cfg.NUM_CLASSES)}

    categories = [id2name[i] for i in sorted(id2name.keys())]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, transform, categories


def load_mobilenet_v3_large(ckpt_path: str):
    """
    Load fine-tuned MobileNetV3-Large (timm mobilenetv3_large_100) and its transforms.

    - Architecture: timm mobilenetv3_large_100(num_classes=NUM_CLASSES)
    - Weights: loaded from fine-tuned checkpoint
    """
    # Load checkpoint for id2name and weights
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        id2name = ckpt.get(
            "id2name",
            {i + 1: str(i + 1) for i in range(cfg.NUM_CLASSES)},
        )
    else:
        id2name = {i + 1: str(i + 1) for i in range(cfg.NUM_CLASSES)}

    # Build model skeleton with correct head size
    model = timm.create_model(
        "mobilenetv3_large_100",
        pretrained=False,
        num_classes=cfg.NUM_CLASSES,
    )

    state_dict = _extract_state_dict(ckpt)
    model.load_state_dict(state_dict)

    # Configure data transforms based on model's default config
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    categories = [id2name[i] for i in sorted(id2name.keys())]
    return model, transform, categories


# -----------------------------
# Inference helper
# -----------------------------


@torch.no_grad()
def run_single_image(
    model,
    transform,
    categories,
    image_path: Path,
    topk: int = 5,
):
    """
    Run a single image through a given model & transform.
    Returns:
        (topk_results, latency_sec, top1_idx)
    where:
        - topk_results is a list of (label_name, prob) pairs.
        - top1_idx is the integer class index (0-based) of the top-1 prediction.
    """
    device = next(model.parameters()).device

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Forward pass with timing
    start = time.perf_counter()
    outputs = model(input_tensor)
    end = time.perf_counter()
    latency = end - start

    probs = torch.softmax(outputs[0], dim=0)
    top_probs, top_idxs = probs.topk(topk)
    topk_results = [
        (categories[idx], float(prob)) for idx, prob in zip(top_idxs, top_probs)
    ]
    top1_idx = int(top_idxs[0])
    return topk_results, latency, top1_idx


# -----------------------------
# Model selection rule
# -----------------------------


def select_model_by_distance(distance: float, threshold: float) -> str:
    """
    Decide whether to use "small" or "large" model.

    Larger distance => object is closer / more urgent.

    Rule:
      - If distance >= threshold -> use "small" (fast)
      - Else                     -> use "large"
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

    # Choose sampling strategy
    sampling_mode = getattr(cfg, "SAMPLING_MODE", "random")
    if sampling_mode == "balanced_by_label":
        sampled_paths = sample_image_paths_with_label_diversity(
            valid_paths=valid_paths,
            metadata=metadata,
            num_samples=num_samples,
        )
    else:
        sampled_paths = random.sample(valid_paths, k=num_samples)

    print("=== Dynamic Approximate Classification (Batch Demo, Fine-Tuned) ===")
    print(f"Image directory        : {cfg.IMAGE_DIR}")
    print(f"Metadata path          : {cfg.METADATA_PATH}")
    print(f"Distance threshold     : {cfg.DISTANCE_THRESHOLD:.6f}")
    print(f"Number of images (pool): {len(valid_paths)}")
    print(f"Number of samples      : {num_samples}")
    print(f"Sampling mode          : {sampling_mode}")
    print(f"Num threads (torch)    : {cfg.NUM_THREADS}")
    print(f"Num classes            : {cfg.NUM_CLASSES}")

    # Show inferred label distribution in the sampled set
    label_counts = {}
    for p in sampled_paths:
        meta = metadata.get(p.name, {})
        lbl = infer_image_label_from_metadata(meta)
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print("--------------------------------------------------------")
    print("Label distribution in sampled images (inferred_label -> count):")
    for lbl, count in sorted(label_counts.items(), key=lambda x: str(x[0])):
        print(f"  {lbl}: {count}")
    print("--------------------------------------------------------")

    # Load both fine-tuned models ONCE
    print("Loading fine-tuned MobileNetV3-Small...")
    small_model, small_transform, small_categories = load_mobilenet_v3_small(
        cfg.IMG_SIZE, cfg.SMALL_CKPT_PATH
    )
    print("Loading fine-tuned MobileNetV3-Large...")
    large_model, large_transform, large_categories = load_mobilenet_v3_large(
        cfg.LARGE_CKPT_PATH
    )
    print("Models loaded. Starting per-image inference...\n")

    # Counters
    stats = {
        "small": 0,
        "large": 0,
    }
    total_latency_small = 0.0
    total_latency_large = 0.0

    # Accuracy counters
    total_labeled = 0
    total_correct = 0

    small_labeled = 0
    small_correct = 0
    large_labeled = 0
    large_correct = 0

    for idx, img_path in enumerate(sampled_paths, start=1):
        img_name = img_path.name

        if img_name not in metadata:
            # Should not happen given how we built valid_paths, but just in case.
            print(f"[WARN] {img_name} not in metadata; skipping.")
            continue

        img_meta = metadata[img_name]
        distance = img_meta["distance"]
        gt_label = infer_image_label_from_metadata(img_meta)  # class_id or None

        model_choice = select_model_by_distance(distance, cfg.DISTANCE_THRESHOLD)

        if model_choice == "small":
            model, transform, categories = small_model, small_transform, small_categories
        else:
            model, transform, categories = large_model, large_transform, large_categories

        stats[model_choice] += 1

        topk_results, latency, top1_idx = run_single_image(
            model=model,
            transform=transform,
            categories=categories,
            image_path=img_path,
            topk=cfg.TOPK,
        )

        # Convert top1_idx (0-based) to class_id (1-based) to compare with metadata class_id.
        pred_class_id = top1_idx + 1

        # Track latency stats
        if model_choice == "small":
            total_latency_small += latency
        else:
            total_latency_large += latency

        # Accuracy bookkeeping (only if we have a ground-truth label)
        correct_flag = None
        if gt_label is not None:
            correct_flag = (pred_class_id == gt_label)
            total_labeled += 1
            if correct_flag:
                total_correct += 1

            if model_choice == "small":
                small_labeled += 1
                if correct_flag:
                    small_correct += 1
            else:
                large_labeled += 1
                if correct_flag:
                    large_correct += 1

        # Per-image output
        print(f"Image {idx}/{num_samples}: {img_name}")
        print(f"  Distance label           : {distance:.6f} (larger = closer)")
        print(f"  Selected model           : MobileNetV3-{model_choice.upper()} (fine-tuned)")

        if gt_label is not None:
            status = "CORRECT" if correct_flag else "WRONG"
            print(f"  Ground-truth class_id    : {gt_label}")
            print(f"  Predicted top-1 class_id : {pred_class_id}  --> {status}")
        else:
            print("  Ground-truth class_id    : None (could not infer from metadata)")
            print(f"  Predicted top-1 class_id : {pred_class_id}")

        print(f"  Inference latency        : {latency * 1000.0:.2f} ms")
        print(f"  Top-{cfg.TOPK} label distribution (name, confidence):")
        for rank, (label_name, prob) in enumerate(topk_results, start=1):
            print(f"    {rank}. {label_name:30s}  {prob * 100.0:6.2f}%")
        print("--------------------------------------------------------")

    # Summary
    print("\n=== Summary over sampled images ===")
    print(f"Total images processed         : {num_samples}")
    print(f"Images using SMALL model       : {stats['small']}")
    print(f"Images using LARGE model       : {stats['large']}")

    if stats["small"] > 0:
        avg_small = (total_latency_small / stats["small"]) * 1000.0
        print(f"Avg SMALL model latency        : {avg_small:.2f} ms")
    if stats["large"] > 0:
        avg_large = (total_latency_large / stats["large"]) * 1000.0
        print(f"Avg LARGE model latency        : {avg_large:.2f} ms")

    # Accuracy summary
    print("\n=== Accuracy (top-1, using inferred class_id) ===")
    if total_labeled > 0:
        overall_acc = 100.0 * total_correct / total_labeled
        print(
            f"Overall accuracy on labeled images : "
            f"{total_correct}/{total_labeled} = {overall_acc:.2f}%"
        )

        if small_labeled > 0:
            small_acc = 100.0 * small_correct / small_labeled
            print(
                f"SMALL model accuracy              : "
                f"{small_correct}/{small_labeled} = {small_acc:.2f}%"
            )
        else:
            print("SMALL model accuracy              : (no labeled images routed to SMALL)")

        if large_labeled > 0:
            large_acc = 100.0 * large_correct / large_labeled
            print(
                f"LARGE model accuracy              : "
                f"{large_correct}/{large_labeled} = {large_acc:.2f}%"
            )
        else:
            print("LARGE model accuracy              : (no labeled images routed to LARGE)")
    else:
        print("No ground-truth labels could be inferred; accuracy not computed.")


if __name__ == "__main__":
    main()
