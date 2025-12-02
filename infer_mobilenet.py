#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# -----------------------
# Custom label mapping
# -----------------------
# Same mapping as in the finetune script
ID2NAME = {
    1: "car",
    2: "truck",
    3: "pedestrian",
    4: "bicyclist",
    5: "light",
}


# -----------------------
# Model builders + transforms
# -----------------------

def build_mobilenet_small_for_inference(num_classes: int = 5):
    """
    Build MobileNetV3-Small with correct head and transforms for inference.
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)

    # Replace classifier to match our num_classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    # Use same evaluation transforms (ImageNet-style)
    transform = weights.transforms()

    return model, transform


def build_mobilenet_large_for_inference(num_classes: int = 5):
    """
    Build MobileNetV3-Large (timm) with correct head and transforms for inference.
    """
    model = timm.create_model(
        "mobilenetv3_large_100",
        pretrained=False,  # we'll load our fine-tuned weights
        num_classes=num_classes,
    )

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config, is_training=False)

    return model, transform


def build_model_and_transform_for_inference(model_name: str, num_classes: int = 5):
    model_name = model_name.lower()
    if model_name == "small":
        return build_mobilenet_small_for_inference(num_classes=num_classes)
    elif model_name == "large":
        return build_mobilenet_large_for_inference(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_name '{model_name}', expected 'small' or 'large'.")


# -----------------------
# Inference helper
# -----------------------

@torch.inference_mode()
def classify_image(model, transform, image_path: Path, device: torch.device):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

    logits = model(x)  # [1, num_classes]
    probs = torch.softmax(logits[0], dim=0)
    pred_idx = int(torch.argmax(probs).item())        # 0..4
    pred_prob = float(probs[pred_idx].item())

    # Map back to label_id {1..5} and name
    label_id = pred_idx + 1
    label_name = ID2NAME.get(label_id, f"unknown-{label_id}")

    return label_id, label_name, pred_prob, probs.cpu()


# -----------------------
# Argparse / main
# -----------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference with a fine-tuned MobileNetV3 (small or large) on a single image."
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the fine-tuned checkpoint (best_mobilenet_*.pth).",
    )
    p.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image to classify.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device, e.g. "cuda" or "cpu".',
    )
    p.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Optionally print top-k probabilities (k <= 5 classes).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = args.checkpoint
    img_path = Path(args.image)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------
    # Load checkpoint
    # -----------------------
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # We saved these in the training script:
    #   - "model_name": "small" or "large"
    #   - "id2name": ID2NAME dict (optional)
    model_name = ckpt.get("model_name", "small")
    saved_id2name = ckpt.get("id2name", None)
    if saved_id2name is not None:
        # Overwrite default mapping if present in checkpoint
        global ID2NAME
        ID2NAME = saved_id2name

    print(f"Loaded checkpoint from: {ckpt_path}")
    print(f"Model type from checkpoint: MobileNetV3-{model_name.upper()}")

    # -----------------------
    # Rebuild model + transforms
    # -----------------------
    num_classes = len(ID2NAME)
    model, transform = build_model_and_transform_for_inference(
        model_name=model_name,
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # -----------------------
    # Run inference
    # -----------------------
    label_id, label_name, pred_prob, probs = classify_image(
        model=model,
        transform=transform,
        image_path=img_path,
        device=device,
    )

    print("\n=== Inference Result ===")
    print(f"Image           : {img_path}")
    print(f"Predicted ID    : {label_id}")
    print(f"Predicted label : {label_name}")
    print(f"Confidence      : {pred_prob * 100.0:.2f}%")

    # Optionally show full distribution over 5 classes
    topk = min(args.topk, probs.shape[0])
    top_vals, top_idxs = torch.topk(probs, k=topk)
    print(f"\nTop-{topk} label distribution (ID, name, prob):")
    for rank, (idx, val) in enumerate(zip(top_idxs.tolist(), top_vals.tolist()), start=1):
        lid = idx + 1
        lname = ID2NAME.get(lid, f"unknown-{lid}")
        print(f"  {rank}. id={lid}, name={lname:11s}, prob={val * 100.0:6.2f}%")


if __name__ == "__main__":
    main()
