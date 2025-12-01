#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# -----------------------
# Custom label mapping
# -----------------------
ID2NAME = {
    1: "car",
    2: "truck",
    3: "pedestrian",
    4: "bicyclist",
    5: "light",
}
NAME2ID = {v: k for k, v in ID2NAME.items()}


# -----------------------
# Dataset
# -----------------------

class CustomLabeledDataset(Dataset):
    """
    CSV format:
        filename,label

    - filename: path relative to images_root (e.g. "img_001.jpg")
    - label: integer in {1,2,3,4,5}

    Internally we convert label ∈ {1..5} -> class_index ∈ {0..4}.
    """

    def __init__(self, images_root: str, csv_path: str, transform=None):
        self.images_root = Path(images_root)
        self.transform = transform

        self.samples = []  # list of (image_path, class_index_in_0_4)
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"]
                label_id = int(row["label"])
                if label_id not in ID2NAME:
                    raise ValueError(
                        f"Label id {label_id} not in expected set {list(ID2NAME.keys())}"
                    )
                img_path = self.images_root / fname
                if not img_path.is_file():
                    raise FileNotFoundError(f"Image not found: {img_path}")

                # Map {1..5} -> {0..4}
                class_index = label_id - 1
                self.samples.append((img_path, class_index))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded from CSV: {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_index = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, class_index


# -----------------------
# Training / evaluation
# -----------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)  # [B, num_classes]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        running_correct += (preds == targets).sum().item()
        running_total += inputs.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


@torch.inference_mode()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        running_correct += (preds == targets).sum().item()
        running_total += inputs.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


# -----------------------
# Model builders + transforms
# -----------------------

def build_mobilenet_small(num_classes: int = 5, freeze_backbone: bool = False):
    """
    torchvision MobileNetV3-Small
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    # Same transform for train/val for simplicity (ImageNet-style)
    train_transform = weights.transforms()
    val_transform = weights.transforms()

    return model, train_transform, val_transform


def build_mobilenet_large(num_classes: int = 5, freeze_backbone: bool = False):
    """
    timm MobileNetV3-Large (mobilenetv3_large_100)
    """
    # Create model with classifier already set to num_classes
    model = timm.create_model(
        "mobilenetv3_large_100",
        pretrained=True,
        num_classes=num_classes,
    )

    if freeze_backbone:
        # Freeze everything except classifier head
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    # Build transforms using timm's data config
    config = resolve_data_config({}, model=model)
    train_transform = create_transform(**config, is_training=True)
    val_transform = create_transform(**config, is_training=False)

    return model, train_transform, val_transform


def build_model_and_transforms(model_name: str, num_classes: int, freeze_backbone: bool):
    model_name = model_name.lower()
    if model_name == "small":
        return build_mobilenet_small(num_classes=num_classes, freeze_backbone=freeze_backbone)
    elif model_name == "large":
        return build_mobilenet_large(num_classes=num_classes, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model_name '{model_name}', expected 'small' or 'large'.")


# -----------------------
# Argparse / main
# -----------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune MobileNetV3 (small or large) on custom labels {1..5}"
    )
    p.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["small", "large"],
        help="Which MobileNetV3 to fine-tune: 'small' or 'large'.",
    )
    p.add_argument(
        "--train-images-root",
        type=str,
        required=True,
        help="Root directory containing TRAIN images."
    )
    p.add_argument(
        "--val-images-root",
        type=str,
        required=True,
        help="Root directory containing VAL images."
    )
    p.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="CSV with training data (filename,label)."
    )
    p.add_argument(
        "--val-csv",
        type=str,
        required=True,
        help="CSV with validation data (filename,label)."
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="If set, freeze backbone and only train classifier head.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save best model checkpoint."
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device, e.g. "cuda" or "cpu".'
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model choice: MobileNetV3-{args.model.upper()}")
    print(f"Train images root: {args.train_images_root}")
    print(f"Val images root  : {args.val_images_root}")

    # Build model + transforms
    model, train_transform, val_transform = build_model_and_transforms(
        model_name=args.model,
        num_classes=len(ID2NAME),
        freeze_backbone=args.freeze_backbone,
    )
    model.to(device)

    # Datasets and loaders
    train_dataset = CustomLabeledDataset(
        images_root=args.train_images_root,
        csv_path=args.train_csv,
        transform=train_transform,
    )
    val_dataset = CustomLabeledDataset(
        images_root=args.val_images_root,
        csv_path=args.val_csv,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(
        args.output_dir, f"best_mobilenet_{args.model}_finetuned.pth"
    )

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "id2name": ID2NAME,
                "model_name": args.model,
            },
            best_ckpt_path,
        )
            

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"Best checkpoint path: {best_ckpt_path}")


if __name__ == "__main__":
    main()
