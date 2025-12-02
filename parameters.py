"""
parameters.py

Configuration for the Dynamic Approximate Classification (DAC) demo.
"""

# -----------------------------
# Data / metadata
# -----------------------------
IMAGE_DIR = "train"
METADATA_PATH = "train/metadata.json"

# Threshold on distance for model selection.
# Larger distance => object is closer / more urgent.
# Rule in DAC.py:
#   if distance >= DISTANCE_THRESHOLD -> use SMALL model
#   else                              -> use LARGE model
DISTANCE_THRESHOLD = 0.05

# Number of images to randomly sample per run.
NUM_SAMPLES = 20

# Random seed for sampling images. Set to None for pure randomness.
RANDOM_SEED = 42

# Torch threading configuration.
NUM_THREADS = 8

# -----------------------------
# Model / classifier config
# -----------------------------

# Number of classes in your fine-tuned classifier.
# For {car, truck, pedestrian, bicyclist, light} -> 5
NUM_CLASSES = 5

# Paths to fine-tuned checkpoints produced by finetune_mobilenet.py
SMALL_CKPT_PATH = "ckpts_small/best_mobilenet_small_finetuned.pth"
LARGE_CKPT_PATH = "ckpts_large/best_mobilenet_large_finetuned.pth"

# Image size for MobileNetV3-Small (used in its torchvision transforms).
IMG_SIZE = 224

# How many top labels to print for each image.
TOPK = 5
