"""
parameters.py

Configuration for the Dynamic Approximate Classification (DAC) demo.
"""

# -----------------------------
# Data / metadata
# -----------------------------
# Directory containing the images to sample from.
IMAGE_DIR = "val"

# Path to metadata.json describing each image.
# Schema per image (from metadata.json):
#   "<filename>.jpg": {
#       "bounding_boxes": [
#           {"xmin": ..., "xmax": ..., "ymin": ..., "ymax": ..., "class_id": int},
#           ...
#       ],
#       "distance": float
#   }
#
# We infer an image-level label as the majority class_id in bounding_boxes.
METADATA_PATH = "val/metadata.json"

# -----------------------------
# Distance threshold for model selection
# -----------------------------
# Larger distance => object is closer / more urgent.
# Rule in DAC.py:
#   if distance >= DISTANCE_THRESHOLD -> use SMALL model (fast)
#   else                              -> use LARGE model (bigger, more accurate)
DISTANCE_THRESHOLD = 0.05

# -----------------------------
# Sampling / randomization
# -----------------------------
# If not None, used to seed Python's random module for reproducible sampling.
RANDOM_SEED = None

# How many images to sample from IMAGE_DIR on each run.
NUM_SAMPLES = 200

# How many CPU threads PyTorch should use.
NUM_THREADS = 8

# -----------------------------
# Sampling strategy for which images to run
# -----------------------------
# How to sample images from the pool.
#   "random"            -> uniform random over all valid images
#   "balanced_by_label" -> infer an image-level label (majority class_id over
#                          bounding_boxes) and try to balance the sample across
#                          these labels.
SAMPLING_MODE = "balanced_by_label"  # or "random"

# -----------------------------
# Model / classifier settings
# -----------------------------

# Number of classes in your fine-tuned classifier.
# For {car, truck, pedestrian, bicyclist, light} -> 5 (adjust as needed)
NUM_CLASSES = 5

# Paths to fine-tuned checkpoints produced by your finetuning scripts.
SMALL_CKPT_PATH = "ckpts_small/best_mobilenet_small_finetuned.pth"
LARGE_CKPT_PATH = "ckpts_large/best_mobilenet_large_finetuned.pth"

# Image size for MobileNetV3-Small (used in its torchvision transforms).
IMG_SIZE = 224

# How many top labels to print for each image.
TOPK = 5
