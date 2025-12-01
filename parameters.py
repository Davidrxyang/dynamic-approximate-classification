"""
parameters.py

Configuration for the Dynamic Approximate Classification (DAC) demo.

All knobs live here so you don't have to touch DAC.py.
"""

# Path to the directory containing your images.
# These filenames must correspond to keys in METADATA_PATH.
# Example: if metadata has "0001.png", "0002.png", then those files
# should exist as IMAGE_DIR / "0001.png", etc.
IMAGE_DIR = "train"

# Path to metadata JSON file with distance labels.
# Format is assumed to be a dict mapping filename -> {..., "distance": float, ...}
METADATA_PATH = "train/metadata.json"

# Threshold on distance for model selection.
# Larger distance => object is closer / more urgent.
# Rule in DAC.py:
#   if distance >= DISTANCE_THRESHOLD -> use SMALL model
#   else                              -> use LARGE model
DISTANCE_THRESHOLD = 0.05

# Number of images to randomly sample per run.
# If this is larger than the number of available images, it will be clipped.
NUM_SAMPLES = 20

# Random seed for sampling images. Set to None for pure randomness.
RANDOM_SEED = 42

# Torch threading configuration.
NUM_THREADS = 8

# Image size for MobileNetV3-Small (used in its torchvision transforms).
IMG_SIZE = 224

# How many top labels to print for each image.
TOPK = 5
