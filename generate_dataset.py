import os
import random
import shutil
import pandas as pd
import glob
import json
import csv

labels = pd.read_csv("/Users/henrydeng/Desktop/Georgetown/2025-26/COSC_5521/dynamic-approximate-classification/archive/labels_trainval.csv")
directory_path = "/Users/henrydeng/Desktop/Georgetown/2025-26/COSC_5521/dynamic-approximate-classification/archive/images"
files_only = []
img_size = 480*300

for entry in os.listdir(directory_path):
    full_path = os.path.join(directory_path, entry)
    if os.path.isfile(full_path):
        files_only.append(entry)

metadata = {}
os.makedirs("/Users/henrydeng/Desktop/Georgetown/2025-26/COSC_5521/dynamic-approximate-classification/train")
train_csv = [['filename','label']]
for i in range(998):
    index = random.randint(0, len(files_only) - 1)
    metadata_indices = [idx for idx, value in enumerate(labels['frame']) if value == files_only[index]]

    metadata_array = []
    max_size = 0
    label = 0
    for k in metadata_indices:
        size = (labels['xmax'][k].item() - labels['xmin'][k].item()) * (labels['ymax'][k].item() - labels['ymin'][k].item())
        if size >= max_size:
            max_size = size
            label = labels['class_id'][k].item()
        metadata_array.append({
            'xmin': labels['xmin'][k].item(),
            'xmax': labels['xmax'][k].item(),
            'ymin': labels['ymin'][k].item(),
            'ymax': labels['ymax'][k].item(),
            'class_id': labels['class_id'][k].item()
        })
    distance = max_size / img_size
    metadata[files_only[index]] = {
        'bounding_boxes': metadata_array,
        'distance': distance
    }
    train_csv.append([files_only[index], label])

    shutil.copy(f"{directory_path}/{files_only[index]}", "/Users/henrydeng/Desktop/Georgetown/2025-26/COSC_5521/dynamic-approximate-classification/train")
    files_only.pop(index)

with open("/Users/henrydeng/Desktop/Georgetown/2025-26/COSC_5521/dynamic-approximate-classification/train/metadata.json", 'w') as json_file:
    json.dump(metadata, json_file, indent=4)
with open('/Users/henrydeng/Desktop/Georgetown/2025-26/COSC_5521/dynamic-approximate-classification/train/train_labels.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(train_csv)

metadata = {}
val_csv = [['filename','label']]
os.makedirs("/Users/henrydeng/Desktop/Georgetown/2025-26/COSC_5521/dynamic-approximate-classification/val")
for i in range(200):
    index = random.randint(0, len(files_only) - 1)
    metadata_indices = [idx for idx, value in enumerate(labels['frame']) if value == files_only[index]]

    metadata_array = []
    max_size = 0
    label = 0
    for k in metadata_indices:
        size = (labels['xmax'][k].item() - labels['xmin'][k].item()) * (labels['ymax'][k].item() - labels['ymin'][k].item())
        if size > max_size:
            max_size = size
            label = labels['class_id'][k].item()
        metadata_array.append({
            'xmin': labels['xmin'][k].item(),
            'xmax': labels['xmax'][k].item(),
            'ymin': labels['ymin'][k].item(),
            'ymax': labels['ymax'][k].item(),
            'class_id': labels['class_id'][k].item()
        })
    distance = max_size / img_size
    metadata[files_only[index]] = {
        'bounding_boxes': metadata_array,
        'distance': distance
    }
    val_csv.append([files_only[index], label])

    shutil.copy(f"{directory_path}/{files_only[index]}", "/Users/henrydeng/Desktop/Georgetown/2025-26/COSC_5521/dynamic-approximate-classification/val")
    files_only.pop(index)

with open("/Users/henrydeng/Desktop/Georgetown/2025-26/COSC_5521/dynamic-approximate-classification/val/metadata.json", 'w') as json_file:
    json.dump(metadata, json_file, indent=4)
with open('/Users/henrydeng/Desktop/Georgetown/2025-26/COSC_5521/dynamic-approximate-classification/val/val_labels.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(val_csv)