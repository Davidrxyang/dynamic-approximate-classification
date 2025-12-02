# dynamic-approximate-classification

dynamic vision object classification framework using approximate computing concepts. This framework is designed for self-driving vehicle decision-making processes. 

## overview

The DAC framework consists of three main components:

### 1. Vision Model Selector

When an object appears in front of the agent, a distance reading is taken immediately. This distance reading is fed into a simple threshold model. Based on how close the object is, the threshold model determines how urgently a decision must be made for the self-driving vehicle in regards to the object. The model accordingly selects a computer vision network of appropriate size, speed, and output granularity based on how fast a decision is required considering the urgency of the situation. 

### 2. Computer Vision Model Set

The selected vision model takes as input a visual snapshot of the scene and outputs a set of labels for objects in the scene as well as a confidence distribution over the predicted labels. Intuitively, smaller and faster models will output lower confidence values for particular labels. 

### 3. Decision Model

The label set and confidence distribution is passed into a decision model. Based on the nature of the recognized objects, the confidence of the vision model, and the urgency of the scenario, the decision model outputs an action for the self-driving vehicle considering all the listed factors. For example, the vehicle may continue in its current course if the system recognizes a paper bag regardless of the confidence; on the other hand, the vehicle should brake if the system recognizes a human in front of the vehicle, with the degree of braking determined by the urgency of the situation (distance). 

## training data

training data is split into train and val datasets. each dataset includes a metadata.json file, which includes two fields for each label: "bounding_boxes" which is an array of all the bounding boxes for the image and its associated label, and "distance" which is a ratio of (the size of the largest bounding box)/(size of the image)

class_id labels: 'car': 1, 'truck': 2, 'pedestrian': 3, 'bicyclist': 4, 'light': 5

## finetuning

to finetune mobilenetv3_small:
python3 finetune_mobilenet.py \
  --model small \
  --train-images-root train \
  --val-images-root val \
  --train-csv train/train_labels.csv \
  --val-csv val/val_labels.csv \
  --batch-size 64 \
  --epochs 5 \
  --lr 1e-4 \
  --output-dir ckpts_small \
  --device cpu

to finetune mobilenetv3_large:
python3 finetune_mobilenet.py \
  --model large \
  --train-images-root train \
  --val-images-root val \
  --train-csv train/train_labels.csv \
  --val-csv val/val_labels.csv \
  --batch-size 64 \
  --epochs 5 \
  --lr 1e-4 \
  --output-dir ckpts_large \
  --device cpu

## setup

make sure you are using a python venv

```
python3 -m venv venv
venv/bin/activate
```

you may have to install a bunch of things in the venv 

## execution 

```
python3 DAC.py
```

configure parameters in the parameters.py file
