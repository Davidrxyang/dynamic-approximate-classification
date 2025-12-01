# dynamic-approximate-classification
dynamic vision object classification framework using approximate computing concepts. 

## training data

training data is split into train and val datasets. each dataset includes a metadata.json file, which includes two fields for each label: "bounding_boxes" which is an array of all the bounding boxes for the image and its associated label, and "distance" which is a ratio of (the size of the largest bounding box)/(size of the image)

class_id labels: 'car': 1, 'truck': 2, 'pedestrian': 3, 'bicyclist': 4, 'light': 5

## setup

make sure you are using a python venv

```
python3 -m venv venv
venv/bin/activate
```

you may have to install a bunch of things in the venv 

## execution 

```
python mobilenetv3_test.py --image PATH --num-threads 8
```

switch out PATH for path to image data, for example:

```
python mobilenetv3_test.py --image data/car.jpg --num-threads 8
```
