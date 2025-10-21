# dynamic-approximate-classification
dynamic vision object classification framework using approximate computing concepts. 

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
