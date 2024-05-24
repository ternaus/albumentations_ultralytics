# Albumentations + Ultralytics

Example of how to use Albumentations with Ultralytics.

## Dataset

### COCO8

First 8 images of the dataset, used for debugging

```bash
wget https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip
```

### COCO

To download COCO dataset

```bash
aria2c -x 16 http://images.cocodataset.org/annotations/annotations_trainval2017.zip
aria2c -x 16 http://images.cocodataset.org/zips/train2017.zip
aria2c -x 16 http://images.cocodataset.org/zips/val2017.zip
```

## Training the Model

To train the model, you need to run

```bash
python -m src.trainer --mode train
```

## Validating the Model

```bash
python -m src.trainer --mode val
```
