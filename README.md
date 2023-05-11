# Express Pascal Voc Tools

A tool for creating, reading and visualizing Pascal VOC annotations.
**Report Bugs** [here](https://github.com/Redcof/pascal_voc_tools/issues)

# Getting Started

## Install

`pip install express-pascal-voc-tools`

## Single file Parsing

```python
from voc_tools import reader as voc_reader

# `from_xml()` parse XML
for anno in voc_reader.from_xml(r"sixray_data\train\Annotations\P00002.xml"):
    print(anno.xmin, anno.xmax)

# `from_image()` Parse XML by providing image path(it will automatically choose the correct XML)
for anno in voc_reader.from_image(r"sixray_data\train\JPEGImages\P00002.jpeg"):
    print(anno.xmin, anno.xmax)

# `from_file()` Get the parsed metadata as a tuple
for anno in voc_reader.from_file(r"sixray_data\train\JPEGImages\P00002.xml"):
    print(anno.raw())
for anno in voc_reader.from_file(r"sixray_data\train\JPEGImages\P00002.jpeg"):
    print(anno.raw())

# `from_dir()` Get the parsed metadata as a tuple for entire directory
for anno in voc_reader.from_dir("sixray_data\train")):
    print(anno.raw())

```

### Dataset level parsing

Using `VOCDataset` class we can address a Pascal VOC dataset. In general Pascal VOC
Datasets are organised as below:

```commandline
my_dataset
    |
    +- train
    |   |
    |   +- Annotations
    |   |  |
    |   |  +- ITEM001.xml
    |   |  +- ITEM002.xml
    |   +- JPEGImages
    |       |
    |       +- ITEM001.jpeg
    |       +- ITEM002.jpeg
    +- test
        |
        +- Annotations
        |  |
        |  +- ITEM0010.xml
        |  +- ITEM0020.xml
        +- JPEGImages
            |
            +- ITEM0010.jpeg
            +- ITEM0020.jpeg
```

```python
from voc_tools.utils import VOCDataset

dataset_path = "/my_dataset"

# initialize a dataset
my_dataset = VOCDataset(dataset_path)

# fetch annotation bulk
for annotations, jpeg in my_dataset.train.fetch():
    print(annotations[0].filename, jpeg.image.shape)
# fetch annotation
for anno, jpeg in my_dataset.train.fetch(bulk=False):
    print(anno, jpeg.image.shape)

# parse the annotations into memory for train dataset
my_dataset.train.load()
my_dataset.test.load()

# returns a list of class names in train dataset
my_dataset.train.class_names()
my_dataset.test.class_names()

# save parsed information into csv
my_dataset.train.load().to_csv("./train_metadata.csv")
my_dataset.test.load().to_csv("./train_metadata.csv")

# purge the parsed metadata to free memory
my_dataset.train.unload()
my_dataset.test.unload()
```

## Caption Support

This is an optional feature introduced to facilitate the new trends in prompt engineering and text based Generative AI.
In this case the dataset must contain a `text` directory as below:

```commandline
my_dataset
    |
    +- train
    |   |
    |   +- Annotations
    |   |  |
    |   |  +- ITEM001.xml
    |   |  +- ITEM002.xml
    |   +- JPEGImages
    |       |
    |       +- ITEM001.jpeg
    |       +- ITEM002.jpeg
    |   +- text
    |       |
    |       +- ITEM001.text
    |       +- ITEM002.text
    +- test
        |
        +- Annotations
        |  |
        |  +- ITEM0010.xml
        |  +- ITEM0020.xml
        +- JPEGImages
            |
            +- ITEM0010.jpeg
            +- ITEM0020.jpeg
        +- text
            |
            +- ITEM0010.text
            +- ITEM0020.text
```

```python
from voc_tools.utils import VOCDataset

dataset_path = "/my_dataset"
voc_caption_data = VOCDataset(dataset_path, caption_support=True)  # init dataset with caption

# read caption bulk
for captions in voc_caption_data.train.caption.fetch():
    print(captions[0].raw())

# read caption one by one
for caption in voc_caption_data.train.caption.fetch(bulk=False):
    print(caption.raw())
# save captions to a CSV
voc_caption_data.train.caption.to_csv("train_captions.csv")
```

### Visualize

```python
from voc_tools.visulizer import from_jpeg, see_jpeg

jpeg = from_jpeg(r"sixray_data\train\JPEGImages\P00002.jpg")
jpeg.see()
# OR
see_jpeg(r"sixray_data\train\JPEGImages\P00002.jpg")
```

# Collaborate

GitHub: [https://github.com/Redcof/pascal_voc_tools.git](https://github.com/Redcof/pascal_voc_tools.git)

**Build and Publish**

1. `python setup.py sdist bdist_wheel`
1. `python -m twine upload dist/*`