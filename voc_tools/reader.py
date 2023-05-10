import os
import pathlib
import xml.etree.ElementTree as ET

from voc_tools.utils import Annotation, Caption


def from_file(file: str):
    """
    Generate a list of Annotation objects for a given image or xml of a PASCAL VOC dataset.
    It also supports captions .txt files
    """
    if file.endswith(".xml"):
        return from_xml(file)
    elif file.endswith(".txt"):
        return caption_from_file(file)
    elif file.endswith(".jpeg"):
        return from_image(file)
    else:
        raise ValueError("Unsupported file format.")


def from_image(image_file: str):
    """
    Generate a list of Annotation objects for a given image of a PASCAL VOC dataset
    """
    image_file = pathlib.Path(image_file)
    parent_path = image_file.parents[1] / "Annotations"
    file_name = image_file.name.replace(".jpeg", ".xml")
    xml_file = str(parent_path / file_name)
    return from_xml(xml_file)


def from_xml(xml_file: str, empty_placeholder="NULL"):
    """
    Generate a list of Annotation objects from a given VOC XML file
    """
    filename = pathlib.Path(xml_file).name
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        no_threat = True

        filename = root.find('filename').text
        for boxes in root.iter('object'):
            no_threat = False
            class_ = boxes.find("name").text
            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2

            single_annotation = Annotation(filename, xmin, ymin, xmax, ymax, cx, cy, class_)
            yield single_annotation
        if no_threat:
            yield Annotation(filename, 0, 0, 0, 0, 0, 0, empty_placeholder)
    except Exception as e:
        yield Annotation(filename, 0, 0, 0, 0, 0, 0, "ERROR:{}".format(e))


VOC_IMAGES = 1
VOC_XMLS = 2
VOC_CAPTIONS = 3


def list_dir(dir_path: str, dir_flag=VOC_XMLS, images=False, fullpath=True):
    """
    Generate a list of XML files form a given PASCAL VOC directory
    Args:
        dir_path: A path to VOC dataset
        dir_flag: A flag which specify the subdirectory to list. Any typical
                    Pascal VOC dataset must contain Annotations, JPEGImages and text[optionsl] directories.
                    VOC_IMAGES - JPEGImages
                    VOC_XMLS - Annotations
                    VOC_CAPTIONS - text
        images: [depreciated] use this flag to load JPEGImages directory. Use dir_flag=VOC_IMAGES instead.
        fullpath: A boolean flag indicates whether to generate full path or the filename.
    """
    dir_path = pathlib.Path(dir_path)
    if images:
        dir_flag = VOC_IMAGES
    annotations_dir = dir_path / (
        "JPEGImages" if dir_flag == VOC_IMAGES else ("Annotations" if dir_flag == VOC_XMLS else "text"))
    for file_item in os.listdir(str(annotations_dir)):
        if fullpath:
            yield str(annotations_dir / file_item)
        else:
            yield (annotations_dir / file_item).name


def from_dir(dir_path: str):
    """
    Generate a list of Annotation object per file form a given PASCAL VOC directory
    """
    for xml_file in list_dir(dir_path):
        for annotation in from_xml(xml_file):
            yield annotation


def caption_from_file(file_path: str, empty_placeholder="NULL"):
    """
    Generate a list of captions object per file form a given text file path directory
    """
    filename = pathlib.Path(file_path).name
    try:
        with open(file_path, "r") as fp:
            no_caption = True
            for caption in map(lambda x: Caption(filename, x.strip()),
                               filter(lambda l: l.strip() != "", fp.readlines())):
                no_caption = False
                yield caption
            if no_caption:
                yield Caption(filename, empty_placeholder)
    except Exception as e:
        yield Caption(filename, "ERROR:{}".format(e))


def caption_from_dir(dir_path: str):
    """
    Generate a list of captions object per file form a given directory
    """
    for file in list_dir(dir_path, dir_flag=VOC_CAPTIONS):
        for caption in caption_from_file(file):
            yield caption
