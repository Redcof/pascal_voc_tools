import os
import pathlib
import xml.etree.ElementTree as ET
from voc_tools.annotation import Annotation


def from_file(file: str):
    """
    Generate a list of Annotation objects for a given image or xml of a PASCAL VOC dataset
    """
    if file.endswith(".xml"):
        return from_xml(file)
    else:
        return from_image(file)


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


def list_dir(dir_path: str, images=False, fullpath=True):
    """
    Generate a list of XML files form a given PASCAL VOC directory
    """
    dir_path = pathlib.Path(dir_path)
    annotations_dir = dir_path / ("JPEGImages" if images else "Annotations")
    for xml_file in os.listdir(str(annotations_dir)):
        if fullpath:
            yield str(annotations_dir / xml_file)
        else:
            yield (annotations_dir / xml_file).name


def from_dir(dir_path: str):
    """
    Generate a list of Annotation object per file form a given PASCAL VOC directory
    """
    for xml_file in list_dir(dir_path):
        for annotation in from_xml(xml_file):
            yield annotation
