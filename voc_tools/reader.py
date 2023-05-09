import os
import pathlib
import xml.etree.ElementTree as ET
from voc_tools.annotation import Annotation


def from_xml(xml_file: str):
    """
    Generate a list of Annotation objects from a given VOC XML file
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    filename = root.find('filename').text
    for boxes in root.iter('object'):
        class_ = boxes.find("name").text
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        single_annotation = Annotation(filename, xmin, ymin, xmax, ymax, cx, cy, class_)
        yield single_annotation


def from_directory(dir_path: str):
    """
    Generate a list of Annotation object per file form a given directory
    """
    dir_path = pathlib.Path(dir_path)
    annotations_dir = dir_path / "Annotations"
    for xml_file in os.listdir(str(annotations_dir)):
        for annotation in from_xml(str(annotations_dir / xml_file)):
            yield annotation
