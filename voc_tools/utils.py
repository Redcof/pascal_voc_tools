import os
import pathlib
from abc import ABC, abstractmethod

import cv2
import numpy as np

from voc_tools.constants import VOC_IMAGES, VOC_CAPTIONS, VOC_XMLS


def path_convert(file: str, converter_flag: int):
    """
    Convert any file path to a required file path
    :param file: A path to xml, jpeg or txt file
    :param converter_flag:
    :return: Converted path
    """
    assert file.endswith(".xml") or file.endswith(".txt") or file.endswith(".jpeg") or file.endswith(".jpg"), (
        "Unsupported file format")
    assert converter_flag in [VOC_IMAGES, VOC_CAPTIONS, VOC_XMLS], "Unsupported converter_flag"
    file = pathlib.Path(file)
    filename = file.name
    parent_path = file.parents[1]
    if converter_flag == VOC_IMAGES:
        path = parent_path / Dataset.IMAGE_DIR / filename
    if converter_flag == VOC_XMLS:
        path = parent_path / Dataset.ANNO_DIR / filename
    if converter_flag == VOC_CAPTIONS:
        path = parent_path / Dataset.CAPTION_DIR / filename
    return str(path)


# ################################################
# Python classes to hold annotation and captions #
# ################################################
class Atomic:
    def __str__(self):
        s = ""
        for k, v in zip(self.raw_attributes(), self.raw()):
            s = "{}{}:{},".format(s, k, v)
        return s.strip(",")

    @classmethod
    def csv_header(cls):
        """Return the CSV header. A method instance for a single class is enough"""
        return ",".join(cls.raw_attributes())

    def csv(self):
        """Returns the CSV formatted data as string"""
        return ",".join(["{}"] * len(self.raw())).format(*self.raw())

    @abstractmethod
    def raw(self):
        """Return raw data as tuple"""
        ...

    @classmethod
    @abstractmethod
    def raw_attributes(cls):
        """Return raw tuple of strings(to be used as key per values). A single method instance is required per class"""
        ...


class JPEG(Atomic):

    def raw(self):
        return self._filename, *self._cv_image.shape

    @classmethod
    def raw_attributes(cls):
        return "file", "height", "width", "channels"

    def __init__(self, filename):
        filename = path_convert(filename, VOC_IMAGES)
        self._filename = pathlib.Path(filename).name
        self._cv_image = cv2.imread(filename)

    @property
    def filename(self):
        return self._filename

    @property
    def image(self):
        return self._cv_image

    def see(self):
        cv2.imshow(self.filename, self._cv_image)
        cv2.waitKey(-1)


class Annotation(Atomic):
    def __init__(self, filename, xmin, ymin, xmax, ymax, center_x, center_y, class_name):
        self._xmin, self._ymin, self._xmax, self._ymax, self._center_x, self._center_y, self._class_name = (
            xmin, ymin, xmax,
            ymax, center_x,
            center_y,
            class_name)
        self._filename = filename

    @property
    def filename(self):
        return self._filename

    @property
    def xmin(self):
        return self._xmin

    @property
    def ymin(self):
        return self._ymin

    @property
    def xmax(self):
        return self._xmax

    @property
    def ymax(self):
        return self._ymax

    @property
    def center_x(self):
        return self._center_x

    @property
    def center_y(self):
        return self._center_y

    @property
    def class_name(self):
        return self._class_name

    @classmethod
    def raw_attributes(cls):
        return "file", "xmin", "ymin", "xmax", "ymax", "center_x", "center_y", "class_name"

    def raw(self):
        return (self._filename, self._xmin, self._ymin, self._xmax, self._ymax, self._center_x, self._center_y,
                self._class_name)


class Caption(Atomic):
    def __init__(self, filename, caption):
        self._filename = filename
        self._caption = caption

    @property
    def filename(self):
        return self._filename

    @property
    def captions(self):
        return self._caption

    @classmethod
    def raw_attributes(cls):
        return "file", "caption"

    def raw(self):
        return self._filename, self._caption


# ########################################################
# Python classes to hold annotation and captions dataset #
# ########################################################
class ABCDataset(ABC):
    def __init__(self, dataset_path):
        assert os.path.exists(dataset_path), dataset_path
        self.dataset_path = dataset_path

    @abstractmethod
    def to_csv(self, path_to_csv, write_mode="w"):
        ...


class CaptionDataset(ABCDataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def fetch(self, bulk=True):
        """
        Generate Caption object
        """
        from voc_tools.reader import caption_from_dir

        for caption in caption_from_dir(self.dataset_path, bulk=bulk):
            yield caption

    def to_csv(self, path_to_csv, write_mode="w"):
        """
        Generate csv file for given VOC dataset
        """
        from voc_tools.reader import caption_from_dir
        with open(path_to_csv, write_mode) as csv_fp:
            csv_fp.write("{}\n".format(Caption.csv_header()))
            for caption in caption_from_dir(self.dataset_path, bulk=False):
                csv_fp.write("{}\n".format(caption.csv()))
        return self


class Dataset(ABCDataset):
    IMAGE_DIR = "JPEGImages"
    ANNO_DIR = "Annotations"
    CAPTION_DIR = "captions"

    def __init__(self, dataset_path, caption_support=False):
        super().__init__(dataset_path)
        self.dataset_path = dataset_path
        self.caption_support = caption_support
        self.meta = np.array([], dtype='object')
        self.image_cache: JPEG = None
        self.loaded = False
        if caption_support:
            self.caption = CaptionDataset(dataset_path)

    def load(self):
        if not self.loaded:
            from voc_tools.reader import from_dir
            self.meta = np.array([anno.raw() for anno in from_dir(self.dataset_path, bulk=False)], dtype='object')
            self.loaded = True
        return self

    def unload(self):
        del self.meta
        self.meta = np.array([], dtype='object')
        self.loaded = False
        return self

    def fetch(self, bulk=True):
        from voc_tools.reader import from_dir
        for anno in from_dir(self.dataset_path, bulk=bulk):
            if bulk:
                yield anno, self.get_image(anno[0].filename)
            else:
                yield anno, self.get_image(anno.filename)

    def class_names(self):
        if not self.loaded:
            self.load()
        class_name_idx = Annotation.raw_attributes().index('class_name')
        return tuple(set(self.meta[:, class_name_idx]))

    def get_image(self, filename):
        return JPEG(os.path.join(self.dataset_path, Dataset.IMAGE_DIR, filename))

    def get_image_meta(self, filename):
        if self.image_cache is None or filename != self.image_cache.filename:
            # save the image
            self.image_cache = self.get_image(filename)
            # if a newer appear, re
        return self.image_cache.csv().split(",")[1:]

    def to_csv(self, path_to_csv, write_mode="w"):
        """
        Generate csv file for given VOC dataset
        """
        from voc_tools.reader import from_dir
        with open(path_to_csv, write_mode) as csv_fp:
            csv_fp.write("{}".format(Annotation.csv_header()))
            csv_fp.write("{}\n".format(JPEG.csv_header().replace("file", "")))
            for anno in from_dir(self.dataset_path, bulk=False):
                csv_fp.write("{},".format(anno.csv()))
                csv_fp.write(",".join(self.get_image_meta(anno.filename)))
                csv_fp.write("\n")
        return self


class TrainDataset(Dataset):
    def __init__(self, dataset_path, caption_support=False):
        super().__init__(dataset_path, caption_support=caption_support)


class TestDataset(Dataset):
    def __init__(self, dataset_path, caption_support=False):
        super().__init__(dataset_path, caption_support=caption_support)


# ####################################
# Python classes to hold VOC Dataset #
# ####################################

class VOCDataset:
    def __init__(self, dataset_path, caption_support=False):
        self.dataset_path = dataset_path
        self.caption_support = caption_support
        self.train = TrainDataset(os.path.join(self.dataset_path, "train"), caption_support=caption_support)
        self.test = TestDataset(os.path.join(self.dataset_path, "test"), caption_support=caption_support)
