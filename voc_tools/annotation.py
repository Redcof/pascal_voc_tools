import os

import numpy as np


class Annotation:
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

    def __str__(self):
        s = ""
        for k, v in zip(self.raw_attributes(), self.raw()):
            s = "{}{}:{},".format(s, k, v)
        return s.strip(",")

    @staticmethod
    def csv_header():
        return ",".join(Annotation.raw_attributes())

    def csv(self):
        return "{},{},{},{},{},{},{},{}".format(*self.raw())

    @staticmethod
    def raw_attributes():
        return "file", "xmin", "ymin", "xmax", "ymax", "center_x", "center_y", "class_name"

    def raw(self):
        return (self._filename, self._xmin, self._ymin, self._xmax, self._ymax, self._center_x, self._center_y,
                self._class_name
                )


class ABCDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.meta = np.array([], dtype='object')

    def load(self):
        from voc_tools.reader import from_dir
        self.meta = np.array([anno.raw() for anno in from_dir(self.dataset_path)], dtype='object')
        return self

    def unload(self):
        del self.meta
        self.meta = np.array([], dtype='object')
        return self

    def class_names(self):
        class_name_idx = Annotation.raw_attributes().index('class_name')
        return set(self.meta[:, class_name_idx])

    def to_csv(self, path_to_csv, write_mode="w"):
        """
        Generate csv file for given VOC dataset
        """
        from voc_tools.reader import from_dir
        with open(path_to_csv, write_mode) as csv_fp:
            csv_fp.write("{}\n".format(Annotation.csv_header()))
            for anno in from_dir(self.dataset_path):
                csv_fp.write("{}\n".format(anno.csv()))
        return self


class VOCDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.train = ABCDataset(os.path.join(self.dataset_path, "train"))
        self.test = ABCDataset(os.path.join(self.dataset_path, "test"))
