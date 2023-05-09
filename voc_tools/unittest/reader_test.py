import pathlib
import unittest

from voc_tools import reader as voc_reader

from voc_tools.annotation import Annotation, VOCDataset


class MyTestCase(unittest.TestCase):
    def test_something(self):
        dataset_path = pathlib.Path(r"sixray_data/train")

        self.assertEqual(list(voc_reader.list_dir(str(dataset_path), fullpath=False)),
                         ["P00002.xml", "P00003.xml", "P00004.xml"], msg="XML files not matched.")
        self.assertEqual(list(voc_reader.list_dir(str(dataset_path), fullpath=False, images=True)),
                         ["P00002.jpg", "P00003.jpg", "P00004.jpg"], msg="JPEG files not matched.")
        self.assertEqual(Annotation.csv_header(), "file,xmin,ymin,xmax,ymax,center_x,center_y,class_name",
                         msg="CSV header mismatch")

        for anno in voc_reader.from_file(r"sixray_data\train\JPEGImages\P00002.jpeg"):
            print(anno.csv())
        for anno in voc_reader.from_file(r"sixray_data\train\Annotations\P00002.xml"):
            print(anno.csv())
        for anno in voc_reader.from_dir(str(dataset_path)):
            print(anno.csv())

        VOCDataset(str(dataset_path)).to_csv(str(dataset_path / "train.csv"))


if __name__ == '__main__':
    unittest.main()
