import pathlib
import unittest

from voc_tools import reader as voc_reader
from voc_tools.utils import Annotation, VOCDataset


class MyTestCase(unittest.TestCase):
    def test_something(self):
        dataset_path = pathlib.Path(r"sixray_data")

        self.assertEqual(list(voc_reader.list_dir(str(dataset_path / "train"), fullpath=False)),
                         ["P00002.xml", "P00003.xml", "P00004.xml"], msg="XML files not matched.")
        self.assertEqual(list(voc_reader.list_dir(str(dataset_path / "train"), fullpath=False, images=True)),
                         ["P00002.jpg", "P00003.jpg", "P00004.jpg"], msg="JPEG files not matched.")
        self.assertEqual(Annotation.csv_header(), "file,xmin,ymin,xmax,ymax,center_x,center_y,class_name",
                         msg="CSV header mismatch")

        ["file:P00002.jpg,xmin:274,ymin:232,xmax:610,ymax:460,center_x:442.0,center_y:346.0,class_name:knife",
         "file:P00002.jpg,xmin:225,ymin:334,xmax:591,ymax:465,center_x:408.0,center_y:399.5,class_name:knife", ]
        ["P00002.jpg,274,232,610,460,442.0,346.0,knife",
         "P00002.jpg,225,334,591,465,408.0,399.5,knife",
         "P00002.jpg,274,232,610,460,442.0,346.0,knife",
         "P00002.jpg,225,334,591,465,408.0,399.5,knife", ]
        "P00003.jpg,858,121,918,321,888.0,221.0,knife",
        "P00004.jpg,83,73,133,425,108.0,249.0,knife",
        "P00004.jpg,1,162,343,318,172.0,240.0,knife",

        for anno in voc_reader.from_file(r"sixray_data\train\JPEGImages\P00002.jpeg"):
            print(str(anno))
        for anno in voc_reader.from_file(r"sixray_data\train\Annotations\P00002.xml"):
            print(anno.csv())
        for anno in voc_reader.from_dir(str(dataset_path / "train")):
            print(anno.csv())
        self.assertEqual("file,xmin,ymin,xmax,ymax,center_x,center_y,class_name", Annotation.csv_header())

        print(VOCDataset(str(dataset_path)).train.load().to_csv(str(dataset_path / "train.csv")).class_names())

        voc_caption_data = VOCDataset(str(dataset_path), caption_support=True)
        for caption in voc_caption_data.train.caption.fetch():
            print(caption)

        for anno in voc_caption_data.train.fetch():
            print(anno)

        voc_caption_data.train.caption.to_csv(str(dataset_path / "captions.csv"))


if __name__ == '__main__':
    unittest.main()
