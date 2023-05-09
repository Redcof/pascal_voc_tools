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
        return "file:{},xmin:{},ymin:{},xmax:{},ymax:{},center_x:{},center_y:{},class_name:{}".format(
            self._filename, self._xmin, self._ymin, self._xmax, self._ymax, self._center_x, self._center_y,
            self._class_name
        )

    @classmethod
    def csv_header(cls):
        return "file,xmin,ymin,xmax,ymax,center_x,center_y,class_name"

    def csv(self):
        return "{},{},{},{},{},{},{},{}".format(
            self._filename, self._xmin, self._ymin, self._xmax, self._ymax, self._center_x, self._center_y,
            self._class_name
        )


class VOCDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def to_csv(self, path_to_csv, write_mode="w"):
        """
        Generate csv file for given VOC dataset
        """
        from voc_tools.reader import from_dir
        with open(path_to_csv, write_mode) as csv_fp:
            csv_fp.write("{}\n".format(Annotation.csv_header()))
            for anno in from_dir(self.dataset_path):
                csv_fp.write("{}\n".format(anno.csv()))
