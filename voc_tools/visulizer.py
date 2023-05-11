from voc_tools.utils import JPEG


def from_jpeg(image_path: str):
    return JPEG(image_path)


def see_jpeg(image_path: str):
    JPEG(image_path).see()
