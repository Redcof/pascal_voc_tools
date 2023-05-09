import pathlib
from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='pascal-voc-tools',
    description="A tool for creating, reading and visualizing Pascal VOC annotations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Soumen Sardar",
    author_email="soumensardarintmain@gmail.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Proprietary Internal Software",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['lxml'],
    version_config=True,
    # setup_requires=["setuptools-git-versioning"]
)