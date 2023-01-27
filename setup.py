from setuptools import setup, find_packages
from distutils.util import convert_path

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="py4D_browser",
    version="0.9",
    packages=find_packages(),
    description="A 4D-STEM data browser built on py4DSTEM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/py4dstem/py4DSTEM/",
    author="Steven E Zeltmann",
    author_email="steven.zeltmann@lbl.gov",
    license="GNU GPLv3",
    keywords="STEM 4DSTEM",
    python_requires=">=3.8",
    install_requires=[
        "py4dstem >= 0.13.11",
        "numpy >= 1.19",
        "matplotlib >= 3.2.2",
        "PyQt5 >= 5.10",
        "pyqtgraph >= 0.11",
    ],
    entry_points={"console_scripts": ["py4DGUI=py4D_browser.runGUI:launch"]},
)
