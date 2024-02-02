
# The `py4DSTEM` GUI

This repository hosts the `pyqt` based graphical 4D--STEM data browser that was originally part of **py4DSTEM** until version 0.13.11. 

## Installation 
The GUI is available on PyPI and conda-forge: 

`pip install py4D-browser`

`conda install -c conda-forge py4d-browser`


## Usage
Run `py4DGUI` in your terminal to open the GUI. Then just drag and drop a 4D-STEM dataset into the window!

Move the virtual detector and the real-space selector using the mouse or using the keyboard shortcuts: WASD moves the detector and IJKL moves the selector, holding down shift to move 5 pixels at a time. 

![Demonstration](/images/demo.gif)

The keyboard map in the Help menu was made using [this tool](https://archie-adams.github.io/keyboard-shortcut-map-maker/) and the map file is in the top level of this repo.

## About 

![py4DSTEM logo](/images/py4DSTEM_logo.png)

**py4DSTEM** is an open source set of python tools for processing and analysis of four-dimensional scanning transmission electron microscopy (4D-STEM) data. Additional information:

- [Our open access py4DSTEM publication in Microscopy and Microanalysis](https://doi.org/10.1017/S1431927621000477) describing this project and demonstrating a variety of applications.
- [The py4DSTEM documentation pages](https://py4dstem.readthedocs.io/en/latest/index.html).
- [Our open access 4D-STEM review in Microscopy and Microanalysis](https://doi.org/10.1017/S1431927619000497) describing this project and demonstrating a variety of applications.


### License

GNU GPLv3

**py4DSTEM** is open source software distributed under a GPLv3 license.
It is free to use, alter, or build on, provided that any work derived from **py4DSTEM** is also kept free and open.
