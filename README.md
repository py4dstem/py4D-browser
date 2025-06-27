
# The `py4DSTEM` GUI

This repository hosts the `pyqt` based graphical 4D--STEM data browser that was originally part of **py4DSTEM** until version 0.13.11.

## Installation
The GUI is available on PyPI and conda-forge:

`pip install py4D-browser`

`conda install -c conda-forge py4d-browser`


## Usage
Run `py4DGUI` in your terminal to open the GUI. Then just drag and drop a 4D-STEM dataset into the window!

### Controls
* Move the virtual detector and the real-space selector using the mouse or using the keyboard shortcuts: WASD moves the detector and IJKL moves the selector, and holding down shift moves 5 pixels at a time.
* Auto scaling of both views is on by default. Press the "Autoscale" buttons in the bottom right to disable. Press either button to apply automatic scaling once, or Shift + click to lock autoscaling back on.
* Different shapes of virtual detector are available in the "Detector Shape" menu, and different detector responses are available in the "Detector Response" menu.
* The information in the bottom bar contains the details of the virtual detector used to generate the images, and can be entered into py4DSTEM to generate the same image.
* The FFT pane can be switched between displaying the FFT of the virtual image and displaying the [exit wave power cepstrum](https://doi.org/10.1016/j.ultramic.2020.112994).
* Virtual images can be exported either as the scaled and clipped displays shown in the GUI or as raw data. The exact datatype stored in the raw TIFF image depends on both the datatype of the dataset and the type of virtual image being displayed (in particular, integer datatypes are converted internally to floating point to prevent overflows when generating any synthesized virtual images).

![Demonstration](/images/demo.gif)

The keyboard map in the Help menu was made using [this tool](https://archie-adams.github.io/keyboard-shortcut-map-maker/) and the map file is in the top level of this repo.

## Plugins

As of version 1.3.0, we now support a simple means for loading plugins that extend the functionality of the browser. Details on creating a plugin can be found in [this document](PLUGINS.md).

The [EMPAD-G2 Raw Reader](https://github.com/sezelt/empad2), which was previously implemented in the browser code itself, is now implemented as a plugin, which can serve as an example.

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
