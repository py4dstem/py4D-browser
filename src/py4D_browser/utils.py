import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import QFrame, QPushButton, QApplication, QLabel
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QSpinBox

from typing import NotRequired, TypedDict
from enum import Enum


class DetectorShape(Enum):
    RECTANGULAR = "rectangular"
    POINT = "point"
    CIRCLE = "circle"
    ANNULUS = "annulus"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.replace("&", "").lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


class DetectorMode(Enum):
    INTEGRATING = "integrating"
    MAXIMUM = "maximum"
    CoM = "com"
    CoMx = "comx"
    CoMy = "comy"
    ICOM = "icom"

    # Strip GUI-related cruft from strings to map to internal representations
    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.replace("&", "").replace(" ", "").lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


RectangleGeometry = TypedDict(
    "RectangleGeometry",
    {
        "xmin": float,
        "xmax": float,
        "ymin": float,
        "ymax": float,
    },
)
CircleGeometry = TypedDict(
    "CircleGeometry",
    {
        "x": float,
        "y": float,
        "R": float,
    },
)
AnnulusGeometry = TypedDict(
    "AnnulusGeometry",
    {
        "x": float,
        "y": float,
        "R_inner": float,
        "R_outer": float,
    },
)
PointGeometry = TypedDict(
    "PointGeometry",
    {
        "x": float,
        "y": float,
    },
)

DetectorInfo = TypedDict(
    "DetectorInfo",
    {
        "shape": DetectorShape,
        "mode": DetectorMode,
        # Geometry is intended for display purposes only
        "geometry": RectangleGeometry
        | CircleGeometry
        | AnnulusGeometry
        | PointGeometry,
        #  The are provided based on the detector shape, and should
        #  be used for any image computation:
        "slice": NotRequired[list[slice]],
        "mask": NotRequired[np.ndarray],
        "point": NotRequired[list[int]],
    },
)


class StatusBarWriter:
    def __init__(self, statusBar):
        self.statusBar = statusBar
        self.app = app = QApplication.instance()

    def write(self, message):
        self.statusBar.showMessage(message, 1_000)
        self.app.processEvents()

    def flush(self):
        pass


class VLine(QFrame):
    # a simple vertical divider line
    def __init__(self):
        super(VLine, self).__init__()
        self.setFrameShape(self.VLine | self.Sunken)


class LatchingButton(QPushButton):
    """
    Subclass of QPushButton that acts as a momentary button,
    unless shift is held down during the click, in which case
    it toggles on.
    Emits the "activated" signal whenever triggered, and
    maintains the "latched" state when latched down.
    """

    activated = pyqtSignal()

    def __init__(self, *args, **kwargs):
        self.status_bar = kwargs.pop("status_bar", None)
        self.latched = kwargs.pop("latched", False)
        super().__init__(*args, **kwargs)
        self.setCheckable(True)
        self.clicked.connect(self.on_click)
        if self.latched:
            self.setChecked(True)
            self.activated.emit()

    def on_click(self, *args):
        modifiers = QApplication.keyboardModifiers()

        if self.latched:
            self.setChecked(False)
            self.latched = False
        else:
            if modifiers == Qt.ShiftModifier:
                self.setChecked(True)
                self.latched = True
                self.activated.emit()
            else:
                self.setChecked(False)
                self.latched = False
                self.activated.emit()
                if self.status_bar is not None:
                    self.status_bar.showMessage("Shift+click to keep on", 5_000)


def pg_point_roi(view_box, center=(-0.5, -0.5), pen=(0, 9), hoverPen=None):
    """
    Point selection.  Based in pyqtgraph, and returns a pyqtgraph CircleROI object.
    This object has a sigRegionChanged.connect() signal method to connect to other functions.
    """
    circ_roi = pg.CircleROI(center, (2, 2), movable=True, pen=pen, hoverPen=hoverPen)
    h = circ_roi.addTranslateHandle((0.5, 0.5))
    h.pen = pg.mkPen("r")
    h.update()
    view_box.addItem(circ_roi)
    circ_roi.removeHandle(0)
    return circ_roi


def make_detector(shape: tuple, mode: str, geometry) -> np.ndarray:
    match mode, geometry:
        case ["point", (qx, qy)]:
            mask = np.zeros(shape, dtype=np.bool_)
            mask[qx, qy] = True
        case ["point", geom]:
            raise ValueError(
                f"Point detector shape must be specified as (qx,qy), not {geom}"
            )

        case [("circle" | "circular"), ((qx, qy), r)]:
            ix, iy = np.indices(shape)
            mask = np.hypot(ix - qx, iy - qy) <= r
        case [("circle" | "circular"), geom]:
            raise ValueError(
                f"Circular detector shape must be specified as ((qx,qy),r), not {geom}"
            )

        case [("annulus" | "annular"), ((qx, qy), (ri, ro))]:
            ix, iy = np.indices(shape)
            ir = np.hypot(ix - qx, iy - qy)
            mask = np.logical_and(ir >= ri, ir <= ro)
        case [("annulus" | "annular"), geom]:
            raise ValueError(
                f"Annular detector shape must be specified as ((qx,qy),(ri,ro)), not {geom}"
            )

        case [("rectangle" | "square" | "rectangular"), (xmin, xmax, ymin, ymax)]:
            mask = np.zeros(shape, dtype=np.bool_)
            mask[xmin:xmax, ymin:ymax] = True
        case [("rectangle" | "square" | "rectangular"), geom]:
            raise ValueError(
                f"Rectangular detector shape must be specified as (xmin,xmax,ymin,ymax), not {geom}"
            )

        case ["mask", mask_arr]:
            mask = mask_arr

        case unknown:
            raise ValueError(f"mode and geometry not understood: {unknown}")

    return mask


def complex_to_Lab(
    im, amin=None, amax=None, gamma=1.0, L_scale=100, ab_scale=64, uniform_L=None
):
    from skimage.color import lab2rgb
    from matplotlib.colors import Normalize
    import warnings

    Lab = np.zeros(im.shape + (3,), dtype=np.float64)
    angle = np.angle(im)

    L = Normalize(vmin=amin, vmax=amax, clip=True)(np.abs(im)) ** gamma
    L = Normalize()(L)

    # attempt at polynomial saturation
    # ab_prescale = 4*L - 4*L*L
    ab_prescale = 0.5

    Lab[..., 0] = uniform_L or L * L_scale
    Lab[..., 1] = np.cos(angle) * ab_scale * ab_prescale
    Lab[..., 2] = np.sin(angle) * ab_scale * ab_prescale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rgb = lab2rgb(Lab)

    return rgb
