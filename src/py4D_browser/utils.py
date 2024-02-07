import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import QFrame, QPushButton, QApplication
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt


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


def pg_point_roi(view_box):
    """
    Point selection.  Based in pyqtgraph, and returns a pyqtgraph CircleROI object.
    This object has a sigRegionChanged.connect() signal method to connect to other functions.
    """
    circ_roi = pg.CircleROI((-0.5, -0.5), (2, 2), movable=True, pen=(0, 9))
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
