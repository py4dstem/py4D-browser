import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import QFrame, QPushButton, QApplication, QLabel
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QSpinBox


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


class ResizeDialog(QDialog):
    def __init__(self, size, parent=None):
        super().__init__(parent=parent)

        self.new_size = size
        Nmax = size[0] * size[1]

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Dataset size unknown. Please enter the shape:"))

        box_layout = QHBoxLayout()
        box_layout.addWidget(QLabel("X:"))

        xbox = QSpinBox()
        xbox.setRange(1, Nmax)
        xbox.setSingleStep(1)
        xbox.setKeyboardTracking(False)
        xbox.valueChanged.connect(self.x_box_changed)
        box_layout.addWidget(xbox)

        box_layout.addStretch()
        box_layout.addWidget(QLabel("Y:"))

        ybox = QSpinBox()
        ybox.setRange(1, Nmax)
        ybox.setSingleStep(1)
        ybox.setValue(Nmax)
        ybox.setKeyboardTracking(False)
        ybox.valueChanged.connect(self.y_box_changed)
        box_layout.addWidget(ybox)

        layout.addLayout(box_layout)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        done_button = QPushButton("Done")
        done_button.pressed.connect(self.close)
        button_layout.addWidget(done_button)
        layout.addLayout(button_layout)

        self.x_box = xbox
        self.y_box = ybox
        self.x_box_last = xbox.value()
        self.y_box_last = ybox.value()
        self.N = Nmax

        self.resize(600, 400)

    @classmethod
    def get_new_size(cls, size, parent=None):
        dialog = cls(size=size, parent=parent)
        dialog.exec_()
        return dialog.new_size

    def x_box_changed(self, new_value):
        if new_value == self.x_box_last:
            return
        x_new, y_new = self.get_next_rect(
            new_value, "down" if new_value < self.x_box_last else "up"
        )

        self.x_box_last = x_new
        self.y_box_last = y_new

        self.x_box.setValue(x_new)
        self.y_box.setValue(y_new)

        self.new_size = [x_new, y_new]

    def y_box_changed(self, new_value):
        if new_value == self.y_box_last:
            return
        y_new, x_new = self.get_next_rect(
            new_value, "down" if new_value < self.y_box_last else "up"
        )

        self.x_box_last = x_new
        self.y_box_last = y_new

        self.x_box.setValue(x_new)
        self.y_box.setValue(y_new)

        self.new_size = [x_new, y_new]

    def get_next_rect(self, current, direction):
        # get the next perfect rectangle
        iterator = (
            range(current, 0, -1) if direction == "down" else range(current, self.N + 1)
        )

        for i in iterator:
            if self.N % i == 0:
                return i, self.N // i

        raise ValueError("Factor finding failed, frustratingly.")


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


def complex_to_Lab(
    im, amin=None, amax=None, gamma=1, L_scale=100, ab_scale=64, uniform_L=None
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
