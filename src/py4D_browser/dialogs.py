from py4DSTEM import DataCube, data
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm
from PyQt5.QtWidgets import QFrame, QPushButton, QApplication, QLabel
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QSpinBox,
    QLineEdit,
    QComboBox,
    QGroupBox,
    QGridLayout,
    QCheckBox,
)
from py4D_browser.utils import make_detector, StatusBarWriter


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


class CalibrateDialog(QDialog):
    def __init__(self, datacube, parent, diffraction_selector_size=None):
        super().__init__(parent=parent)

        self.datacube = datacube
        self.parent = parent
        self.diffraction_selector_size = diffraction_selector_size

        layout = QVBoxLayout(self)

        ####### LAYOUT ########

        realspace_box = QGroupBox("Real Space")
        layout.addWidget(realspace_box)
        realspace_layout = QHBoxLayout()
        realspace_box.setLayout(realspace_layout)

        realspace_left_layout = QGridLayout()
        realspace_layout.addLayout(realspace_left_layout)

        realspace_left_layout.addWidget(QLabel("Pixel Size"), 0, 0, Qt.AlignRight)
        self.realspace_pix_box = QLineEdit()
        self.realspace_pix_box.setValidator(QDoubleValidator())
        realspace_left_layout.addWidget(self.realspace_pix_box, 0, 1)

        realspace_left_layout.addWidget(QLabel("Full Width"), 1, 0, Qt.AlignRight)
        self.realspace_fov_box = QLineEdit()
        realspace_left_layout.addWidget(self.realspace_fov_box, 1, 1)

        realspace_right_layout = QHBoxLayout()
        realspace_layout.addLayout(realspace_right_layout)
        self.realspace_unit_box = QComboBox()
        self.realspace_unit_box.addItems(["Å", "nm"])
        self.realspace_unit_box.setMinimumContentsLength(5)
        realspace_right_layout.addWidget(self.realspace_unit_box)

        diff_box = QGroupBox("Diffraction")
        layout.addWidget(diff_box)
        diff_layout = QHBoxLayout()
        diff_box.setLayout(diff_layout)

        diff_left_layout = QGridLayout()
        diff_layout.addLayout(diff_left_layout)

        diff_left_layout.addWidget(QLabel("Pixel Size"), 0, 0, Qt.AlignRight)
        self.diff_pix_box = QLineEdit()
        diff_left_layout.addWidget(self.diff_pix_box, 0, 1)

        diff_left_layout.addWidget(QLabel("Full Width"), 1, 0, Qt.AlignRight)
        self.diff_fov_box = QLineEdit()
        diff_left_layout.addWidget(self.diff_fov_box, 1, 1)

        diff_left_layout.addWidget(QLabel("Selection Radius"), 2, 0, Qt.AlignRight)
        self.diff_selection_box = QLineEdit()
        diff_left_layout.addWidget(self.diff_selection_box, 2, 1)
        self.diff_selection_box.setEnabled(self.diffraction_selector_size is not None)

        diff_right_layout = QHBoxLayout()
        diff_layout.addLayout(diff_right_layout)
        self.diff_unit_box = QComboBox()
        self.diff_unit_box.setMinimumContentsLength(5)
        self.diff_unit_box.addItems(
            [
                "mrad",
                "Å⁻¹",
                # "nm⁻¹",
            ]
        )
        diff_right_layout.addWidget(self.diff_unit_box)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        cancel_button = QPushButton("Cancel")
        cancel_button.pressed.connect(self.close)
        button_layout.addWidget(cancel_button)
        done_button = QPushButton("Done")
        done_button.pressed.connect(self.set_and_close)
        button_layout.addWidget(done_button)
        layout.addLayout(button_layout)

        ######### CALLBACKS ########
        self.realspace_pix_box.textEdited.connect(self.realspace_pix_box_changed)
        self.realspace_fov_box.textEdited.connect(self.realspace_fov_box_changed)
        self.diff_pix_box.textEdited.connect(self.diffraction_pix_box_changed)
        self.diff_fov_box.textEdited.connect(self.diffraction_fov_box_changed)
        self.diff_selection_box.textEdited.connect(
            self.diffraction_selection_box_changed
        )

    def realspace_pix_box_changed(self, new_text):
        pix_size = float(new_text)

        fov = pix_size * self.datacube.R_Ny
        self.realspace_fov_box.setText(f"{fov:g}")

    def realspace_fov_box_changed(self, new_text):
        fov = float(new_text)

        pix_size = fov / self.datacube.R_Ny
        self.realspace_pix_box.setText(f"{pix_size:g}")

    def diffraction_pix_box_changed(self, new_text):
        pix_size = float(new_text)

        fov = pix_size * self.datacube.Q_Ny
        self.diff_fov_box.setText(f"{fov:g}")

        if self.diffraction_selector_size:
            sel_size = pix_size * self.diffraction_selector_size
            self.diff_selection_box.setText(f"{sel_size:g}")

    def diffraction_fov_box_changed(self, new_text):
        fov = float(new_text)

        pix_size = fov / self.datacube.Q_Ny
        self.diff_pix_box.setText(f"{pix_size:g}")

        if self.diffraction_selector_size:
            sel_size = pix_size * self.diffraction_selector_size
            self.diff_selection_box.setText(f"{sel_size:g}")

    def diffraction_selection_box_changed(self, new_text):
        if self.diffraction_selector_size:
            sel_size = float(new_text)

            pix_size = sel_size / self.diffraction_selector_size
            fov = pix_size * self.datacube.Q_Nx
            self.diff_pix_box.setText(f"{pix_size:g}")
            self.diff_fov_box.setText(f"{fov:g}")

            sel_size = pix_size * self.diffraction_selector_size
            self.diff_selection_box.setText(f"{sel_size:g}")

    def set_and_close(self):

        print("Old calibration")
        print(self.datacube.calibration)

        realspace_text = self.realspace_pix_box.text()
        if realspace_text != "":
            realspace_pix = float(realspace_text)
            self.datacube.calibration.set_R_pixel_size(realspace_pix)
            self.datacube.calibration.set_R_pixel_units(
                self.realspace_unit_box.currentText().replace("Å", "A")
            )

        diff_text = self.diff_pix_box.text()
        if diff_text != "":
            diff_pix = float(diff_text)
            self.datacube.calibration.set_Q_pixel_size(diff_pix)
            translation = {
                "mrad": "mrad",
                "Å⁻¹": "A^-1",
                "nm⁻¹": "1/nm",
            }
            self.datacube.calibration.set_Q_pixel_units(
                translation[self.diff_unit_box.currentText()]
            )

        self.parent.update_scalebars()

        print("New calibration")
        print(self.datacube.calibration)

        self.close()
