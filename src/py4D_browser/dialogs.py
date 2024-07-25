from py4DSTEM import DataCube, data
import pyqtgraph as pg
import numpy as np
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
from py4D_browser.utils import make_detector


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


class ManualTCBFDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)

        self.parent = parent

        layout = QVBoxLayout(self)

        ####### LAYOUT ########

        params_box = QGroupBox("Parameters")
        layout.addWidget(params_box)

        params_layout = QGridLayout()
        params_box.setLayout(params_layout)

        params_layout.addWidget(QLabel("Rotation [deg]"), 0, 0, Qt.AlignRight)
        rotation_box = QLineEdit()
        rotation_box.setValidator(QDoubleValidator())
        params_layout.addWidget(rotation_box, 0, 1)

        params_layout.addWidget(QLabel("Transpose x/y"), 1, 0, Qt.AlignRight)
        transpose_box = QCheckBox()
        params_layout.addWidget(transpose_box, 1, 1)

        params_layout.addWidget(QLabel("Max Shift [px]"), 2, 0, Qt.AlignRight)
        self.max_shift_box = QLineEdit()
        self.max_shift_box.setValidator(QDoubleValidator())
        params_layout.addWidget(self.max_shift_box, 2, 1)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        cancel_button = QPushButton("Cancel")
        cancel_button.pressed.connect(self.close)
        button_layout.addWidget(cancel_button)
        done_button = QPushButton("Reconstruct")
        done_button.pressed.connect(self.reconstruct)
        button_layout.addWidget(done_button)
        layout.addLayout(button_layout)

    def reconstruct(self):
        datacube = self.parent.datacube

        # tcBF requires an area detector for generating the mask
        detector_shape = (
            self.parent.detector_shape_group.checkedAction().text().replace("&", "")
        )
        if detector_shape not in [
            "Rectangular",
            "Circle",
        ]:
            self.parent.statusBar().showMessage(
                "tcBF requires a selection of the BF disk"
            )
            return

        if detector_shape == "Rectangular":
            # Get slices corresponding to ROI
            slices, _ = self.parent.virtual_detector_roi.getArraySlice(
                self.parent.datacube.data[0, 0, :, :],
                self.parent.diffraction_space_widget.getImageItem(),
            )
            slice_y, slice_x = slices

            mask = np.zeros(
                (self.parent.datacube.Q_Nx, self.parent.datacube.Q_Ny), dtype=np.bool_
            )
            mask[slice_x, slice_y] = True

        elif detector_shape == "Circle":
            R = self.parent.virtual_detector_roi.size()[0] / 2.0

            x0 = self.parent.virtual_detector_roi.pos()[0] + R
            y0 = self.parent.virtual_detector_roi.pos()[1] + R

            mask = make_detector(
                (self.parent.datacube.Q_Nx, self.parent.datacube.Q_Ny),
                "circle",
                ((x0, y0), R),
            )
        else:
            raise ValueError("idk how we got here...")

        if self.max_shift_box.text() == "":
            self.parent.statusBar().showMessage("Max Shift must be specified")
            return

        rotation = float(self.rotation_box.text() or 0.0)
        transpose = self.transpose_box.checkState()
        max_shift = float(self.max_shift_box.text())

        x, y = np.meshgrid(
            np.arange(datacube.Q_Nx), np.arange(datacube.Q_Ny), indexing="ij"
        )

        mask_comx = np.sum(mask * x) / np.sum(mask)
        mask_comy = np.sum(mask * y) / np.sum(mask)

        pix_coord_x = x - mask_comx
        pix_coord_y = y - mask_comy

        q_pix = np.hypot(pix_coord_x, pix_coord_y)
        # unrotated shifts in scan pixels
        shifts_pix_x = pix_coord_x / np.max(q_pix * mask) * max_shift
        shifts_pix_y = pix_coord_y / np.max(q_pix * mask) * max_shift
        # shifts_pix = np.

        R = np.array(
            [
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)],
            ]
        )
        T = np.array([[0.0, 1.0], [1.0, 0.0]])

        if transpose:
            R = T @ R
