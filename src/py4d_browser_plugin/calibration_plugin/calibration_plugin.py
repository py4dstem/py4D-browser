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
    QWidget,
)
from py4D_browser.utils import (
    DetectorShape,
    DetectorInfo,
    RectangleGeometry,
    CircleGeometry,
)


class CalibrationPlugin(QWidget):

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "py4DGUI.internal.calibration"

    uses_single_action = True
    display_name = "Calibrate..."

    def __init__(self, parent, plugin_action, **kwargs):
        super().__init__()

        self.parent = parent

        plugin_action.triggered.connect(self.launch_dialog)

    def close(self):
        pass

    def launch_dialog(self):
        parent = self.parent
        # If the selector has a size, figure that out
        detector_info: DetectorInfo = parent.get_diffraction_detector()

        match detector_info["shape"]:
            case DetectorShape.CIRCLE:
                circle_geometry: CircleGeometry = detector_info["geometry"]
                selector_size = circle_geometry["R"]
            case _:
                selector_size = None
                parent.statusBar().showMessage(
                    "Use a Circle selection to calibrate based on a known spacing...",
                    5_000,
                )

        dialog = CalibrateDialog(
            parent.datacube, parent=parent, diffraction_selector_size=selector_size
        )
        dialog.open()


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
