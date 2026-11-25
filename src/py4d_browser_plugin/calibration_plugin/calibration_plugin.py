from py4DSTEM import DataCube, data
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm
from PyQt5.QtWidgets import QFrame, QPushButton, QApplication, QLabel
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QPointF, Qt, QObject
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
    format_unit,
)


class CalibrationPlugin(QWidget):

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "py4DGUI.internal.calibration"

    # the plugin API version this plugin was written against
    api_version = (1, 6)

    uses_single_action = True
    display_name = "Calibrate..."

    def __init__(self, api, plugin_action, **kwargs):
        super().__init__()

        self.api = api

        plugin_action.triggered.connect(self.launch_dialog)

    def close(self):
        pass

    def launch_dialog(self):
        api = self.api
        # If the selector has a size, figure that out
        detector_info: DetectorInfo = api.get_diffraction_detector()

        match detector_info["shape"]:
            case DetectorShape.CIRCLE:
                circle_geometry: CircleGeometry = detector_info["geometry"]
                selector_size = circle_geometry["R"]
            case _:
                selector_size = None
                api.status_bar.showMessage(
                    "Use a Circle selection to calibrate based on a known spacing...",
                    5_000,
                )

        dialog = CalibrateDialog(
            api,
            diffraction_selector_size=selector_size,
        )
        dialog.open()


class CalibrateDialog(QDialog):
    def __init__(self, api, diffraction_selector_size=None):
        # Parented to the API's dialog-parent widget so it shows over the
        # main window; all data/method access goes through the API object.
        super().__init__(parent=api.qt_window)

        self.api = api
        self.datacube = api.datacube
        self.diffraction_selector_size = diffraction_selector_size

        layout = QVBoxLayout(self)

        ####### LAYOUT ########

        realspace_box = QGroupBox("Real Space")
        layout.addWidget(realspace_box)
        realspace_layout = QVBoxLayout()
        realspace_box.setLayout(realspace_layout)

        realspace_top_layout = QHBoxLayout()
        realspace_layout.addLayout(realspace_top_layout)

        realspace_left_layout = QGridLayout()
        realspace_top_layout.addLayout(realspace_left_layout)

        realspace_left_layout.addWidget(QLabel("Pixel Size"), 0, 0, Qt.AlignRight)
        self.realspace_pix_box = QLineEdit()
        self.realspace_pix_box.setValidator(QDoubleValidator())
        realspace_left_layout.addWidget(self.realspace_pix_box, 0, 1)

        realspace_left_layout.addWidget(QLabel("Full Width"), 1, 0, Qt.AlignRight)
        self.realspace_fov_box = QLineEdit()
        realspace_left_layout.addWidget(self.realspace_fov_box, 1, 1)

        realspace_right_layout = QHBoxLayout()
        realspace_top_layout.addLayout(realspace_right_layout)
        self.realspace_unit_box = QComboBox()
        self.realspace_unit_box.addItems(["Å", "nm"])
        self.realspace_unit_box.setMinimumContentsLength(5)
        realspace_right_layout.addWidget(self.realspace_unit_box)

        # Calibrate the R pixel size from a periodicity in the virtual image's FFT.
        period_button = QPushButton("From FFT...")
        period_button.pressed.connect(self.launch_period_dialog)
        realspace_layout.addWidget(period_button)

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
        self.diff_selection_box.setValidator(QDoubleValidator())
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

        kV_box = QGroupBox("Energy")
        layout.addWidget(kV_box)
        kV_layout = QHBoxLayout()
        kV_box.setLayout(kV_layout)
        kV_left_layout = QGridLayout()
        kV_layout.addLayout(kV_left_layout)
        kV_left_layout.addWidget(
            QLabel("Accelerating Voltage [kV]"), 0, 0, Qt.AlignRight
        )
        self.kV_input = QLineEdit()
        kV_left_layout.addWidget(self.kV_input, 0, 1)

        orientation_box = QGroupBox("Orientation")
        layout.addWidget(orientation_box)
        orientation_layout = QHBoxLayout()
        orientation_box.setLayout(orientation_layout)
        orientation_left_layout = QGridLayout()
        orientation_layout.addLayout(orientation_left_layout)
        orientation_left_layout.addWidget(
            QLabel("Scan Rotation [deg]"), 0, 0, Qt.AlignRight
        )
        self.rotation_input = QLineEdit()
        self.rotation_input.setValidator(QDoubleValidator())
        orientation_left_layout.addWidget(self.rotation_input, 0, 1)

        orientation_left_layout.addWidget(
            QLabel("Pattern Transpose"), 1, 0, Qt.AlignRight
        )
        self.transpose_checkbox = QCheckBox()
        orientation_left_layout.addWidget(self.transpose_checkbox, 1, 1)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        cancel_button = QPushButton("Cancel")
        cancel_button.pressed.connect(self.close)
        button_layout.addWidget(cancel_button)
        done_button = QPushButton("Done")
        done_button.pressed.connect(self.set_and_close)
        button_layout.addWidget(done_button)
        layout.addLayout(button_layout)

        ######### POPULATE ########

        # Real Space
        r_unit = self.datacube.calibration.get_R_pixel_units()
        r_unit_map = {"A": "Å", "nm": "nm"}

        # A unit not in the map (e.g. "pixels") means this field was never
        # calibrated in physical units; leave the boxes empty so set_and_close
        # (which skips empty fields) can't apply a spurious calibration.
        if r_unit in r_unit_map:
            r_pix = self.datacube.calibration.get_R_pixel_size()
            self.realspace_pix_box.setText(f"{r_pix:g}")
            self.realspace_fov_box.setText(f"{r_pix * self.datacube.R_Ny:g}")
        self.realspace_unit_box.setCurrentText(r_unit_map.get(r_unit, "Å"))

        # Diffraction
        q_unit = self.datacube.calibration.get_Q_pixel_units()
        q_unit_map = {"mrad": "mrad", "A^-1": "Å⁻¹", "1/nm": "nm⁻¹"}

        # Same rule as above: don't pre-fill a field that has no physical-unit
        # calibration yet.
        if q_unit in q_unit_map:
            q_pix = self.datacube.calibration.get_Q_pixel_size()
            self.diff_pix_box.setText(f"{q_pix:g}")
            self.diff_fov_box.setText(f"{q_pix * self.datacube.Q_Ny:g}")
            if self.diffraction_selector_size:
                self.diff_selection_box.setText(
                    f"{q_pix * self.diffraction_selector_size:g}"
                )
        self.diff_unit_box.setCurrentText(q_unit_map.get(q_unit, "mrad"))

        # Voltage
        try:
            voltage = self.datacube.calibration["voltage"]
            self.kV_input.setText(f"{voltage / 1e3:g}")
        except (KeyError, TypeError):
            pass

        # Orientation
        rotation = self.datacube.calibration.get_QR_rotation_degrees()
        if rotation is not None:
            self.rotation_input.setText(f"{rotation:g}")
        self.transpose_checkbox.setChecked(self.datacube.calibration.get_QR_flip())

        ######### CALLBACKS ########
        self.realspace_pix_box.textEdited.connect(self.realspace_pix_box_changed)
        self.realspace_fov_box.textEdited.connect(self.realspace_fov_box_changed)
        self.diff_pix_box.textEdited.connect(self.diffraction_pix_box_changed)
        self.diff_fov_box.textEdited.connect(self.diffraction_fov_box_changed)
        self.diff_selection_box.textEdited.connect(
            self.diffraction_selection_box_changed
        )
        self.rotation_input.textEdited.connect(self.rotation_input_changed)

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
            try:
                sel_size = float(new_text)
            except ValueError:
                return

            pix_size = sel_size / self.diffraction_selector_size
            fov = pix_size * self.datacube.Q_Ny
            self.diff_pix_box.setText(f"{pix_size:g}")
            self.diff_fov_box.setText(f"{fov:g}")

    def rotation_input_changed(self, new_text):
        pass  # no dependent fields to update

    def launch_period_dialog(self):
        # Calibrate the R pixel size from a periodicity in the virtual image's
        # FFT; requires a 2D virtual image.
        vimg = self.api.current_virtual_image
        if vimg is None:
            self.api.status_bar.showMessage(
                "From FFT: no virtual image is on display.",
                5_000,
            )
            return
        if vimg.ndim != 2:
            self.api.status_bar.showMessage(
                "From FFT requires a 2D virtual image.",
                5_000,
            )
            return
        dialog = CalibrateFromPeriodDialog(self.api, self)
        dialog.open()

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

        kV_text = self.kV_input.text()
        if kV_text != "":
            kV = float(kV_text)
            self.datacube.calibration["voltage"] = kV * 1e3
            # note there is no canonical tag for voltage, so we are
            # going to make our own key for it

        rotation_text = self.rotation_input.text()
        if rotation_text != "":
            rotation = float(rotation_text)
            self.datacube.calibration.set_QR_rotation(np.deg2rad(rotation))

        self.datacube.calibration.set_QR_flip(self.transpose_checkbox.isChecked())

        # use the versioned API's set_scalebar rather than poking the
        # scale bar objects directly
        self.api.set_scalebar(
            "real_space",
            self.datacube.calibration.get_R_pixel_size(),
            format_unit(self.datacube.calibration.get_R_pixel_units()),
        )
        self.api.set_scalebar(
            "diffraction",
            self.datacube.calibration.get_Q_pixel_size(),
            format_unit(self.datacube.calibration.get_Q_pixel_units()),
        )

        # The result bar (FFT / EWPC) only updates on a result-view re-render,
        # which closing doesn't trigger — recompute it from the new calibration.
        self.api.refresh_result_scale_bar()

        print("New calibration")
        print(self.datacube.calibration)

        self.close()


class CalibrateFromPeriodDialog(QDialog):
    """
    Calibrate the real-space pixel size from a periodicity in the virtual
    image.
    """

    def __init__(self, api, parent_dialog):
        # parent_dialog is the CalibrateDialog we launched from; we use it to
        # reflect the new calibration back into its fields.
        super().__init__(parent=api.qt_window)

        self.api = api
        self.datacube = api.datacube
        self.parent_dialog = parent_dialog

        vimg = api.current_virtual_image
        vimg = vimg if np.isrealobj(vimg) else np.abs(vimg)
        self.N1 = vimg.shape[1]  # side of the square display; see class doc.

        layout = QVBoxLayout(self)

        ######### FFT VIEW + ROI #########

        self.image_widget = pg.ImageView()
        self.image_widget.getView().setMenuEnabled(False)
        layout.addWidget(self.image_widget)

        # Magnitude FFT, sqrt-scaled and 0.1-99.9% clipped, mirroring the
        # result pane's 2D-FFT computation.
        fx = np.hanning(vimg.shape[0])
        fy = np.hanning(vimg.shape[1])
        window = fx[:, None] * fy[None, :]
        fft = np.abs(np.fft.fftshift(np.fft.fft2(vimg * window))) ** 0.5
        lo = float(np.percentile(fft, 0.1))
        hi = float(np.percentile(fft, 99.9))
        self.image_widget.setImage(
            fft.T,
            autoLevels=False,
            levels=(lo, hi),
            autoRange=True,
        )
        # Force a square view: this is what makes the frequency axis
        # isotropic, so a circular ROI is a constant-period ring.
        image_item = self.image_widget.getImageItem()
        image_item.setRect(0, 0, self.N1, self.N1)
        self.image_widget.autoRange()

        # Fixed-center circular ROI on the zero frequency (the view center).
        main_pen = {"color": "g", "width": 6}
        handle_pen = {"color": "r", "width": 9}
        hover_pen = {"color": "c", "width": 6}
        hover_handle = {"color": "c", "width": 9}
        # fftshift places DC at floor(N/2) on each axis. Map that pixel's
        # center through the rectangular-to-square display transform.
        center = image_item.mapToParent(
            QPointF(vimg.shape[1] // 2 + 0.5, vimg.shape[0] // 2 + 0.5)
        )
        default_r = max(1.0, min(vimg.shape) / 10.0)
        self.roi = pg.CircleROI(
            [center.x() - default_r, center.y() - default_r],
            [2 * default_r, 2 * default_r],
            movable=False,
            pen=main_pen,
            handlePen=handle_pen,
            hoverPen=hover_pen,
            handleHoverPen=hover_handle,
        )
        self.image_widget.getView().addItem(self.roi)

        ######### CONTROLS #########

        controls = QGridLayout()
        layout.addLayout(controls)

        controls.addWidget(QLabel("Real-space Period"), 0, 0, Qt.AlignRight)
        self.period_box = QLineEdit()
        self.period_box.setValidator(QDoubleValidator())
        controls.addWidget(self.period_box, 0, 1)

        self.unit_box = QComboBox()
        self.unit_box.addItems(["Å", "nm"])
        self.unit_box.setMinimumContentsLength(5)
        # Default to the unit already chosen in the parent dialog.
        self.unit_box.setCurrentText(
            self.parent_dialog.realspace_unit_box.currentText()
        )
        controls.addWidget(self.unit_box, 0, 2)

        self.pixel_size_label = QLabel("")
        controls.addWidget(self.pixel_size_label, 1, 0, 1, 3)

        ######### CALLBACKS #########

        self.roi.sigRegionChanged.connect(self.update_readout)
        self.period_box.textEdited.connect(self.update_readout)
        self.unit_box.currentTextChanged.connect(self.update_readout)
        self.update_readout()

        ######### BUTTONS #########

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.pressed.connect(self.close)
        button_layout.addWidget(close_button)
        calibrate_button = QPushButton("Calibrate")
        calibrate_button.pressed.connect(self.calibrate_and_close)
        button_layout.addWidget(calibrate_button)
        layout.addLayout(button_layout)

        # Size the FFT to roughly half the main window width
        top = self
        while top.parent() is not None:
            top = top.parent()
        main_width = top.width() if top is not None else 0
        w = int(main_width / 2) if main_width > 0 else 900
        self.resize(w, w + 140)

    def roi_radius(self):
        # The ROI radius in view units (the isotropic frequency space).
        return self.roi.size()[0] / 2.0

    def update_readout(self, *args):
        r_view = self.roi_radius()
        try:
            period = float(self.period_box.text())
        except ValueError:
            self.pixel_size_label.setText(
                f"ROI radius: {r_view:.4g} px  (set the period to compute the pixel size)"
            )
            return
        unit = self.unit_box.currentText()
        pixel_size = r_view * period / self.N1
        self.pixel_size_label.setText(f"Pixel size: {pixel_size:.6g} {unit}")

    def calibrate_and_close(self):
        try:
            period = float(self.period_box.text())
        except ValueError:
            self.api.status_bar.showMessage(
                "From FFT: enter a real-space period first.",
                5_000,
            )
            return

        pixel_size = self.roi_radius() * period / self.N1
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            self.api.status_bar.showMessage(
                "From FFT: enter a positive period and select a nonzero radius.", 5_000
            )
            return
        unit = self.unit_box.currentText()

        # Apply to the datacube calibration (R pixel size is isotropic).
        self.datacube.calibration.set_R_pixel_size(pixel_size)
        self.datacube.calibration.set_R_pixel_units(unit.replace("Å", "A"))

        # Refresh the real-space scale bar through the versioned API.
        self.api.set_scalebar(
            "real_space",
            self.datacube.calibration.get_R_pixel_size(),
            format_unit(self.datacube.calibration.get_R_pixel_units()),
        )
        self.api.refresh_result_scale_bar()

        # Reflect the new calibration back into the parent dialog.
        self.parent_dialog.realspace_unit_box.setCurrentText(unit)
        self.parent_dialog.realspace_pix_box.setText(f"{pixel_size:g}")
        self.parent_dialog.realspace_fov_box.setText(
            f"{pixel_size * self.datacube.R_Ny:g}"
        )

        self.close()
