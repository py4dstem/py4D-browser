import numpy as np
from tqdm import tqdm
from PyQt5.QtWidgets import QPushButton, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
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
    StatusBarWriter,
)
import py4DSTEM


class tcBFPlugin(QWidget):

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "py4DGUI.internal.tcBF"

    uses_plugin_menu = True
    display_name = "Tilt-Corrected BF"

    def __init__(self, parent, plugin_menu, **kwargs):
        super().__init__()

        self.parent = parent

        manual_action = plugin_menu.addAction("Manual tcBF...")
        manual_action.triggered.connect(self.launch_manual)

        auto_action = plugin_menu.addAction("Automatic tcBF")
        auto_action.triggered.connect(self.launch_auto)

    def close(self):
        pass  # perform any shutdown activities

    def launch_manual(self):
        dialog = ManualTCBFDialog(parent=self.parent)
        dialog.show()

    def launch_auto(self):
        parent = self.parent

        detector: DetectorInfo = self.parent.get_diffraction_detector()

        if detector["shape"] is DetectorShape.POINT:
            parent.statusBar().showMessage("tcBF requires an area detector!", 5_000)
            return

        if (
            parent.datacube.calibration.get_R_pixel_units == "pixels"
            or parent.datacube.calibration.get_Q_pixel_units == "pixels"
        ):
            parent.statusBar().showMessage("Auto tcBF requires caibrated data", 5_000)
            return

        # do tcBF!
        parent.statusBar().showMessage("Reconstructing... (This may take a while)")
        parent.qtapp.processEvents()

        tcBF = py4DSTEM.process.phase.Parallax(
            energy=300e3,
            datacube=parent.datacube,
        )
        tcBF.preprocess(
            dp_mask=detector["mask"],
            plot_average_bf=False,
            vectorized_com_calculation=False,
            store_initial_arrays=False,
        )
        tcBF.reconstruct(
            plot_aligned_bf=False,
            plot_convergence=False,
        )

        parent.set_virtual_image(tcBF.recon_BF, reset=True)


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
        self.rotation_box = QLineEdit()
        self.rotation_box.setValidator(QDoubleValidator())
        params_layout.addWidget(self.rotation_box, 0, 1)

        params_layout.addWidget(QLabel("Transpose x/y"), 1, 0, Qt.AlignRight)
        self.transpose_box = QCheckBox()
        params_layout.addWidget(self.transpose_box, 1, 1)

        params_layout.addWidget(QLabel("Max Shift [px]"), 2, 0, Qt.AlignRight)
        self.max_shift_box = QLineEdit()
        self.max_shift_box.setValidator(QDoubleValidator())
        params_layout.addWidget(self.max_shift_box, 2, 1)

        params_layout.addWidget(QLabel("Pad Images"), 3, 0, Qt.AlignRight)
        self.pad_checkbox = QCheckBox()
        params_layout.addWidget(self.pad_checkbox, 3, 1)

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
        detector: DetectorInfo = self.parent.get_diffraction_detector()

        if detector["shape"] is DetectorShape.POINT:
            self.parent.statusBar().showMessage(
                "tcBF requires an area detector!", 5_000
            )
            return

        mask = detector["mask"]

        if self.max_shift_box.text() == "":
            self.parent.statusBar().showMessage("Max Shift must be specified")
            return

        rotation = np.radians(float(self.rotation_box.text() or 0.0))
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

        R = np.array(
            [
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)],
            ]
        )
        T = np.array([[0.0, 1.0], [1.0, 0.0]])

        if transpose:
            R = T @ R

        shifts_pix = np.stack([shifts_pix_x, shifts_pix_y], axis=2) @ R
        shifts_pix_x, shifts_pix_y = shifts_pix[..., 0], shifts_pix[..., 1]

        # generate image to accumulate reconstruction
        pad = self.pad_checkbox.checkState()
        pad_width = int(
            np.maximum(np.abs(shifts_pix_x).max(), np.abs(shifts_pix_y).max())
        )

        reconstruction = (
            np.zeros((datacube.R_Nx + 2 * pad_width, datacube.R_Ny + 2 * pad_width))
            if pad
            else np.zeros((datacube.R_Nx, datacube.R_Ny))
        )

        qx = np.fft.fftfreq(reconstruction.shape[0])
        qy = np.fft.fftfreq(reconstruction.shape[1])

        qx_operator, qy_operator = np.meshgrid(qx, qy, indexing="ij")
        qx_operator = qx_operator * -2.0j * np.pi
        qy_operator = qy_operator * -2.0j * np.pi

        # loop over images and shift
        img_indices = np.argwhere(mask)
        for mx, my in tqdm(
            img_indices,
            desc="Shifting images",
            file=StatusBarWriter(self.parent.statusBar()),
            mininterval=1.0,
        ):
            if mask[mx, my]:
                img_raw = datacube.data[:, :, mx, my]

                if pad:
                    img = np.zeros_like(reconstruction) + img_raw.mean()
                    img[
                        pad_width : img_raw.shape[0] + pad_width,
                        pad_width : img_raw.shape[1] + pad_width,
                    ] = img_raw
                else:
                    img = img_raw

                reconstruction += np.real(
                    np.fft.ifft2(
                        np.fft.fft2(img)
                        * np.exp(
                            qx_operator * shifts_pix_x[mx, my]
                            + qy_operator * shifts_pix_y[mx, my]
                        )
                    )
                )

        # crop away padding so the image lines up with the original
        if pad:
            reconstruction = reconstruction[pad_width:-pad_width, pad_width:-pad_width]

        self.parent.set_virtual_image(reconstruction, reset=True)
