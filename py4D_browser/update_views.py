import pyqtgraph as pg
import numpy as np


def update_real_space_view(self, reset=False):
    scaling_mode = self.vimg_scaling_group.checkedAction().text().strip("&")
    assert scaling_mode in ["Linear", "Log", "Square Root"]

    detector_shape = self.detector_shape_group.checkedAction().text().strip("&")
    assert detector_shape in ["Rectangular", "Circle", "Annulus"]


def update_diffraction_space_view(self, reset=False):
    scaling_mode = self.diff_scaling_group.checkedAction().text().strip("&")
    assert scaling_mode in ["Linear", "Log", "Square Root"]

    if self.datacube is None:
        return

    roi_state = self.real_space_point_selector.saveState()
    y0, x0 = roi_state["pos"]
    xc, yc = int(x0 + 1), int(y0 + 1)

    # Set the diffraction space image
    # Normalize coordinates
    xc = np.clip(xc, 0, self.datacube.R_Nx - 1)
    yc = np.clip(yc, 0, self.datacube.R_Ny - 1)
    DP = self.datacube.data[xc, yc]

    self.real_space_view_text.setText(f"[{xc},{yc}]")

    if scaling_mode == "Linear":
        new_view = DP
    elif scaling_mode == "Log":
        new_view = np.log(np.maximum(DP, 1e-30))
    elif scaling_mode == "Square Root":
        new_view = np.sqrt(np.maximum(DP, 0))
    else:
        raise ValueError("Mode not recognized")

    self.diffraction_space_widget.setImage(
        new_view.T, autoLevels=reset, autoRange=reset
    )


def update_diffraction_detector(self):
    # change the shape of the detector, then update the view

    detector_shape = self.detector_shape_group.checkedAction().text().strip("&")
    assert detector_shape in ["Rectangular", "Circle", "Annulus"]

    if self.datacube is None:
        return

    x, y = self.datacube.shape[2:]
    x0, y0 = x / 2, y / 2
    xr, yr = x / 10, y / 10

    # Remove existing detector
    if hasattr(self, "virtual_detector_roi"):
        self.diffraction_space_widget.view.scene().removeItem(self.virtual_detector_roi)
    if hasattr(self, "virtual_detector_roi_inner"):
        self.diffraction_space_widget.view.scene().removeItem(
            self.virtual_detector_roi_inner
        )
    if hasattr(self, "virtual_detector_roi_outer"):
        self.diffraction_space_widget.view.scene().removeItem(
            self.virtual_detector_roi_outer
        )

    # Rectangular detector
    if detector_shape == "Rectangular":
        self.virtual_detector_roi = pg.RectROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)], [int(xr), int(yr)], pen=(3, 9)
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChangeFinished.connect(
            self.update_real_space_view
        )

    # Circular detector
    elif detector_shape == "Circle":
        self.virtual_detector_roi = pg.CircleROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)], [int(xr), int(yr)], pen=(3, 9)
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChangeFinished.connect(
            self.update_real_space_view
        )

    # Annular dector
    elif detector_shape == "Annulus":
        # Make outer detector
        self.virtual_detector_roi_outer = pg.CircleROI(
            [int(x0 - xr), int(y0 - yr)], [int(2 * xr), int(2 * yr)], pen=(3, 9)
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi_outer)

        # Make inner detector
        self.virtual_detector_roi_inner = pg.CircleROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)],
            [int(xr), int(yr)],
            pen=(4, 9),
            movable=False,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi_inner)

        # Connect size/position of inner and outer detectors
        self.virtual_detector_roi_outer.sigRegionChangeFinished.connect(
            self.update_annulus_pos
        )
        self.virtual_detector_roi_outer.sigRegionChangeFinished.connect(
            self.update_annulus_radii
        )
        self.virtual_detector_roi_inner.sigRegionChangeFinished.connect(
            self.update_annulus_radii
        )

        # Connect to real space view update function
        self.virtual_detector_roi_outer.sigRegionChangeFinished.connect(
            self.update_real_space_view
        )
        self.virtual_detector_roi_inner.sigRegionChangeFinished.connect(
            self.update_real_space_view
        )

    else:
        raise ValueError(
            "Unknown detector shape value {}.  Must be 0, 1, or 2.".format(
                detector_shape
            )
        )

    self.update_real_space_view()


def update_annulus_pos(self):
    """
    Function to keep inner and outer rings of annulus aligned.
    """
    R_outer = self.virtual_detector_roi_outer.size().x() / 2
    R_inner = self.virtual_detector_roi_inner.size().x() / 2
    # Only outer annulus is draggable; when it moves, update position of inner annulus
    x0 = self.virtual_detector_roi_outer.pos().x() + R_outer
    y0 = self.virtual_detector_roi_outer.pos().y() + R_outer
    self.virtual_detector_roi_inner.setPos(x0 - R_inner, y0 - R_inner)


def update_annulus_radii(self):
    R_outer = self.virtual_detector_roi_outer.size().x() / 2
    R_inner = self.virtual_detector_roi_inner.size().x() / 2
    if R_outer < R_inner:
        x0 = self.virtual_detector_roi_outer.pos().x() + R_outer
        y0 = self.virtual_detector_roi_outer.pos().y() + R_outer
        self.virtual_detector_roi_outer.setSize(2 * R_inner + 6)
        self.virtual_detector_roi_outer.setPos(x0 - R_inner - 3, y0 - R_inner - 3)
