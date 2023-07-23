import pyqtgraph as pg
import numpy as np
import py4DSTEM

from py4D_browser.utils import pg_point_roi


def update_real_space_view(self, reset=False):
    scaling_mode = self.vimg_scaling_group.checkedAction().text().replace("&", "")
    assert scaling_mode in ["Linear", "Log", "Square Root"], scaling_mode

    detector_shape = self.detector_shape_group.checkedAction().text().replace("&", "")
    assert detector_shape in ["Point", "Rectangular", "Circle", "Annulus"], detector_shape

    detector_mode = self.detector_mode_group.checkedAction().text().replace("&", "")
    assert detector_mode in [
        "Integrating",
        "Maximum",
        "CoM Magnitude",
        "CoM Angle",
    ], detector_mode

    # If a CoM method is checked, ensure linear scaling
    if detector_mode in ["CoM Magnitude", "CoM Angle"] and scaling_mode != "Linear":
        print("Warning! Setting linear scaling for CoM image")
        self.vimg_scale_linear_action.setChecked(True)
        scaling_mode = "Linear"

    if self.datacube is None:
        return

    # We will branch through certain combinations of detector shape and mode.
    # If we happen across a special case that can be handled directly, we
    # compute vimg. If we encounter a case that needs a more complicated
    # computation we compute the mask and then do the virtual image later
    mask = None
    if detector_shape == "Rectangular":
        # Get slices corresponding to ROI
        slices, transforms = self.virtual_detector_roi.getArraySlice(
            self.datacube.data[0, 0, :, :], self.diffraction_space_widget.getImageItem()
        )
        slice_y, slice_x = slices

        # update the label:
        self.diffraction_space_view_text.setText(
            f"[{slice_x.start}:{slice_x.stop},{slice_y.start}:{slice_y.stop}]"
        )

        if detector_mode == "Integrating":
            vimg = np.sum(self.datacube.data[:, :, slice_x, slice_y], axis=(2, 3))
        elif detector_mode == "Maximum":
            vimg = np.max(self.datacube.data[:, :, slice_x, slice_y], axis=(2, 3))
        else:
            mask = np.zeros((self.datacube.Q_Nx, self.datacube.Q_Ny), dtype=np.bool_)
            mask[slice_x, slice_y] = True

    elif detector_shape == "Circle":
        (slice_x, slice_y), _ = self.virtual_detector_roi.getArraySlice(
            self.datacube.data[0, 0, :, :], self.diffraction_space_widget.getImageItem()
        )
        x0 = (slice_x.start + slice_x.stop) / 2.0
        y0 = (slice_y.start + slice_y.stop) / 2.0
        R = (slice_y.stop - slice_y.start) / 2.0

        self.diffraction_space_view_text.setText(f"[({x0},{y0}),{R}]")

        mask = py4DSTEM.datacube.virtualimage.DataCubeVirtualImager.make_detector(
            (self.datacube.Q_Nx, self.datacube.Q_Ny), "circle", ((x0, y0), R)
        )
    elif detector_shape == "Annulus":
        (slice_x, slice_y), _ = self.virtual_detector_roi_outer.getArraySlice(
            self.datacube.data[0, 0, :, :], self.diffraction_space_widget.getImageItem()
        )
        x0 = (slice_x.start + slice_x.stop) / 2.0
        y0 = (slice_y.start + slice_y.stop) / 2.0
        R_outer = (slice_y.stop - slice_y.start) / 2.0

        (slice_ix, slice_iy), _ = self.virtual_detector_roi_inner.getArraySlice(
            self.datacube.data[0, 0, :, :], self.diffraction_space_widget.getImageItem()
        )
        R_inner = (slice_iy.stop - slice_iy.start) / 2.0

        if R_inner == R_outer:
            R_inner -= 1

        self.diffraction_space_view_text.setText(f"[({x0},{y0}),({R_inner},{R_outer})]")

        mask = py4DSTEM.datacube.virtualimage.DataCubeVirtualImager.make_detector(
            (self.datacube.Q_Nx, self.datacube.Q_Ny),
            "annulus",
            ((x0, y0), (R_inner, R_outer)),
        )
    elif detector_shape == "Point":
        roi_state = self.virtual_detector_point.saveState()
        y0, x0 = roi_state["pos"]
        xc, yc = int(x0 + 1), int(y0 + 1)

        # Set the diffraction space image
        # Normalize coordinates
        xc = np.clip(xc, 0, self.datacube.Q_Nx - 1)
        yc = np.clip(yc, 0, self.datacube.Q_Ny - 1)
        vimg = self.datacube.data[: ,: , xc, yc]

        self.diffraction_space_view_text.setText(f"[{xc},{yc}]")

    else:
        raise ValueError("Detector shape not recognized")

    if mask is not None:
        vimg = np.zeros((self.datacube.R_Nx, self.datacube.R_Ny))
        iterator = py4DSTEM.tqdmnd(self.datacube.R_Nx, self.datacube.R_Ny, disable=True)

        if detector_mode == "Integrating":
            for rx, ry in iterator:
                vimg[rx, ry] = np.sum(self.datacube.data[rx, ry] * mask)

        elif detector_mode == "Maximum":
            for rx, ry in iterator:
                vimg[rx, ry] = np.max(self.datacube.data[rx, ry] * mask)

        elif "CoM" in detector_mode:
            ry_coord, rx_coord = np.meshgrid(
                np.arange(self.datacube.Q_Ny), np.arange(self.datacube.Q_Nx)
            )
            CoMx = np.zeros_like(vimg)
            CoMy = np.zeros_like(vimg)
            for rx, ry in iterator:
                ar = self.datacube.data[rx, ry] * mask
                tot_intens = np.sum(ar)
                CoMx[rx, ry] = np.sum(rx_coord * ar) / tot_intens
                CoMy[rx, ry] = np.sum(ry_coord * ar) / tot_intens

            CoMx -= np.mean(CoMx)
            CoMy -= np.mean(CoMy)

            if detector_mode == "CoM Magnitude":
                vimg = np.hypot(CoMx, CoMy)
            elif detector_mode == "CoM Angle":
                vimg = np.arctan2(CoMy, CoMx)
            elif detector_mode == "iCoM":
                raise NotImplementedError("Coming soon...")
            else:
                raise ValueError("Mode logic gone haywire!")

        else:
            raise ValueError("Oppsie")

    if scaling_mode == "Linear":
        new_view = vimg
    elif scaling_mode == "Log":
        new_view = np.log2(np.maximum(vimg, self.LOG_SCALE_MIN_VALUE))
    elif scaling_mode == "Square Root":
        new_view = np.sqrt(np.maximum(vimg, 0))
    else:
        raise ValueError("Mode not recognized")
    self.real_space_widget.setImage(new_view.T, autoLevels=True)


def update_diffraction_space_view(self, reset=False):
    scaling_mode = self.diff_scaling_group.checkedAction().text().replace("&", "")
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
        new_view = np.log2(np.maximum(DP, self.LOG_SCALE_MIN_VALUE))
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
    assert detector_shape in ["Point", "Rectangular", "Circle", "Annulus"]

    if self.datacube is None:
        return

    x, y = self.datacube.shape[2:]
    x0, y0 = x / 2, y / 2
    xr, yr = x / 10, y / 10

    # Remove existing detector
    if hasattr(self, "virtual_detector_point"):
        self.diffraction_space_widget.view.scene().removeItem(self.virtual_detector_point)
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
    if detector_shape == "Point":
        self.virtual_detector_point = pg_point_roi(self.diffraction_space_widget.getView())
        self.virtual_detector_point.sigRegionChanged.connect(
            self.update_real_space_view
        )

    elif detector_shape == "Rectangular":
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
            "Unknown detector shape! Got: {}".format(
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
