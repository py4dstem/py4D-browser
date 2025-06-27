import pyqtgraph as pg
import numpy as np
import py4DSTEM
from functools import partial
from PyQt5.QtWidgets import QApplication, QToolTip
from PyQt5 import QtCore
from PyQt5.QtGui import QCursor
import os


from py4D_browser.utils import (
    pg_point_roi,
    make_detector,
    complex_to_Lab,
    StatusBarWriter,
    DetectorShape,
    DetectorMode,
    DetectorInfo,
    RectangleGeometry,
    CircleGeometry,
    AnnulusGeometry,
    PointGeometry,
)


def get_diffraction_detector(self) -> DetectorInfo:
    """
    Get the current detector and its position on the diffraction view.
    Returns a DetectorInfo dictionary, which contains the shape and
    response mode of the detector and information on the selection
    it represents. The selection is described using one (or more) of
    the `slice`, `mask`, and `point` entries, depending on the detector
    type. The selections are expressed in data coordinates.
    """
    shape = DetectorShape(self.detector_shape_group.checkedAction().text())
    mode = DetectorMode(self.detector_mode_group.checkedAction().text())

    match shape:
        case DetectorShape.POINT:
            roi_state = self.virtual_detector_point.saveState()
            y0, x0 = roi_state["pos"]
            xc, yc = int(x0 + 1), int(y0 + 1)

            # Normalize coordinates
            xc = np.clip(xc, 0, self.datacube.Q_Nx - 1)
            yc = np.clip(yc, 0, self.datacube.Q_Ny - 1)

            return DetectorInfo(
                shape=shape,
                mode=mode,
                point=[xc, yc],
                geometry=PointGeometry(x=xc, y=yc),
            )

        case DetectorShape.RECTANGULAR:
            slices, _ = self.virtual_detector_roi.getArraySlice(
                self.datacube.data[0, 0, :, :].T,
                self.diffraction_space_widget.getImageItem(),
            )
            slice_y, slice_x = slices

            mask = np.zeros(self.datacube.Qshape, dtype=np.bool_)
            mask[slice_x, slice_y] = True

            return DetectorInfo(
                shape=shape,
                mode=mode,
                slice=[slice_x, slice_y],
                mask=mask,
                geometry=RectangleGeometry(
                    xmin=slice_x.start,
                    xmax=slice_x.stop,
                    ymin=slice_y.start,
                    ymax=slice_y.stop,
                ),
            )
        case DetectorShape.CIRCLE:
            R = self.virtual_detector_roi.size()[0] / 2.0

            x0 = self.virtual_detector_roi.pos()[1] + R
            y0 = self.virtual_detector_roi.pos()[0] + R

            mask = make_detector(
                (self.datacube.Q_Nx, self.datacube.Q_Ny), "circle", ((x0, y0), R)
            )

            return DetectorInfo(
                shape=shape,
                mode=mode,
                mask=mask,
                geometry=CircleGeometry(x=x0, y=y0, R=R),
            )

        case DetectorShape.ANNULUS:
            inner_pos = self.virtual_detector_roi_inner.pos()
            inner_size = self.virtual_detector_roi_inner.size()
            R_inner = inner_size[0] / 2.0
            x0 = inner_pos[1] + R_inner
            y0 = inner_pos[0] + R_inner

            outer_size = self.virtual_detector_roi_outer.size()
            R_outer = outer_size[0] / 2.0

            if R_inner <= R_outer:
                R_inner -= 1

            mask = make_detector(
                (self.datacube.Q_Nx, self.datacube.Q_Ny),
                "annulus",
                ((x0, y0), (R_inner, R_outer)),
            )

            return DetectorInfo(
                shape=shape,
                mode=mode,
                mask=mask,
                geometry=AnnulusGeometry(x=x0, y=y0, R_inner=R_inner, R_outer=R_outer),
            )

        case _:
            raise ValueError("Detector could not be determined")


def get_virtual_image_detector(self) -> DetectorInfo:
    """
    Get the current detector and its position on the diffraction view.
    Returns a DetectorInfo dictionary, which contains the shape and
    response mode of the detector and information on the selection
    it represents. The selection is described using one (or more) of
    the `slice`, `mask`, and `point` entries, depending on the detector
    type. The selections are expressed in data coordinates.
    """
    shape = DetectorShape(self.rs_detector_shape_group.checkedAction().text())
    mode = DetectorMode(self.realspace_detector_mode_group.checkedAction().text())

    match shape:
        case DetectorShape.POINT:
            roi_state = self.real_space_point_selector.saveState()
            y0, x0 = roi_state["pos"]
            xc, yc = int(x0 + 1), int(y0 + 1)

            # Normalize coordinates
            xc = np.clip(xc, 0, self.datacube.R_Nx - 1)
            yc = np.clip(yc, 0, self.datacube.R_Ny - 1)

            return DetectorInfo(
                shape=shape,
                mode=mode,
                point=[xc, yc],
                geometry=PointGeometry(x=xc, y=yc),
            )

        case DetectorShape.RECTANGULAR:
            slices, _ = self.real_space_rect_selector.getArraySlice(
                np.zeros((self.datacube.Rshape)).T,
                self.real_space_widget.getImageItem(),
            )
            slice_y, slice_x = slices

            mask = np.zeros(self.datacube.Rshape, dtype=np.bool_)
            mask[slice_x, slice_y] = True

            return DetectorInfo(
                shape=shape,
                mode=mode,
                slice=[slice_x, slice_y],
                mask=mask,
                geometry=RectangleGeometry(
                    xmin=slice_x.start,
                    xmax=slice_x.stop,
                    ymin=slice_y.start,
                    ymax=slice_y.stop,
                ),
            )

        case _:
            raise ValueError("Detector could not be determined")


def update_real_space_view(self, reset=False):
    if self.datacube is None:
        return

    detector = self.get_diffraction_detector()

    # If a CoM method is checked, ensure linear scaling
    scaling_mode = self.vimg_scaling_group.checkedAction().text().replace("&", "")
    if (
        detector["mode"] in (DetectorMode.CoM, DetectorMode.CoMx, DetectorMode.CoMy)
        and scaling_mode != "Linear"
    ):
        self.statusBar().showMessage("Warning! Setting linear scaling for CoM image")
        self.vimg_scale_linear_action.setChecked(True)
        scaling_mode = "Linear"

    # We will branch through certain combinations of detector shape and mode.
    # If we happen across a special case that can be handled directly, we
    # compute vimg. If we don't encounter a special case, the image is calculated
    # in the next block using the mask
    vimg = None
    match detector["shape"]:
        case DetectorShape.RECTANGULAR:
            # Get slices corresponding to ROI
            slice_x, slice_y = detector["slice"]

            # update the label:
            self.diffraction_space_view_text.setText(
                f"Diffraction Slice: [{slice_x.start}:{slice_x.stop},{slice_y.start}:{slice_y.stop}]"
            )

            if detector["mode"] is DetectorMode.INTEGRATING:
                vimg = np.sum(self.datacube.data[:, :, slice_x, slice_y], axis=(2, 3))
            elif detector["mode"] is DetectorMode.MAXIMUM:
                vimg = np.max(self.datacube.data[:, :, slice_x, slice_y], axis=(2, 3))

        case DetectorShape.CIRCLE:
            # This has no direct methods, so vimg will be made with mask
            circle_geometry: CircleGeometry = detector["geometry"]
            self.diffraction_space_view_text.setText(
                f"Diffraction Circle: Center ({circle_geometry['x']:.0f},{circle_geometry['y']:.0f}), Radius {circle_geometry['R']:.0f}"
            )

        case DetectorShape.ANNULUS:
            # No direct computation, so vimg gets made with mask
            annulus_geometry: AnnulusGeometry = detector["geometry"]

            self.diffraction_space_view_text.setText(
                f"Diffraction Annulus: Center ({annulus_geometry['x']:.0f},{annulus_geometry['y']:.0f}), Radii ({annulus_geometry['R_inner']:.0f},{annulus_geometry['R_outer']:.0f})"
            )

        case DetectorShape.POINT:
            xc, yc = detector["point"]
            vimg = self.datacube.data[:, :, xc, yc]

            self.diffraction_space_view_text.setText(f"Diffraction: Point [{xc},{yc}]")

        case _:
            raise ValueError("Detector shape not recognized")

    if vimg is None:
        mask = detector["mask"]

        # Debug mode for displaying the mask
        if "MASK_DEBUG" in os.environ:
            self.set_diffraction_image(mask.astype(np.float32), reset=reset)
            return

        mask = mask.astype(np.float32)
        vimg = np.zeros((self.datacube.R_Nx, self.datacube.R_Ny))
        iterator = py4DSTEM.tqdmnd(
            self.datacube.R_Nx,
            self.datacube.R_Ny,
            file=StatusBarWriter(self.statusBar()),
            mininterval=0.1,
        )

        if detector["mode"] is DetectorMode.INTEGRATING:
            for rx, ry in iterator:
                vimg[rx, ry] = np.sum(self.datacube.data[rx, ry] * mask)

        elif detector["mode"] is DetectorMode.MAXIMUM:
            for rx, ry in iterator:
                vimg[rx, ry] = np.max(self.datacube.data[rx, ry] * mask)

        elif detector["mode"] in (
            DetectorMode.CoM,
            DetectorMode.CoMx,
            DetectorMode.CoMy,
            DetectorMode.ICOM,
        ):
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

            if detector["mode"] is DetectorMode.CoM:
                vimg = CoMx + 1.0j * CoMy
            elif detector["mode"] is DetectorMode.CoMx:
                vimg = CoMx
            elif detector["mode"] is DetectorMode.CoMy:
                vimg = CoMy
            elif detector["mode"] is DetectorMode.ICOM:
                dpc = py4DSTEM.process.phase.DPC(verbose=False)
                dpc.preprocess(
                    force_com_measured=[CoMx, CoMy],
                    plot_rotation=False,
                    plot_center_of_mass="",
                )
                dpc.reconstruct(max_iter=1, step_size=1)
                vimg = dpc.object_phase
            else:
                raise ValueError("Mode logic gone haywire!")

        else:
            raise ValueError("Oopsie")

    self.set_virtual_image(vimg, reset=reset)


def set_virtual_image(self, vimg, reset=False):
    self.unscaled_realspace_image = vimg
    self._render_virtual_image(reset=reset)


def _render_virtual_image(self, reset=False):
    vimg = self.unscaled_realspace_image

    # for 2D images, use the scaling set by the user
    # for RGB (3D) images, always scale linear
    if np.isrealobj(vimg):
        scaling_mode = self.vimg_scaling_group.checkedAction().text().replace("&", "")
        assert scaling_mode in ["Linear", "Log", "Square Root"], scaling_mode

        if scaling_mode == "Linear":
            new_view = vimg.copy()
        elif scaling_mode == "Log":
            new_view = np.log2(np.maximum(vimg, self.LOG_SCALE_MIN_VALUE))
        elif scaling_mode == "Square Root":
            new_view = np.sqrt(np.maximum(vimg, 0))
        else:
            raise ValueError("Mode not recognized")

        auto_level = reset or self.realspace_rescale_button.latched

        self.real_space_widget.setImage(
            new_view.T,
            autoLevels=False,
            levels=(
                (
                    np.percentile(new_view, self.real_space_autoscale_percentiles[0]),
                    np.percentile(new_view, self.real_space_autoscale_percentiles[1]),
                )
                if auto_level
                else None
            ),
            autoRange=reset,
        )
    else:
        new_view = complex_to_Lab(vimg)
        self.real_space_widget.setImage(
            np.transpose(new_view, (1, 0, 2)),  # flip x/y but keep RGB ordering
            autoLevels=False,
            levels=(0, 1),
            autoRange=reset,
        )

    stats_text = [
        f"Min:\t{vimg.min():.5g}",
        f"Max:\t{vimg.max():.5g}",
        f"Mean:\t{vimg.mean():.5g}",
        f"Sum:\t{vimg.sum():.5g}",
        f"Std:\t{np.std(vimg):.5g}",
    ]

    for t, m in zip(stats_text, self.realspace_statistics_actions):
        m.setText(t)

    # Update FFT view
    self.unscaled_fft_image = None
    vimg_2D = vimg if np.isrealobj(vimg) else np.abs(vimg)
    fft_window = (
        np.hanning(vimg_2D.shape[0])[:, None] * np.hanning(vimg_2D.shape[1])[None, :]
    )
    if self.fft_source_action_group.checkedAction().text() == "Virtual Image FFT":
        fft = np.abs(np.fft.fftshift(np.fft.fft2(vimg_2D * fft_window))) ** 0.5
        levels = (np.min(fft), np.percentile(fft, 99.9))
        mode_switch = self.fft_widget_text.textItem.toPlainText() != "Virtual Image FFT"
        self.fft_widget_text.setText("Virtual Image FFT")
        self.fft_widget.setImage(
            fft.T, autoLevels=False, levels=levels, autoRange=mode_switch
        )
        self.fft_widget.getImageItem().setRect(0, 0, fft.shape[1], fft.shape[1])
        if mode_switch:
            # Need to autorange after setRect
            self.fft_widget.autoRange()
        self.unscaled_fft_image = fft
    elif (
        self.fft_source_action_group.checkedAction().text()
        == "Virtual Image FFT (complex)"
    ):
        fft = np.fft.fftshift(np.fft.fft2(vimg_2D * fft_window))
        levels = (np.min(np.abs(fft)), np.percentile(np.abs(fft), 99.9))
        mode_switch = self.fft_widget_text.textItem.toPlainText() != "Virtual Image FFT"
        self.fft_widget_text.setText("Virtual Image FFT")
        fft_img = complex_to_Lab(
            fft.T,
            amin=levels[0],
            amax=levels[1],
            ab_scale=128,
            gamma=0.5,
        )
        self.fft_widget.setImage(
            fft_img,
            autoLevels=False,
            autoRange=mode_switch,
            levels=(0, 1),
        )

        self.fft_widget.getImageItem().setRect(0, 0, fft.shape[1], fft.shape[1])
        if mode_switch:
            # Need to autorange after setRect
            self.fft_widget.autoRange()
        self.unscaled_fft_image = fft


def update_diffraction_space_view(self, reset=False):
    if self.datacube is None:
        return

    detector = self.get_virtual_image_detector()

    match detector["shape"]:
        case DetectorShape.POINT:
            xc, yc = detector["point"]

            self.real_space_view_text.setText(f"Virtual Image: Point [{xc},{yc}]")

            DP = self.datacube.data[xc, yc]

        case DetectorShape.RECTANGULAR:
            slice_x, slice_y = detector["slice"]

            self.real_space_view_text.setText(
                f"Virtual Image: Slice [{slice_x.start}:{slice_x.stop},{slice_y.start}:{slice_y.stop}]"
            )

            match detector["mode"]:
                case DetectorMode.INTEGRATING:
                    DP = np.sum(self.datacube.data[slice_x, slice_y], axis=(0, 1))
                case DetectorMode.MAXIMUM:
                    DP = np.max(self.datacube.data[slice_x, slice_y], axis=(0, 1))
                case _:
                    raise ValueError("Unsupported detector response")

        case _:
            raise ValueError("Unsupported detector shape...")

    self.set_diffraction_image(DP, reset=reset)


def set_diffraction_image(self, DP, reset=False):
    self.unscaled_diffraction_image = DP
    self._render_diffraction_image(reset=reset)


def _render_diffraction_image(self, reset=False):
    DP = self.unscaled_diffraction_image

    scaling_mode = self.diff_scaling_group.checkedAction().text().replace("&", "")
    assert scaling_mode in ["Linear", "Log", "Square Root"]

    if scaling_mode == "Linear":
        new_view = DP.copy()
    elif scaling_mode == "Log":
        new_view = np.log2(np.maximum(DP, self.LOG_SCALE_MIN_VALUE))
    elif scaling_mode == "Square Root":
        new_view = np.sqrt(np.maximum(DP, 0))
    else:
        raise ValueError("Mode not recognized")

    stats_text = [
        f"Min:\t{DP.min():.5g}",
        f"Max:\t{DP.max():.5g}",
        f"Mean:\t{DP.mean():.5g}",
        f"Sum:\t{DP.sum():.5g}",
        f"Std:\t{np.std(DP):.5g}",
    ]

    for t, m in zip(stats_text, self.diffraction_statistics_actions):
        m.setText(t)

    auto_level = reset or self.diffraction_rescale_button.latched

    self.diffraction_space_widget.setImage(
        new_view.T,
        autoLevels=False,
        levels=(
            (
                np.percentile(new_view, self.diffraction_autoscale_percentiles[0]),
                np.percentile(new_view, self.diffraction_autoscale_percentiles[1]),
            )
            if auto_level
            else None
        ),
        autoRange=reset,
    )

    if self.fft_source_action_group.checkedAction().text() == "EWPC":
        log_clip = np.maximum(1e-10, np.percentile(np.maximum(DP, 0.0), 0.1))
        fft = np.abs(np.fft.fftshift(np.fft.fft2(np.log(np.maximum(DP, log_clip)))))
        levels = (np.min(fft), np.percentile(fft, 99.9))
        mode_switch = self.fft_widget_text.textItem.toPlainText() != "EWPC"
        self.fft_widget_text.setText("EWPC")
        self.fft_widget.setImage(
            fft.T, autoLevels=False, levels=levels, autoRange=mode_switch
        )


def update_realspace_detector(self):
    # change the shape of the detector, then update the view

    detector_shape = (
        self.rs_detector_shape_group.checkedAction().text().replace("&", "")
    )
    assert detector_shape in ["Point", "Rectangular"], detector_shape

    main_pen = {"color": "g", "width": 6}
    handle_pen = {"color": "r", "width": 9}
    hover_pen = {"color": "c", "width": 6}
    hover_handle = {"color": "c", "width": 9}

    if self.datacube is None:
        x0, y0 = 0, 0
        xr, yr = 4, 4
    else:
        x, y = self.datacube.data.shape[2:]
        y0, x0 = x // 2, y // 2
        xr, yr = (np.minimum(x, y) / 10,) * 2

    # Remove existing detector
    if hasattr(self, "real_space_point_selector"):
        self.real_space_widget.view.scene().removeItem(self.real_space_point_selector)
        self.real_space_point_selector = None
    if hasattr(self, "real_space_rect_selector"):
        self.real_space_widget.view.scene().removeItem(self.real_space_rect_selector)
        self.real_space_rect_selector = None

    # Point detector
    if detector_shape == "Point":
        self.real_space_point_selector = pg_point_roi(
            self.real_space_widget.getView(),
            center=(x0 - 0.5, y0 - 0.5),
            pen=main_pen,
            hoverPen=hover_pen,
        )
        self.real_space_point_selector.sigRegionChanged.connect(
            partial(self.update_diffraction_space_view, False)
        )

    # Rectangular detector
    elif detector_shape == "Rectangular":
        self.real_space_rect_selector = pg.RectROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)],
            [int(xr), int(yr)],
            pen=main_pen,
            handlePen=handle_pen,
            hoverPen=hover_pen,
            handleHoverPen=hover_handle,
        )
        self.real_space_widget.getView().addItem(self.real_space_rect_selector)
        self.real_space_rect_selector.sigRegionChangeFinished.connect(
            partial(self.update_diffraction_space_view, False)
        )

    else:
        raise ValueError("Unknown detector shape! Got: {}".format(detector_shape))

    self.update_diffraction_space_view(reset=True)


def update_diffraction_detector(self):
    # change the shape of the detector, then update the view

    detector_shape = self.detector_shape_group.checkedAction().text().strip("&")
    assert detector_shape in ["Point", "Rectangular", "Circle", "Annulus"]

    main_pen = {"color": "g", "width": 6}
    handle_pen = {"color": "r", "width": 9}
    hover_pen = {"color": "c", "width": 6}
    hover_handle = {"color": "c", "width": 9}

    if self.datacube is None:
        x0, y0 = 0, 0
        xr, yr = 4, 4
    else:
        x, y = self.datacube.data.shape[2:]
        y0, x0 = x // 2, y // 2
        xr, yr = (np.minimum(x, y) / 10,) * 2

    # Remove existing detector
    if hasattr(self, "virtual_detector_point"):
        self.diffraction_space_widget.view.scene().removeItem(
            self.virtual_detector_point
        )
        self.virtual_detector_point = None
    if hasattr(self, "virtual_detector_roi"):
        self.diffraction_space_widget.view.scene().removeItem(self.virtual_detector_roi)
        self.virtual_detector_roi = None
    if hasattr(self, "virtual_detector_roi_inner"):
        self.diffraction_space_widget.view.scene().removeItem(
            self.virtual_detector_roi_inner
        )
        self.virtual_detector_roi_inner = None
    if hasattr(self, "virtual_detector_roi_outer"):
        self.diffraction_space_widget.view.scene().removeItem(
            self.virtual_detector_roi_outer
        )
        self.virtual_detector_roi_outer = None

    # Point detector
    if detector_shape == "Point":
        self.virtual_detector_point = pg_point_roi(
            self.diffraction_space_widget.getView(),
            center=(x0 - 0.5, y0 - 0.5),
            pen=main_pen,
            hoverPen=hover_pen,
        )
        self.virtual_detector_point.sigRegionChanged.connect(
            partial(self.update_real_space_view, False)
        )

    # Rectangular detector
    elif detector_shape == "Rectangular":
        self.virtual_detector_roi = pg.RectROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)],
            [int(xr), int(yr)],
            pen=main_pen,
            handlePen=handle_pen,
            hoverPen=hover_pen,
            handleHoverPen=hover_handle,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )

    # Circular detector
    elif detector_shape == "Circle":
        self.virtual_detector_roi = pg.CircleROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)],
            [int(xr), int(yr)],
            pen=main_pen,
            handlePen=handle_pen,
            hoverPen=hover_pen,
            handleHoverPen=hover_handle,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )

    # Annular dector
    elif detector_shape == "Annulus":
        # Make outer detector
        self.virtual_detector_roi_outer = pg.CircleROI(
            [int(x0 - xr), int(y0 - yr)],
            [int(2 * xr), int(2 * yr)],
            pen=main_pen,
            handlePen=handle_pen,
            hoverPen=hover_pen,
            handleHoverPen=hover_handle,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi_outer)

        # Make inner detector
        self.virtual_detector_roi_inner = pg.CircleROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)],
            [int(xr), int(yr)],
            pen=main_pen,
            hoverPen=hover_pen,
            handlePen=handle_pen,
            handleHoverPen=hover_handle,
            movable=False,
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi_inner)

        # Connect size/position of inner and outer detectors
        self.virtual_detector_roi_outer.sigRegionChanged.connect(
            self.update_annulus_pos
        )
        self.virtual_detector_roi_outer.sigRegionChanged.connect(
            self.update_annulus_radii
        )
        self.virtual_detector_roi_inner.sigRegionChanged.connect(
            self.update_annulus_radii
        )

        # Connect to real space view update function
        self.virtual_detector_roi_outer.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )
        self.virtual_detector_roi_inner.sigRegionChangeFinished.connect(
            partial(self.update_real_space_view, False)
        )

    else:
        raise ValueError("Unknown detector shape! Got: {}".format(detector_shape))

    self.update_real_space_view(reset=True)


def set_diffraction_autoscale_range(self, percentiles, redraw=True):
    self.diffraction_autoscale_percentiles = percentiles
    self.settings.setValue("last_state/diffraction_autorange", list(percentiles))

    if redraw:
        self._render_diffraction_image(reset=False)


def set_real_space_autoscale_range(self, percentiles, redraw=True):
    self.real_space_autoscale_percentiles = percentiles
    self.settings.setValue("last_state/realspace_autorange", list(percentiles))

    if redraw:
        self._render_virtual_image(reset=False)


def nudge_real_space_selector(self, dx, dy):
    if (
        hasattr(self, "real_space_point_selector")
        and self.real_space_point_selector is not None
    ):
        selector = self.real_space_point_selector
    elif (
        hasattr(self, "real_space_rect_selector")
        and self.real_space_rect_selector is not None
    ):
        selector = self.real_space_rect_selector
    else:
        raise RuntimeError("Can't find the real space selector!")

    position = selector.pos()
    position[0] += dy
    position[1] += dx

    selector.setPos(position)


def nudge_diffraction_selector(self, dx, dy):
    if (
        hasattr(self, "virtual_detector_point")
        and self.virtual_detector_point is not None
    ):
        selector = self.virtual_detector_point
    elif (
        hasattr(self, "virtual_detector_roi") and self.virtual_detector_roi is not None
    ):
        selector = self.virtual_detector_roi
    elif (
        hasattr(self, "virtual_detector_roi_outer")
        and self.virtual_detector_roi_outer is not None
    ):
        selector = self.virtual_detector_roi_outer
    else:
        raise RuntimeError("Can't find the diffraction space selector!")

    position = selector.pos()
    position[0] += dy
    position[1] += dx

    selector.setPos(position)


def update_tooltip(self):
    modifier_keys = QApplication.queryKeyboardModifiers()

    if self.datacube is not None and self.isActiveWindow():
        global_pos = QCursor.pos()

        for scene, data in [
            (self.diffraction_space_widget, self.unscaled_diffraction_image),
            (self.real_space_widget, self.unscaled_realspace_image),
            (self.fft_widget, self.unscaled_fft_image),
        ]:
            pos_in_scene = scene.mapFromGlobal(QCursor.pos())
            if scene.getView().rect().contains(pos_in_scene):
                pos_in_data = scene.view.mapSceneToView(pos_in_scene)

                y = int(np.clip(np.floor(pos_in_data.x()), 0, data.shape[1] - 1))
                x = int(np.clip(np.floor(pos_in_data.y()), 0, data.shape[0] - 1))

                if np.isrealobj(data):
                    display_text = f"[{x},{y}]: {data[x,y]:.5g}"
                else:
                    display_text = f"[{x},{y}]: |z|={np.abs(data[x,y]):.5g}, ϕ={np.degrees(np.angle(data[x,y])):.5g}°"

                self.cursor_value_text.setText(display_text)


def update_annulus_pos(self):
    """
    Function to keep inner and outer rings of annulus aligned.
    """
    R_outer = self.virtual_detector_roi_outer.size().x() / 2
    R_inner = self.virtual_detector_roi_inner.size().x() / 2
    # Only outer annulus is draggable; when it moves, update position of inner annulus
    x0 = self.virtual_detector_roi_outer.pos().x() + R_outer
    y0 = self.virtual_detector_roi_outer.pos().y() + R_outer
    self.virtual_detector_roi_inner.setPos(x0 - R_inner, y0 - R_inner, update=False)


def update_annulus_radii(self):
    R_outer = self.virtual_detector_roi_outer.size().x() / 2
    R_inner = self.virtual_detector_roi_inner.size().x() / 2
    if R_outer < R_inner:
        x0 = self.virtual_detector_roi_outer.pos().x() + R_outer
        y0 = self.virtual_detector_roi_outer.pos().y() + R_outer
        self.virtual_detector_roi_outer.setSize(2 * R_inner + 6, update=False)
        self.virtual_detector_roi_outer.setPos(
            x0 - R_inner - 3, y0 - R_inner - 3, update=False
        )
