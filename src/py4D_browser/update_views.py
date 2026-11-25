from typing import Optional
import numpy as np
from tqdm import tqdm
import py4DSTEM
from py4DSTEM.process.utils import electron_wavelength_angstrom
from PyQt5.QtWidgets import QApplication, QToolTip
from PyQt5 import QtCore
from PyQt5.QtGui import QCursor
import os
from py4D_browser.utils import format_unit


from py4D_browser.utils import (
    complex_to_Lab,
    scale_image,
    StatusBarWriter,
    DetectorShape,
    DetectorMode,
    CircleGeometry,
    AnnulusGeometry,
    strtobool,
)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py4D_browser import DataViewer


def update_scalebars(self: "DataViewer"):
    """
    Update scale bars from the datacube calibration.
    """
    if self.datacube is not None:
        # Diffraction scale bar
        self.diffraction_scale_bar.pixel_size = (
            self.datacube.calibration.get_Q_pixel_size()
        )
        self.diffraction_scale_bar.units = format_unit(
            self.datacube.calibration.get_Q_pixel_units()
        )
        self.diffraction_scale_bar.updateBar()

        # Real space scale bar
        self.real_space_scale_bar.pixel_size = (
            self.datacube.calibration.get_R_pixel_size()
        )
        self.real_space_scale_bar.units = format_unit(
            self.datacube.calibration.get_R_pixel_units()
        )
        self.real_space_scale_bar.updateBar()


def update_real_space_view(self: "DataViewer", reset=False):
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
            elif detector["mode"] is DetectorMode.AVERAGE:
                vimg = np.mean(self.datacube.data[:, :, slice_x, slice_y], axis=(2, 3))

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
            self.set_diffraction_image(
                mask.astype(np.float32),
                reset=reset,
                pixel_size=self.datacube.calibration.get_Q_pixel_size(),
                pixel_units=format_unit(self.datacube.calibration.get_Q_pixel_units()),
            )
            return

        # The masked response is computed in slabs of scan positions so
        # peak memory stays bounded (one slab at a time) and each slab is
        # reduced with vectorized, multithreaded (BLAS) reductions instead
        # of a Python loop over every position. The number of rows per
        # slab (batch) is configurable in the GUI configuration file;
        # 32 is the default.
        try:
            batch_size = int(self.settings.value("gui/batch_size_num_rows", 32))
        except (TypeError, ValueError):
            batch_size = 32
        batch_size = max(batch_size, 1)

        mask = mask.astype(np.float32)
        n_pixels = float(np.sum(mask))
        R_Nx, R_Ny = self.datacube.R_Nx, self.datacube.R_Ny
        vimg = np.zeros((R_Nx, R_Ny))
        iterator = tqdm(
            range(0, R_Nx, batch_size),
            file=StatusBarWriter(self.statusBar()),
            mininterval=0.1,
        )

        if detector["mode"] in (
            DetectorMode.CoM,
            DetectorMode.CoMx,
            DetectorMode.CoMy,
            DetectorMode.ICOM,
        ):
            # The center of mass is a ratio of two contractions,
            # sum(data*mask*coord) / sum(data*mask). Precompute the
            # coordinate-weighted masks once so each slab needs only BLAS
            # matmuls, with no slab-sized temporaries.
            ry_coord, rx_coord = np.meshgrid(
                np.arange(self.datacube.Q_Ny), np.arange(self.datacube.Q_Nx)
            )
            mask_x = mask * rx_coord
            mask_y = mask * ry_coord

            CoMx = np.zeros_like(vimg)
            CoMy = np.zeros_like(vimg)
            for r0 in iterator:
                r1 = min(r0 + batch_size, R_Nx)
                slab = self.datacube.data[r0:r1]
                intensity = np.tensordot(mask, slab, axes=([0, 1], [2, 3]))
                CoMx[r0:r1] = (
                    np.tensordot(mask_x, slab, axes=([0, 1], [2, 3])) / intensity
                )
                CoMy[r0:r1] = (
                    np.tensordot(mask_y, slab, axes=([0, 1], [2, 3])) / intensity
                )

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
            for r0 in iterator:
                r1 = min(r0 + batch_size, R_Nx)
                slab = self.datacube.data[r0:r1]
                if detector["mode"] is DetectorMode.INTEGRATING:
                    vimg[r0:r1] = np.tensordot(mask, slab, axes=([0, 1], [2, 3]))
                elif detector["mode"] is DetectorMode.AVERAGE:
                    vimg[r0:r1] = (
                        np.tensordot(mask, slab, axes=([0, 1], [2, 3])) / n_pixels
                    )
                elif detector["mode"] is DetectorMode.MAXIMUM:
                    vimg[r0:r1] = np.max(np.where(mask, slab, -np.inf), axis=(2, 3))
                else:
                    raise ValueError("Oopsie")

    self.set_virtual_image(
        vimg,
        reset=reset,
        pixel_size=self.datacube.calibration.get_R_pixel_size(),
        pixel_units=format_unit(self.datacube.calibration.get_R_pixel_units()),
    )


def set_virtual_image(
    self: "DataViewer", vimg, reset=False, pixel_size=None, pixel_units=None
):
    self.unscaled_realspace_image = vimg
    self._render_virtual_image(reset=reset)
    if pixel_size is not None or pixel_units is not None:
        self.set_scalebar("real_space", pixel_size, pixel_units)
    self.signal_virtual_image_data_changed.emit()
    # The built-in virtual image is the default tab (index 0). If it is the
    # currently-visible tab, its update also changes the "current virtual
    # image" that the result machinery follows. (If a plugin tab is visible,
    # this update does not affect what's on screen, so no notification.)
    if self.virtual_image_tab_widget.currentIndex() == 0:
        self.signal_current_virtual_image_changed.emit()


def _render_virtual_image(
    self: "DataViewer", reset=False, auto_level: Optional[bool] = None
):
    vimg = self.unscaled_realspace_image

    if vimg is None:
        # no image loaded yet (e.g. an autoscale request fired before the
        # first datacube was shown); nothing to render
        return

    # for 2D images, use the scaling set by the user
    # for RGB (3D) images, always scale linear
    if np.isrealobj(vimg):
        scaling_mode = self.vimg_scaling_group.checkedAction().text().replace("&", "")
        assert scaling_mode in ["Linear", "Log", "Square Root"], scaling_mode

        # Shared with the per-tab controls (see ``scale_image`` in utils.py) so
        # the built-in pane and a plugin tab can never drift apart.
        new_view = scale_image(vimg, scaling_mode)

        if auto_level is None:
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

    # The status-bar statistics for the virtual image are updated via
    # ``signal_current_virtual_image_changed`` (see
    # ``_update_visible_virtual_image_statistics`` in main_window.py), which
    # ``set_virtual_image`` emits when the default tab is visible — so they
    # track whichever tab is on screen, not just this render. (Direct
    # re-renders, e.g. a scaling-mode change, don't alter the raw array the
    # stats describe, so no update is needed here.)


def update_diffraction_space_view(self: "DataViewer", reset=False):
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
                case DetectorMode.AVERAGE:
                    DP = np.mean(self.datacube.data[slice_x, slice_y], axis=(0, 1))
                case _:
                    raise ValueError("Unsupported detector response")

        case _:
            raise ValueError("Unsupported detector shape...")

    self.set_diffraction_image(
        DP,
        reset=reset,
        pixel_size=self.datacube.calibration.get_Q_pixel_size(),
        pixel_units=format_unit(self.datacube.calibration.get_Q_pixel_units()),
    )


def set_diffraction_image(
    self: "DataViewer", DP, reset=False, pixel_size=None, pixel_units=None
):
    self.unscaled_diffraction_image = DP
    self._render_diffraction_image(reset=reset)
    if pixel_size is not None or pixel_units is not None:
        self.set_scalebar("diffraction", pixel_size, pixel_units)
    self.signal_diffraction_data_changed.emit()


def _render_diffraction_image(
    self: "DataViewer", reset=False, auto_level: Optional[bool] = None
):
    DP = self.unscaled_diffraction_image

    if DP is None:
        # no image loaded yet (e.g. an autoscale request fired before the
        # first datacube was shown); nothing to render
        return

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

    if auto_level is None:
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


def set_scalebar(self: "DataViewer", view, pixel_size=None, units=None):
    """
    Set the scale bar of one of the image views to an explicit pixel size
    and units string.

    view is one of "diffraction", "real_space", or "result". Either
    ``pixel_size`` and/or ``units`` may be omitted (None) to leave that
    field unchanged; if both are omitted the bar is left as-is.

    This is the supported way for plugins to update a scale bar (in
    preference to touching the *_scale_bar objects directly); the image
    setters delegate their scale updates here.
    """
    view_map = {
        "diffraction": self.diffraction_scale_bar,
        "real_space": self.real_space_scale_bar,
        "result": self.fft_scale_bar,
    }
    try:
        scale_bar = view_map[view]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown view {view!r}; expected one of {sorted(view_map)}"
        ) from None

    if pixel_size is not None:
        scale_bar.pixel_size = pixel_size
    if units is not None:
        scale_bar.units = units
    if pixel_size is not None or units is not None:
        scale_bar.updateBar()


def set_diffraction_autoscale_range(self: "DataViewer", percentiles, redraw=True):
    self.diffraction_autoscale_percentiles = percentiles
    self.settings.setValue("last_state/diffraction_autorange", list(percentiles))

    if redraw:
        # apply the newly-set range immediately (even if the rescale button
        # is not latched)
        self._render_diffraction_image(reset=False, auto_level=True)


def set_real_space_autoscale_range(self: "DataViewer", percentiles, redraw=True):
    self.real_space_autoscale_percentiles = percentiles
    self.settings.setValue("last_state/realspace_autorange", list(percentiles))

    if redraw:
        # apply the newly-set range immediately (even if the rescale button
        # is not latched)
        self._render_virtual_image(reset=False, auto_level=True)


def update_tooltip(self: "DataViewer"):
    modifier_keys = QApplication.queryKeyboardModifiers()

    if self.isActiveWindow():
        global_pos = QCursor.pos()

        for scene, data in [
            (self.diffraction_space_widget, self.unscaled_diffraction_image),
            # Use the *visible* virtual-image pane (default or plugin tab) and
            # its data: a hidden tab shares the visible tab's screen rect, so
            # reading the default widget/data here would sample the wrong image.
            (self._visible_real_space_widget, self.current_virtual_image),
            (self.fft_widget, self.unscaled_fft_image),
        ]:
            # A volume tab's widget is a QWidget around a 3D OpenGL view,
            # not a 2D image scene: there's no 2D pixel to sample, so skip
            # it (and move on to the other panes).
            if getattr(scene, "getView", None) is None:
                continue
            if data is None or (scene is self.fft_widget and not scene.isVisible()):
                continue
            graphics_view = scene.ui.graphicsView
            pos = graphics_view.viewport().mapFromGlobal(global_pos)
            pos_in_scene = graphics_view.mapToScene(pos)
            if scene.getView().sceneBoundingRect().contains(pos_in_scene):
                pos_in_data = scene.getImageItem().mapFromScene(pos_in_scene)

                y = int(np.clip(np.floor(pos_in_data.x()), 0, data.shape[1] - 1))
                x = int(np.clip(np.floor(pos_in_data.y()), 0, data.shape[0] - 1))

                if np.isrealobj(data):
                    if (
                        QtCore.Qt.ControlModifier == modifier_keys
                        and strtobool(
                            self.settings.value("gui/concatenation_tooltip", "0")
                        )
                        and data.itemsize == 4
                    ):
                        val = data.view(np.uint32)[x, y]
                        analog = val & 0x3FFF
                        digital = (val & 0x3FFFC000) >> 14
                        gain = (val & 0x80000000) >> 31
                        reserved = (val & 0x40000000) >> 30
                        display_text = f"[{x},{y}]: {data.view(np.uint32)[x,y]:#041_b} (R{reserved} G{gain} D{digital} A{analog})"
                    else:
                        display_text = f"[{x},{y}]: {data[x,y]:.5g}"
                else:
                    display_text = f"[{x},{y}]: |z|={np.abs(data[x,y]):.5g}, ϕ={np.degrees(np.angle(data[x,y])):.5g}°"

                if scene is self.fft_widget and self._result_fft_mode:
                    rows, cols = data.shape[:2]
                    step = self.fft_scale_bar.pixel_size
                    distance = np.hypot((x - rows // 2) * cols / rows, y - cols // 2)
                    magnitude = distance * step
                    units = self.fft_scale_bar.units
                    if self._result_fft_mode == "EWPC":
                        if units == "mrad⁻¹" and self.datacube is not None:
                            try:
                                voltage = float(self.datacube.calibration["voltage"])
                            except (KeyError, TypeError, ValueError):
                                voltage = 0
                            if np.isfinite(voltage) and voltage > 0:
                                magnitude *= 1e3 * electron_wavelength_angstrom(voltage)
                                units = "Å"
                        display_text += f" r={magnitude:.3g} {units}"
                    else:
                        display_text += f" k={magnitude:.3g} {units}"
                        if magnitude > 0 and units in (
                            "Å⁻¹",
                            "nm⁻¹",
                            "px⁻¹",
                            "pixels⁻¹",
                        ):
                            display_text += f", d={1.0 / magnitude:.3g} {units[:-2]}"

                self.cursor_value_text.setText(display_text)
