from typing import Optional
import numpy as np
import pyqtgraph as pg
from pyqtgraph import opengl as gl
from PyQt5 import QtCore
from py4D_browser.utils import format_unit
from py4D_browser.utils import scale_image, complex_to_Lab, try_get_cmap
from py4D_browser.volume_tabs import VolumeTab

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py4D_browser import DataViewer


def update_fft_view(self: "DataViewer", mode: Optional[str] = None):
    # called via signals when the Result menu has an internal option chosen
    # TODO: architect this as a plugin as well!

    # This gets called when *any* internal option is picked and
    # the mode is read from which menu item is selected

    mode = mode or self.result_source_action_group.checkedAction().text()

    # Read the *currently visible* virtual image (built-in default tab or a
    # plugin tab) rather than always the built-in image, so the result pane
    # follows whatever is on display in the virtual-image pane.
    vimg = self.current_virtual_image
    DP = self.unscaled_diffraction_image

    if vimg is None:
        # No visible virtual image (yet); nothing to compute a result from.
        return

    # "Virtual Image FFT" is shape-aware: with a genuine 3D volume tab visible
    # it renders the 3D FFT; with a 2D image, the 2D FFT below.
    #
    # Keyed on the visible tab, not just vimg.ndim==3: a 2D RGB(A) image is
    # also ndim==3 and is a supported virtual-image shape, but it lives in the
    # built-in pane or an image tab, never a VolumeTab. Keying on ndim alone
    # would FFT the channel axis as a fake z-axis.
    if (
        mode == "Virtual Image FFT"
        and vimg.ndim == 3
        and self._visible_volume_tab() is not None
    ):
        # A genuine 3D volume FFT — a separate path from the 2D branches below,
        # which would render a 3D array as RGB. Never touches unscaled_fft_image
        # / fft_widget, so the existing 2D→RGB path is unaffected.
        vimg_3D = vimg if np.isrealobj(vimg) else np.abs(vimg)
        fft_window = (
            np.hanning(vimg_3D.shape[0])[:, None, None]
            * np.hanning(vimg_3D.shape[1])[None, :, None]
            * np.hanning(vimg_3D.shape[2])[None, None, :]
        )
        fft = np.abs(np.fft.fftshift(np.fft.fftn(vimg_3D * fft_window))) ** 0.5
        # **0.5 keeps the same display scaling as the 2D branch.

        # Store the (unscaled) FFT + reciprocal-space voxel size for re-render
        # by the scaling / rescale controls (see _update_result_volume_display).
        self._result_volume_fft = fft
        if self.datacube is not None:
            self._result_volume_px = (
                1.0 / self.datacube.calibration.get_R_pixel_size() / self.datacube.R_Ny
            )
        else:
            # No dataset → no calibration for a voxel size; use a unit voxel
            # (the z-stretch is applied by _render_result_volume).
            self._result_volume_px = 1.0
        self._update_result_volume_display()
        self._result_stack.setCurrentIndex(1)  # show the 3D page
        self._enable_view_lock()
        return

    if DP is None or self.datacube is None:
        # The 2D modes need a loaded dataset for their reciprocal-space scale;
        # skip (the 3D branch above already returned for a volume).
        return

    # A 2D-only mode can't act on a non-2D virtual image, so skip silently
    # (this runs on every image update, so a status message would be noisy).
    # EWPC doesn't use the virtual image and still updates.
    if vimg.ndim != 2 and mode in ("Virtual Image FFT", "Virtual Image FFT (complex)"):
        self._result_stack.setCurrentIndex(0)
        self._disable_view_lock()
        return

    # The 2D modes render into the 2D page; switch back if the pane is still on
    # the 3D page from a prior 3D FFT, and drop the view lock.
    if mode in ("Virtual Image FFT", "Virtual Image FFT (complex)", "EWPC"):
        self._result_stack.setCurrentIndex(0)
        self._disable_view_lock()

    if mode == "Virtual Image FFT":
        vimg_2D = vimg if np.isrealobj(vimg) else np.abs(vimg)
        fft_window = (
            np.hanning(vimg_2D.shape[0])[:, None]
            * np.hanning(vimg_2D.shape[1])[None, :]
        )

        fft = np.abs(np.fft.fftshift(np.fft.fft2(vimg_2D * fft_window))) ** 0.5
        mode_switch = self.fft_widget_text.textItem.toPlainText() != "Virtual Image FFT"
        self.set_result_image(
            fft,
            reset=mode_switch,
            title="Virtual Image FFT",
            pixel_size=(
                1.0 / self.datacube.calibration.get_R_pixel_size() / self.datacube.R_Ny
            ),
            pixel_units=format_unit(
                f"{self.datacube.calibration.get_R_pixel_units()}⁻¹"
            ),
        )
        self.fft_widget.getImageItem().setRect(0, 0, fft.shape[1], fft.shape[1])
        if mode_switch:
            # Need to autorange after setRect
            self.fft_widget.autoRange()
    elif mode == "Virtual Image FFT (complex)":
        vimg_2D = vimg if np.isrealobj(vimg) else np.abs(vimg)
        fft_window = (
            np.hanning(vimg_2D.shape[0])[:, None]
            * np.hanning(vimg_2D.shape[1])[None, :]
        )

        fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(vimg_2D * fft_window)))
        mode_switch = self.fft_widget_text.textItem.toPlainText() != "Virtual Image FFT"
        self.set_result_image(
            fft,
            reset=mode_switch,
            title="Virtual Image FFT",
            pixel_size=(
                1.0 / self.datacube.calibration.get_R_pixel_size() / self.datacube.R_Ny
            ),
            pixel_units=format_unit(
                f"{self.datacube.calibration.get_R_pixel_units()}⁻¹"
            ),
        )
        self.fft_widget.getImageItem().setRect(0, 0, fft.shape[1], fft.shape[1])
        if mode_switch:
            # Need to autorange after setRect
            self.fft_widget.autoRange()

    elif mode == "EWPC":
        log_clip = np.maximum(1e-10, np.percentile(np.maximum(DP, 0.0), 0.1))
        fft = np.abs(np.fft.fftshift(np.fft.fft2(np.log(np.maximum(DP, log_clip)))))
        mode_switch = self.fft_widget_text.textItem.toPlainText() != "EWPC"
        self.set_result_image(
            fft,
            reset=mode_switch,
            title="EWPC",
            pixel_size=(
                1.0 / self.datacube.calibration.get_Q_pixel_size() / self.datacube.Q_Ny
            ),
            pixel_units=format_unit(
                f"{self.datacube.calibration.get_Q_pixel_units()}⁻¹"
            ),
        )

    else:
        raise RuntimeError(
            f"The internal FFT view callback was triggered but {mode} is checked!"
        )


def set_result_image(
    self: "DataViewer", vimg, reset=False, pixel_size=1.0, pixel_units="", title=""
):
    self.unscaled_fft_image = vimg
    self.fft_widget_text.setText(title)
    self._render_result_image(reset=reset)

    self.set_scalebar("result", pixel_size, pixel_units)


def refresh_result_scale_bar(self: "DataViewer"):
    # The result scale bar is only set when the result view re-renders, but a
    # calibration change doesn't re-render it — so recompute it from the current
    # calibration and mode.
    if self.datacube is None:
        return

    mode = self.result_source_action_group.checkedAction().text()
    calib = self.datacube.calibration

    if mode in ("Virtual Image FFT", "Virtual Image FFT (complex)"):
        # A visible 3D volume tab renders the 3D FFT (its own scale); the 2D
        # result bar is not refreshed.
        if mode == "Virtual Image FFT" and self._visible_volume_tab() is not None:
            return
        pixel_size = 1.0 / calib.get_R_pixel_size() / self.datacube.R_Ny
        pixel_units = format_unit(f"{calib.get_R_pixel_units()}⁻¹")
    elif mode == "EWPC":
        pixel_size = 1.0 / calib.get_Q_pixel_size() / self.datacube.Q_Ny
        pixel_units = format_unit(f"{calib.get_Q_pixel_units()}⁻¹")
    else:
        # Plugin-owned result pane; it sets its own scale bar.
        return

    self.set_scalebar("result", pixel_size, pixel_units)


def _render_result_image(
    self: "DataViewer", reset=False, auto_level: Optional[bool] = None
):
    # While the 3D FFT page is showing, re-render the volume instead (the
    # scaling / rescale controls route through this function); the 2D widget and
    # the existing 2D→RGB path are left untouched.
    if self._result_volume_fft is not None and self._result_stack.currentIndex() == 1:
        self._update_result_volume_display()
        return

    vimg = self.unscaled_fft_image

    if vimg is None:
        # no image loaded yet (e.g. an autoscale request fired before the
        # first datacube was shown); nothing to render
        return

    # for 2D images, use the scaling set by the user
    # for RGB (3D) images, always scale linear
    if np.isrealobj(vimg):
        scaling_mode = self.result_scaling_group.checkedAction().text().replace("&", "")
        assert scaling_mode in ["Linear", "Log", "Square Root"], scaling_mode

        if scaling_mode == "Linear":
            new_view = vimg.copy()
        elif scaling_mode == "Log":
            new_view = np.log2(np.maximum(vimg, self.LOG_SCALE_MIN_VALUE))
        elif scaling_mode == "Square Root":
            new_view = np.sqrt(np.maximum(vimg, 0))
        else:
            raise ValueError("Mode not recognized")

        if auto_level is None:
            auto_level = reset or self.result_rescale_button.latched

        self.fft_widget.setImage(
            new_view.T,
            autoLevels=False,
            levels=(
                (
                    np.percentile(new_view, self.result_autoscale_percentiles[0]),
                    np.percentile(new_view, self.result_autoscale_percentiles[1]),
                )
                if auto_level
                else None
            ),
            autoRange=reset,
        )
    else:
        new_view = complex_to_Lab(
            vimg,
            amin=np.percentile(np.abs(vimg), self.result_autoscale_percentiles[0]),
            amax=np.percentile(np.abs(vimg), self.result_autoscale_percentiles[1]),
            ab_scale=128,
            gamma=0.5,
        )
        self.fft_widget.setImage(
            np.transpose(new_view, (1, 0, 2)),  # flip x/y but keep RGB ordering
            autoLevels=False,
            levels=(0, 1),
            autoRange=reset,
        )


def _update_result_volume_display(self: "DataViewer"):
    """Re-render the stored 3D FFT with the current result scaling and autoscale
    percentiles. Called by update_fft_view and by the scaling/rescale controls
    (which route through _render_result_image while the 3D page is showing)."""
    if self._result_volume_fft is None:
        return

    # Apply the result scaling group, as _render_result_image does.
    scaling_mode = self.result_scaling_group.checkedAction().text().replace("&", "")
    assert scaling_mode in ["Linear", "Log", "Square Root"], scaling_mode
    scaled = scale_image(self._result_volume_fft, scaling_mode)

    # No manual levels here; always derive them from the autoscale percentiles.
    levels = (
        float(np.percentile(scaled, self.result_autoscale_percentiles[0])),
        float(np.percentile(scaled, self.result_autoscale_percentiles[1])),
    )
    self._render_result_volume(scaled, levels, reciprocal_px=self._result_volume_px)


def _render_result_volume(
    self: "DataViewer",
    scaled: np.ndarray,
    levels,
    reciprocal_px: float = 1.0,
):
    """Render a scaled 3D array into the result pane's 3D volume view, mapping
    values through the result-pane colormap to RGBA.

    Deliberately independent of _render_result_image: it never touches
    unscaled_fft_image or the 2D fft_widget, so the 2D→RGB path is unaffected.

    The first axis (z) is stretched to match the visible volume tab via
    _apply_result_volume_geometry, so a z-stretch change re-applies only the
    (cheap) transform, not this recompute."""
    low, high = levels
    span = max(high - low, 1e-12)
    idx = (np.clip((scaled - low) / span, 0.0, 1.0) * 255).astype(np.uint8)

    # Same colormap setting as the 2D result pane (gui/fft_colormap).
    cmap = try_get_cmap(self.settings.value("gui/fft_colormap", "yellowy"))
    if cmap is None:
        cmap = pg.ColorMap([0, 1], [(0, 0, 0, 255), (255, 255, 255, 255)], mode="float")
    # nPts must be a keyword: getLookupTable(start, stop, nPts), so a positional
    # getLookupTable(256) would set start=256 and return an all-white table.
    color_lut = cmap.getLookupTable(nPts=256)

    # Opacity: a gamma=2 transparent→opaque ramp. A peaky FFT's bulk is dim; a
    # linear ramp renders that bulk at a high, near-constant opacity (a solid
    # blob). gamma>1 keeps the dim bulk mostly transparent so the bright
    # features (DC + Bragg peaks) stand out. Fixed default, not user-selectable.
    _t = np.arange(256, dtype=np.float32) / 255.0
    alpha_lut = (255.0 * (_t**2.0)).astype(np.uint8)

    rgba = np.empty(scaled.shape + (4,), dtype=np.ubyte)
    rgba[..., 0] = color_lut[idx, 0]
    rgba[..., 1] = color_lut[idx, 1]
    rgba[..., 2] = color_lut[idx, 2]
    rgba[..., 3] = alpha_lut[idx]

    self._result_volume_shape = scaled.shape

    # Create the volume item on first render, update it thereafter.
    item = self._result_volume_item
    if item is None:
        self._result_volume_item = item = gl.GLVolumeItem(rgba)
        self.result_volume_view.addItem(item)
        # Frame it, including the z-stretch (applied just below) in the extent.
        z_stretch = self._result_volume_z_stretch()
        extent = max(
            scaled.shape[0] * reciprocal_px * z_stretch,
            scaled.shape[1] * reciprocal_px,
            scaled.shape[2] * reciprocal_px,
        )
        self.result_volume_view.setCameraPosition(distance=extent * 1.5)
    else:
        item.setData(rgba)

    self._apply_result_volume_geometry()


def _result_volume_z_stretch(self: "DataViewer") -> float:
    """First-axis (z) stretch for the result FFT, from the visible volume tab's
    voxel size (live-adjustable by its z-size slider). 1.0 if none is visible."""
    tab = self._visible_volume_tab()
    if tab is not None:
        vs = tab.voxel_size
        if vs[1] > 0:
            return vs[0] / vs[1]
    return 1.0


def _apply_result_volume_geometry(self: "DataViewer"):
    """(Re-)apply the result volume's transform — center at the origin and
    stretch its first axis (z) by the visible volume tab's z-stretch — without
    recomputing its RGBA, so a z-size-slider drag is cheap."""
    item = self._result_volume_item
    if item is None:
        return
    shape = self._result_volume_shape
    z_stretch = self._result_volume_z_stretch()
    z_px = self._result_volume_px * z_stretch
    xy_px = self._result_volume_px
    item.resetTransform()
    item.scale(z_px, xy_px, xy_px)
    item.translate(
        -shape[0] * z_px / 2,
        -shape[1] * xy_px / 2,
        -shape[2] * xy_px / 2,
        local=False,
    )


def _on_volume_voxel_size_changed(self: "DataViewer", tab: "VolumeTab"):
    """Re-apply the result 3D FFT's transform when the visible volume tab's z-size
    slider changes, so it follows the tab's new z-stretch in lockstep. A no-op
    unless a 3D result is showing and this tab is the visible volume tab."""
    if self._result_volume_item is None or self._result_volume_fft is None:
        return
    if tab is not self._visible_volume_tab():
        return
    if self._result_stack.currentIndex() != 1:
        return
    self._apply_result_volume_geometry()


def _visible_volume_tab(self: "DataViewer"):
    """The currently-visible :class:`VolumeTab` (or ``None``). Used to lock the
    result pane's view direction to the tab's view and to read its z scaling."""
    index = self.virtual_image_tab_widget.currentIndex() - 1
    tabs = self.virtual_image_tabs
    if 0 <= index < len(tabs) and isinstance(tabs[index], VolumeTab):
        return tabs[index]
    return None


def _enable_view_lock(self: "DataViewer"):
    """Activate the view-direction lock with the visible volume tab's view, if
    one is visible (see ViewDirectionSync)."""
    tab = self._visible_volume_tab()
    if tab is not None:
        self._view_sync.set_active(True, tab.view)
        return
    self._disable_view_lock()


def _disable_view_lock(self: "DataViewer"):
    """Deactivate the result pane's view-direction lock (idempotent)."""
    self._view_sync.set_active(False)


class ViewDirectionSync(QtCore.QObject):
    """Keep the view *direction* (elevation/azimuth) of the result 3D view and
    a volume tab's view in sync, bidirectionally. pyqtgraph emits no
    camera-change signal, so this watches both views for mouse/wheel input and,
    after the event settles, copies the driver's elevation/azimuth to the peer.
    Zoom (distance), pan (center), and fov stay independent per view. No-op
    while inactive."""

    _SYNC_EVENTS = (
        QtCore.QEvent.MouseButtonPress,
        QtCore.QEvent.MouseButtonRelease,
        QtCore.QEvent.MouseMove,
        QtCore.QEvent.Wheel,
    )

    def __init__(self, result_view):
        super().__init__()
        self._result_view = result_view
        self._volume_view = None
        self._active = False
        self._driver = None
        self._syncing = False
        # Fixed for the sync's lifetime; a no-op until set_active(True, ...).
        result_view.installEventFilter(self)

    def eventFilter(self, obj, event):
        if (
            self._active
            and event.type() in self._SYNC_EVENTS
            and (obj is self._result_view or obj is self._volume_view)
        ):
            self._driver = obj
            # Defer until the view's camera has settled.
            QtCore.QTimer.singleShot(0, self._sync)
        return super().eventFilter(obj, event)

    def _sync(self):
        if not self._active or self._syncing or self._volume_view is None:
            return
        driver = self._driver
        self._driver = None
        if driver is None:
            return
        peer = self._result_view if driver is self._volume_view else self._volume_view
        if driver is peer:
            return
        params = driver.cameraParams()
        if "elevation" not in params or "azimuth" not in params:
            # Only the default ('euler') rotation method exposes these.
            return
        self._syncing = True
        try:
            peer.setCameraParams(
                elevation=params["elevation"], azimuth=params["azimuth"]
            )
        finally:
            self._syncing = False

    def set_active(self, active: bool, volume_view=None):
        """Enable/disable the sync, and (when enabling) point it at
        ``volume_view`` (swapping the event filter to the new view)."""
        old = self._volume_view
        if active and volume_view is not None and volume_view is not old:
            if old is not None:
                old.removeEventFilter(self)
            self._volume_view = volume_view
            volume_view.installEventFilter(self)
        self._active = active


def set_result_autoscale_range(self: "DataViewer", percentiles, redraw=True):
    self.result_autoscale_percentiles = percentiles
    self.settings.setValue("last_state/result_autorange", list(percentiles))

    if redraw:
        # apply the newly-set range immediately (even if the rescale button
        # is not latched)
        self._render_result_image(reset=False, auto_level=True)
