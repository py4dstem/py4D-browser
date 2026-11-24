from typing import Optional
import numpy as np
import pyqtgraph as pg
from pyqtgraph import opengl as gl
from PyQt5 import QtCore
from py4D_browser.utils import format_unit
from py4D_browser.utils import scale_image, complex_to_Lab, try_get_cmap
from py4D_browser.virtual_image_tabs import VirtualImageTab
from py4D_browser.volume_tabs import VolumeTab

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py4D_browser import DataViewer


def update_fft_view(self: "DataViewer", mode: Optional[str] = None):
    mode = mode or self.result_source_action_group.checkedAction().text()
    vimg = self.current_virtual_image
    if not hasattr(self, "_result_stack"):
        return

    if (
        mode == "Virtual Image FFT"
        and vimg is not None
        and self._visible_volume_tab() is not None
    ):
        fft_window = (
            np.hanning(vimg.shape[0])[:, None, None]
            * np.hanning(vimg.shape[1])[None, :, None]
            * np.hanning(vimg.shape[2])[None, None, :]
        )
        self._result_volume_fft = (
            np.abs(np.fft.fftshift(np.fft.fftn(vimg * fft_window))) ** 0.5
        )
        self._update_result_volume_display()
        self._result_stack.setCurrentIndex(1)  # show the 3D page
        self._enable_view_lock()
        return

    scale = _result_scale(self, mode)
    if scale is None:
        self._result_stack.setCurrentIndex(0)
        self._disable_view_lock()
        return

    if mode in ("Virtual Image FFT", "Virtual Image FFT (complex)"):
        vimg_2D = vimg if np.isrealobj(vimg) else np.abs(vimg)
        fft_window = (
            np.hanning(vimg_2D.shape[0])[:, None]
            * np.hanning(vimg_2D.shape[1])[None, :]
        )
        windowed = vimg_2D * fft_window
        if mode == "Virtual Image FFT (complex)":
            fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(windowed)))
        else:
            fft = np.abs(np.fft.fftshift(np.fft.fft2(windowed))) ** 0.5
        title = "Virtual Image FFT"
    else:
        DP = self.unscaled_diffraction_image
        log_clip = np.maximum(1e-10, np.percentile(np.maximum(DP, 0.0), 0.1))
        fft = np.abs(np.fft.fftshift(np.fft.fft2(np.log(np.maximum(DP, log_clip)))))
        title = "EWPC"

    reset = (
        self._result_stack.currentIndex() == 1
        or self.fft_widget_text.textItem.toPlainText() != title
        or self.unscaled_fft_image is None
        or self.unscaled_fft_image.shape != fft.shape
    )
    self.set_result_image(
        fft, reset=reset, title=title, pixel_size=scale[0], pixel_units=scale[1]
    )
    self._result_fft_mode = mode
    # Both axes span the same frequency range for isotropic input pixels.
    width = fft.shape[1]
    self.fft_widget.getImageItem().setRect(0, 0, width, width)
    if reset:
        self.fft_widget.autoRange()


def set_result_image(
    self: "DataViewer", vimg, reset=False, pixel_size=1.0, pixel_units="", title=""
):
    self._result_stack.setCurrentIndex(0)
    self._disable_view_lock()
    self._result_fft_mode = None
    self.unscaled_fft_image = vimg
    self.fft_widget_text.setText(title)
    if vimg is None:
        self.fft_widget.clear()
    else:
        self._render_result_image(reset=reset)

    self.set_scalebar("result", pixel_size, pixel_units)


def refresh_result_scale_bar(self: "DataViewer"):
    mode = self.result_source_action_group.checkedAction().text()
    scale = _result_scale(self, mode)
    if scale is not None:
        self.set_scalebar("result", *scale)


def _result_scale(self: "DataViewer", mode):
    if mode in ("Virtual Image FFT", "Virtual Image FFT (complex)"):
        image = self.current_virtual_image
        index = self.virtual_image_tab_widget.currentIndex()
        bar = self.real_space_scale_bar
        if index > 0:
            tab = self.virtual_image_tabs[index - 1]
            if not isinstance(tab, VirtualImageTab):
                return None
            bar = tab.scale_bar
    elif mode == "EWPC":
        image = self.unscaled_diffraction_image
        bar = self.diffraction_scale_bar
    else:
        return None

    if image is None or image.ndim != 2:
        return None

    # The result view uses the horizontal bin spacing for both display axes.
    units = format_unit(bar.units or "px")
    if units.endswith("⁻¹"):
        units = units[:-2]
    elif units.startswith("1/"):
        units = units[2:]
    else:
        units += "⁻¹"
    return 1.0 / (bar.pixel_size * image.shape[1]), units


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

    # ImageView resets its transform on every setImage call.
    scale = (1, vimg.shape[1] / vimg.shape[0]) if self._result_fft_mode else None

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
            scale=scale,
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
            scale=scale,
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
    self._render_result_volume(scaled, levels)


def _render_result_volume(
    self: "DataViewer",
    scaled: np.ndarray,
    levels,
):
    """Render a scaled 3D array into the result pane's 3D volume view, mapping
    values through the result-pane colormap to RGBA.

    Deliberately independent of _render_result_image: it never touches
    unscaled_fft_image or the 2D fft_widget, so the 2D→RGB path is unaffected.

    Voxel-size changes only update the geometry transform."""
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

    # Create the volume item on first render, update it thereafter.
    item = self._result_volume_item
    if item is None:
        self._result_volume_item = item = gl.GLVolumeItem(rgba)
        self.result_volume_view.addItem(item)
        extent = max(1.0 / size for size in self._visible_volume_tab().voxel_size)
        self.result_volume_view.setCameraPosition(distance=extent * 1.5)
    else:
        item.setData(rgba)

    self._apply_result_volume_geometry()


def _apply_result_volume_geometry(self: "DataViewer"):
    """Scale each FFT axis by its reciprocal-space bin spacing."""
    item = self._result_volume_item
    tab = self._visible_volume_tab()
    if item is None or tab is None:
        return
    shape = self._result_volume_fft.shape
    spacing = [1.0 / (n * size) for n, size in zip(shape, tab.voxel_size)]
    origin = [-(n // 2 + 0.5) * step for n, step in zip(shape, spacing)]
    item.resetTransform()
    item.scale(*spacing)
    item.translate(*origin, local=False)


def _on_volume_voxel_size_changed(self: "DataViewer", tab: "VolumeTab"):
    """Update reciprocal-space spacing when the active volume's z size changes."""
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
