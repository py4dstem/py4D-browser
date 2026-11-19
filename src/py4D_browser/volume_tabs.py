"""
3D volume tabs: an independent display pane a plugin can drive to render a 3D
volume alongside the built-in virtual image, using pyqtgraph's OpenGL volume
rendering.

A :class:`VolumeTab` wraps a ``pyqtgraph.opengl.GLViewWidget`` (which provides
built-in mouse controls: drag to orbit, Ctrl+drag to pan, wheel to zoom), two
``GradientEditorItem`` s — a color and an alpha transfer function — and a
log-scale slider for the first-axis (z/stack) voxel size, all adjustable by
the user live. It is created and owned by the
:class:`~py4D_browser.main_window.DataViewer` (see ``create_volume_tab`` /
``close_virtual_image_tab``), which is what performs the actual add/remove of
the tab in the pane and tears the tab down when the user closes it.

Rendering is deliberately "raw": like a :class:`VirtualImageTab`, the volume
is not passed through the browser's Linear/Log/Square-Root scaling group; the
plugin's data is mapped through the tab's own transfer functions as-is.
"""

from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

import pyqtgraph as pg

from py4D_browser.utils import try_get_cmap

try:
    import pyqtgraph.opengl as gl

    _OPENGL_AVAILABLE = True
except ImportError:
    gl = None
    _OPENGL_AVAILABLE = False


class VolumeTab(QtCore.QObject):
    """
    A single 3D volume tab.

    Instances are created by the ``DataViewer`` (via ``create_volume_tab``)
    and returned to the plugin. A plugin drives its display through
    :meth:`set_volume` and :meth:`set_levels`, and tears it down with
    :meth:`close` (or by the user closing the tab, which routes here via the
    viewer). The user adjusts the appearance through the tab's color and
    alpha transfer-function editors (:attr:`color_editor`,
    :attr:`alpha_editor`) and the mouse (orbit/pan/zoom via the
    :attr:`view`). All state lives in the tab's own widget; the viewer only
    keeps a reference for bookkeeping.

    Parameters
    ----------
    title : str
        The text shown on the tab's label.
    viewer : DataViewer
        The owning window; used to update the tab's label and to route
        :meth:`close` through the viewer's teardown path.

    Raises
    ------
    RuntimeError
        If PyOpenGL is not installed (``pyqtgraph.opengl`` cannot be
        imported), since the tab cannot render without it.
    """

    # Log-scale mapping for the z-size slider: position `v` is
    # 10 ** (_Z_SLIDER_LOG_MIN + v / 100) — one step is 1/100 of a decade,
    # covering 10**-3 .. 10**3 (voxel sizes span orders of magnitude, so a
    # linear slider would be useless below ~1).
    _Z_SLIDER_MIN = 0
    _Z_SLIDER_MAX = 600
    _Z_SLIDER_LOG_MIN = -3.0

    # Emitted (with the tab itself as the argument) after :meth:`set_volume`.
    signal_data_changed = QtCore.pyqtSignal(object)

    def __init__(self, title: str, viewer: "DataViewer"):
        if not _OPENGL_AVAILABLE:
            raise RuntimeError(
                "Creating a 3D volume tab requires PyOpenGL, which is not "
                "installed. Install it with `pip install PyOpenGL`."
            )
        super().__init__()
        self._viewer = viewer
        self._title = title
        self._closed = False
        self._volume: Optional[np.ndarray] = None
        self._levels: Optional[tuple] = None
        self._voxel_size = (1.0, 1.0, 1.0)
        self._volume_item: Optional["gl.GLVolumeItem"] = None
        self._aids_sized_for: Optional[tuple] = None

        # Lay out the tab: the 3D OpenGL view fills the space, with a
        # fixed-height strip of transfer-function editors pinned to the
        # bottom (at its natural size; not user-resizable).
        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor("k")

        # The editors sit side by side, centered in the bottom strip, each
        # with its label to the left.
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        for label, editor_attr in (
            ("Color", "_color_widget"),
            ("Alpha", "_alpha_widget"),
        ):
            group = QHBoxLayout()
            group.addWidget(QLabel(label))
            widget = pg.GradientWidget(maxDim=60)
            group.addWidget(widget)
            setattr(self, editor_attr, widget)
            layout.addLayout(group)

        # A log-scale slider for the first-axis (z/stack) voxel size, with a
        # live readout: the common case is a stack of images whose x/y pixels
        # are equal size, so adjusting just the z size is what it takes to
        # scale the volume correctly.
        z_group = QHBoxLayout()
        z_group.addWidget(QLabel("z size"))
        z_slider = QSlider(QtCore.Qt.Horizontal)
        z_slider.setRange(self._Z_SLIDER_MIN, self._Z_SLIDER_MAX)
        z_slider.setFixedWidth(150)
        z_readout = QLabel()
        z_readout.setFixedWidth(50)
        z_group.addWidget(z_slider)
        z_group.addWidget(z_readout)
        layout.addLayout(z_group)
        layout.addStretch()

        self._z_slider = z_slider
        self._z_readout = z_readout
        z_slider.valueChanged.connect(self._on_z_slider_changed)
        self._sync_z_slider()  # initial position (1.0) + readout

        widget = QWidget()
        outer = QVBoxLayout(widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._view, 1)  # the view fills the remaining space
        outer.addWidget(panel)  # its natural height; not resizable

        self._widget = widget

        # Orientation aid in the 3D view. (Deliberately no grid: a
        # ``GLGridItem`` draws its lines in the z=0 plane, which — since the
        # volume is centered on the origin — would cut straight through the
        # middle of the volume. The axes are enough to read orientation.)
        self._axis = gl.GLAxisItem()
        self._view.addItem(self._axis)

        # The underlying editors (the widgets wrap them; __getattr__ on the
        # widget delegates, but we keep the items directly for the public
        # surface and signal connections).
        self._color_editor = self._color_widget.item
        self._alpha_editor = self._alpha_widget.item

        # Right-clicking a gradient editor pops up its colormap menu
        # (GradientEditorItem.showMenu). Only each tick's alpha value
        # matters in the alpha editor, so disable that menu there; the
        # color editor keeps its (useful) colormap/preset choices.
        self._alpha_editor.showMenu = lambda ev: None

        self._seed_editors()

        # Live transfer-function edits: throttle the expensive RGBA recompute
        # (a full LUT pass over the volume plus a 3D-texture re-upload) while
        # the user drags, and recompute immediately on release. Connect only
        # after seeding, so the seed edits don't schedule a recompute.
        self._recompute_timer = QtCore.QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(100)
        self._recompute_timer.timeout.connect(self._recompute_rgba)
        for editor in (self._color_editor, self._alpha_editor):
            editor.sigGradientChanged.connect(self._on_gradient_changed)
            editor.sigGradientChangeFinished.connect(self._on_gradient_finished)

    ########## read-only surface ##########

    @property
    def widget(self):
        """The tab's backing ``QWidget`` (3D view on top, editor strip below)."""
        return self._widget

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value: str):
        self._title = value
        if not self._closed and self._viewer is not None:
            self._viewer._update_tab_title(self)

    @property
    def volume(self):
        """The raw array last passed to :meth:`set_volume` (or ``None``)."""
        return self._volume

    @property
    def image(self):
        """
        The raw array last passed to :meth:`set_volume` (or ``None``).

        Aliased as ``image`` so the tab is interchangeable with a
        :class:`VirtualImageTab` in ``viewer.current_virtual_image`` and
        friends — for a volume tab this is the 3D volume.
        """
        return self._volume

    @property
    def levels(self):
        """The current ``(low, high)`` display level range (or ``None``)."""
        return self._levels

    @property
    def voxel_size(self):
        """
        The current ``(dx, dy, dz)`` physical voxel size (a 3-tuple of
        floats), last set via :meth:`set_volume`. Defaults to
        ``(1.0, 1.0, 1.0)``: unit, isotropic voxels. The volume and the
        orientation axes are rendered in these physical units. The tab's
        z-size slider live-adjusts the first component.
        """
        return self._voxel_size

    @property
    def view(self):
        """
        The ``pyqtgraph.opengl.GLViewWidget`` showing the volume.

        Built-in mouse controls: drag = orbit, Ctrl+drag = pan (view
        relative), middle-drag = pan (view upright), wheel = zoom, arrows =
        orbit. The camera can also be driven programmatically with
        ``setCameraPosition`` / ``setCameraParams`` / ``cameraParams``.
        """
        return self._view

    @property
    def color_editor(self):
        """
        The ``pyqtgraph`` ``GradientEditorItem`` for the color transfer
        function. Edit its ticks (or call ``setColorMap``) to change the
        volume's colors; only the tick colors matter (not their alpha).
        """
        return self._color_editor

    @property
    def alpha_editor(self):
        """
        The ``pyqtgraph`` ``GradientEditorItem`` for the alpha (opacity)
        transfer function. Edit its ticks to change the volume's transparency;
        in this editor only each tick's alpha value matters (and its right-click
        colormap menu is disabled, as there is no colormap to choose).
        """
        return self._alpha_editor

    @property
    def axis(self):
        """The ``pyqtgraph.opengl.GLAxisItem`` showing the x/y/z axes."""
        return self._axis

    ########## display ##########

    def set_volume(self, volume, reset: bool = True, voxel_size=None):
        """
        Set the scalar volume rendered in this tab.

        Parameters
        ----------
        volume : array_like
            A real 3D array (x, y, z).
        reset : bool
            If True, re-derive the display level range from the data, using
            the user's configured real-space autoscale percentiles (the same
            range the built-in virtual image uses). If False, keep the tab's
            current levels (deriving min/max on the very first call).
        voxel_size : float or (float, float, float), optional
            The physical size of a single voxel: a scalar for isotropic
            voxels, or a ``(dx, dy, dz)`` 3-tuple for anisotropic ones. All
            values must be finite and positive. The volume and the
            orientation axes are then rendered in these physical units
            (an anisotropic size stretches the volume accordingly), and the
            initial camera distance is scaled to the volume's physical
            extent. If omitted, the tab's current voxel size is kept
            (default ``(1.0, 1.0, 1.0)`` — unit cubes). The tab's z-size
            slider live-adjusts the first component.
        """
        if self._closed:
            raise RuntimeError("Cannot set the volume of a closed volume tab.")

        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError(
                f"A volume tab requires a 3D volume; "
                f"got an array with {volume.ndim} dimensions."
            )
        if np.iscomplexobj(volume):
            raise ValueError("A volume tab cannot render a complex volume.")

        if voxel_size is not None:
            self._voxel_size = self._parse_voxel_size(voxel_size)
        self._sync_z_slider()

        self._volume = volume
        if reset:
            p0, p1 = (
                self._viewer.real_space_autoscale_percentiles
                if self._viewer is not None
                else (0.0, 100.0)
            )
            self._levels = (
                float(np.percentile(volume, p0)),
                float(np.percentile(volume, p1)),
            )
        elif self._levels is None:
            self._levels = (float(volume.min()), float(volume.max()))
        self._recompute_rgba()

        self.signal_data_changed.emit(self)

        # If this tab is the one currently on display, tell the result
        # machinery (and any listeners) that the visible virtual image
        # changed. This mirrors ``VirtualImageTab.set_image``.
        if self._viewer is not None and self._viewer.is_virtual_image_tab_visible(self):
            self._viewer.signal_current_virtual_image_changed.emit()

    @staticmethod
    def _parse_voxel_size(voxel_size):
        """
        Normalize a ``voxel_size`` argument (a positive scalar, or a length-3
        sequence of positive numbers) into a ``(dx, dy, dz)`` float tuple,
        raising ``ValueError`` on anything else.
        """
        try:
            arr = np.asarray(voxel_size, dtype=np.float64).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "voxel_size must be a positive number (isotropic) or a "
                "3-tuple of positive numbers (dx, dy, dz); "
                f"got {voxel_size!r}."
            ) from exc
        if arr.size == 1:
            values = (float(arr[0]),) * 3
        elif arr.size == 3:
            values = tuple(float(v) for v in arr)
        else:
            raise ValueError(
                "voxel_size must be a positive number (isotropic) or a "
                f"3-tuple of positive numbers (dx, dy, dz); got {voxel_size!r}."
            )
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError(
                "voxel_size values must be finite and positive; " f"got {values!r}."
            )
        return values

    def set_levels(self, low: float, high: float):
        """
        Set the display level range used to map the volume through the
        transfer functions, and re-render.
        """
        if self._closed:
            raise RuntimeError("Cannot set the levels of a closed volume tab.")
        self._levels = (float(low), float(high))
        self._recompute_rgba()

    ########## internals ##########

    def _seed_editors(self):
        """
        Give a fresh tab sensible default transfer functions:

        - Color: the volume tab's default colormap (the
          ``gui/volume_colormap`` setting, default ``"thermal"`` — the same
          default as the built-in virtual image) if it resolves.
        - Alpha: transparent at the low end, opaque at the high end, so the
          volume is see-through in its faintest regions. (In the alpha
          editor, only each tick's alpha value matters.)
        """
        cmap = None
        if self._viewer is not None:
            cmap_name = self._viewer.settings.value("gui/volume_colormap", "thermal")
            cmap = try_get_cmap(cmap_name)
        if cmap is not None:
            self._color_editor.setColorMap(cmap)

        # Replace the editor's default black/white ticks with a
        # transparent→opaque ramp. (Tick colors must be QColors: pyqtgraph
        # stores them as-is and later calls QColor methods on them.)
        for tick in list(self._alpha_editor.ticks):
            self._alpha_editor.removeTick(tick, finish=False)
        self._alpha_editor.addTick(0.0, QtGui.QColor(255, 255, 255, 0), finish=False)
        self._alpha_editor.addTick(1.0, QtGui.QColor(255, 255, 255, 255), finish=False)

    def _on_gradient_changed(self, _editor=None):
        # Live edit in progress: recompute shortly after the last change
        # rather than once per drag event (the recompute is a full LUT pass
        # over the volume plus a 3D-texture re-upload).
        if self._closed:
            return
        if not self._recompute_timer.isActive():
            self._recompute_timer.start()

    def _on_gradient_finished(self, _editor=None):
        # The edit is done: recompute immediately.
        if self._closed:
            return
        self._recompute_timer.stop()
        self._recompute_rgba()

    def _recompute_rgba(self):
        """
        Map the volume through the color/alpha transfer functions into the
        4D ``(x, y, z, RGBA)`` ``np.ubyte`` array ``GLVolumeItem`` renders,
        and push it into the 3D view.
        """
        if self._closed or self._volume is None or self._levels is None:
            return
        low, high = self._levels
        span = high - low
        if span <= 0:
            span = 1.0
        norm = np.clip((self._volume - low) / span, 0.0, 1.0)
        idx = (norm * 255).astype(np.uint8)

        color_lut = self._color_editor.getLookupTable(256, alpha=False)
        alpha_lut = self._alpha_editor.getLookupTable(256, alpha=True)[:, 3]

        rgba = np.empty(self._volume.shape + (4,), dtype=np.ubyte)
        rgba[..., 0] = color_lut[idx, 0]
        rgba[..., 1] = color_lut[idx, 1]
        rgba[..., 2] = color_lut[idx, 2]
        rgba[..., 3] = alpha_lut[idx]

        shape = self._volume.shape
        if self._volume_item is None:
            # First render: build the GL item (this never uploads the 3D
            # texture; the upload happens at paint time).
            self._volume_item = gl.GLVolumeItem(rgba)
            self._view.addItem(self._volume_item)
            # Frame the volume's physical extent.
            extents = self._physical_extents(shape)
            self._view.setCameraPosition(distance=max(extents) * 1.5)
        else:
            self._volume_item.setData(rgba)
        # Place the item for the current voxel size (a new volume may have
        # a different shape, and the z-size slider can change it at any
        # time).
        self._apply_voxel_geometry()

    def _apply_voxel_geometry(self):
        """
        Place and scale the displayed volume for the current voxel size.

        The data spans voxel indices (0, 0, 0)..shape, so the item is
        scaled by the voxel sizes and the physical box is centered on the
        origin (so the built-in orbit control pivots around it). (Order:
        scale in local coords, then translate in parent coords -> transform
        T*S.) The orientation aids are kept in step with the volume's
        footprint.
        """
        if self._volume is None or self._volume_item is None:
            return
        shape = self._volume.shape
        self._volume_item.resetTransform()
        self._volume_item.scale(*self._voxel_size)
        self._volume_item.translate(
            -shape[0] * self._voxel_size[0] / 2,
            -shape[1] * self._voxel_size[1] / 2,
            -shape[2] * self._voxel_size[2] / 2,
            local=False,
        )
        self._size_aids(shape)

    ########## z-size slider ##########

    def _z_from_slider(self, value):
        """The first-axis voxel size encoded by a slider position (log scale)."""
        return 10.0 ** (self._Z_SLIDER_LOG_MIN + value / 100.0)

    def _on_z_slider_changed(self, value):
        # Live first-axis (z/stack) voxel-size adjustment: update the size
        # and re-place the displayed volume (transform only — a voxel-size
        # change does not alter the RGBA data, so no recompute is needed).
        z = self._z_from_slider(value)
        self._voxel_size = (z, self._voxel_size[1], self._voxel_size[2])
        self._z_readout.setText(f"{z:.4g}")
        self._apply_voxel_geometry()
        # Keep the result 3D FFT (if shown — "Virtual Image FFT" with this
        # volume displayed) in lockstep with this z-stretch. A no-op unless the
        # 3D result is on screen and this is the visible volume tab (see
        # _on_volume_voxel_size_changed).
        if self._viewer is not None:
            self._viewer._on_volume_voxel_size_changed(self)

    def _sync_z_slider(self):
        """
        Reflect the tab's current first-axis voxel size in the slider
        (clamped to its range) and the readout.
        """
        z = self._voxel_size[0]
        value = int(round((np.log10(z) - self._Z_SLIDER_LOG_MIN) * 100))
        value = min(max(value, self._Z_SLIDER_MIN), self._Z_SLIDER_MAX)
        self._z_slider.blockSignals(True)
        self._z_slider.setValue(value)
        self._z_slider.blockSignals(False)
        self._z_readout.setText(f"{z:.4g}")

    def _physical_extents(self, shape):
        """The volume's ``(dx, dy, dz)`` size in physical units."""
        return tuple(s * v for s, v in zip(shape, self._voxel_size))

    def _size_aids(self, shape):
        """Size the orientation axes to the volume's footprint."""
        key = (shape, self._voxel_size)
        if self._aids_sized_for == key:
            return
        extents = self._physical_extents(shape)
        # The axes are a fixed orientation aid at the origin; match their
        # length to the volume's longest extent so they stay visible.
        length = max(extents)
        self._axis.setSize(length, length, length)
        self._aids_sized_for = key

    ########## lifecycle ##########

    @property
    def closed(self):
        """True once :meth:`close` has run (or the tab has been closed)."""
        return self._closed

    def detach_items(self):
        """
        No-op: unlike a :class:`VirtualImageTab`, a volume tab tracks no ROIs
        or annotations. Provided so the viewer's ``close_virtual_image_tab``
        teardown path works uniformly for both tab kinds.
        """

    def close(self):
        """
        Close this tab, freeing its widget and GL resources.

        Delegates to the viewer's teardown path, which removes it from the
        pane and frees the widget. Safe to call more than once (a no-op after
        the first close).
        """
        if self._closed:
            return
        self._viewer.close_virtual_image_tab(self)
