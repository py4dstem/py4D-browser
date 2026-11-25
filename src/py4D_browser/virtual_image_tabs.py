"""
The virtual-image tab: an independent display pane a plugin can drive to show
its own image alongside the built-in virtual image, without overwriting it.

A :class:`VirtualImageTab` wraps a ``pyqtgraph.ImageView`` (exposed as
:attr:`image_view`) and its own
:class:`~py4D_browser.scalebar.ScaleBar`, and tracks any pyqtgraph ROIs and
annotations a plugin attaches to it. It is created and owned by the
:class:`~py4D_browser.main_window.DataViewer` (see
``create_virtual_image_tab`` / ``close_virtual_image_tab``), which is what
performs the actual add/remove of the tab in the pane and tears the tab down
(cleanly detaching its ROIs/annotations) when the user closes it.

Unlike the built-in virtual image, each tab carries its **own** display
settings — a Linear/Log/Square-Root scaling, an Autorange latch, and the
low/high autoscale percentiles — controlled from a bar pinned to the bottom of
the tab view (the same pattern the volume tabs use). A new tab starts by
matching the built-in pane's current settings (see
``create_virtual_image_tab``) and is then independent of both the built-in and
the other tabs. The bar is collapsible and starts **hidden** (so a fresh
tab's view owns the full height): a toggle button in the tab's bottom-left
corner — hugging the pane frame while collapsed, the bar's left edge while
shown — or the :attr:`controls_visible` property, brings the bar in and out
to reclaim the vertical space.

The tab's :attr:`widget` is the QTabWidget **page container** (the render view
plus the bottom bar); the ``pyqtgraph.ImageView`` itself is :attr:`image_view`.
"""

from typing import Optional

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

from py4D_browser.scalebar import ScaleBar
from py4D_browser.utils import (
    LatchingButton,
    SCALING_MODES,
    complex_to_Lab,
    format_unit,
    scale_image,
    try_get_cmap,
)


class VirtualImageTab(QtCore.QObject):
    """
    A single virtual-image tab.

    Instances are created by the ``DataViewer`` (via
    ``create_virtual_image_tab``) and returned to the plugin. A plugin drives
    its display through :meth:`set_image`, :meth:`set_scalebar`, and
    :meth:`add_roi` / :meth:`add_annotation`, and tears it down with
    :meth:`close` (or by the user closing the tab, which routes here via the
    viewer).

    A tab's appearance is settable both by the user (via the bottom bar: a
    Scaling selector, an Autorange latch, and low/high Range percentiles) and
    by the plugin, through the :attr:`scaling`, :attr:`autorange`,
    :attr:`autorange_percentiles`, and :meth:`set_levels` properties. The
    bottom bar is collapsible and starts hidden: a small toggle button in the
    tab's bottom-left corner (hugging the pane frame while the bar is
    collapsed, the bar's left edge while it is shown) or the
    :attr:`controls_visible` property brings the whole bar in and out to
    reclaim the space. All state lives in the tab's own widget; the viewer
    only keeps a reference for bookkeeping.

    Parameters
    ----------
    title : str
        The text shown on the tab's label.
    viewer : DataViewer
        The owning window; used to update the tab's label, to seed the tab's
        initial settings from the built-in pane, and to route :meth:`close`
        through the viewer's teardown path.
    """

    # Emitted (with the tab itself as the argument) after :meth:`set_image`.
    signal_data_changed = QtCore.pyqtSignal(object)

    def __init__(self, title: str, viewer: "DataViewer"):
        super().__init__()
        self._viewer = viewer
        self._title = title
        self._closed = False
        self._image: Optional[np.ndarray] = None

        # Tracked so the viewer can detach them cleanly on close.
        self._rois = []
        self._annotations = []

        # Per-tab display settings, seeded from the built-in pane so a new
        # tab matches whatever the user is currently looking at (falling back
        # to the built-in defaults when there is no owning viewer — e.g. a tab
        # constructed in isolation for testing).
        scaling, autorange, percentiles = self._builtin_settings()
        self._scaling = scaling
        self._autorange = autorange
        self._autorange_percentiles = percentiles

        # An independent render view, mirroring the built-in real-space
        # widget: menu disabled, and a scale bar anchored to the lower-right.
        self._image_view = pg.ImageView()
        self._image_view.getView().setMenuEnabled(False)

        self._scale_bar = ScaleBar(pixel_size=1, units="px", width=10)
        self._scale_bar.setParentItem(self._image_view.getView())
        self._scale_bar.anchor((1, 1), (1, 1), offset=(-40, -40))

        # The collapsible display-control strip (Scaling / Autorange / Range),
        # shown and hidden as a unit via :attr:`controls_visible`.
        self._controls = self._build_controls()

        # The page container: the render view fills the space, with the control
        # strip below it (hidden when collapsed, so the view reclaims the space).
        self._widget = QWidget()
        outer = QVBoxLayout(self._widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._image_view, 1)  # the view fills the remaining space
        outer.addWidget(self._controls)  # its natural height; hidden when collapsed

        # The toggle button is always visible and can never be lost: it sits
        # in the page's bottom-left corner (hugging the pane frame) while the
        # strip is collapsed, and at the strip's left edge (reading as its
        # handle) while it is shown. Clicking it — or setting
        # :attr:`controls_visible` — toggles the strip. The bar starts hidden,
        # so a fresh tab's view owns the full height until the bar is summoned.
        self._controls_visible = False
        self._controls.setVisible(False)
        self._controls_toggle = QToolButton(self._widget)
        self._controls_toggle.setAutoRaise(True)
        self._controls_toggle.setToolTip("Show display controls")
        self._controls_toggle.setArrowType(QtCore.Qt.DownArrow)
        self._controls_toggle.clicked.connect(self._toggle_controls)
        self._controls_toggle.raise_()
        self._position_toggle()
        # Keep the toggle anchored (corner or bar edge) as the page resizes.
        self._widget.installEventFilter(self)

    def _builtin_settings(self):
        """
        The display settings a fresh tab should start with: the built-in
        pane's current scaling, Autorange latch, and autoscale percentiles.
        Returns ``(scaling, autorange, (low, high))``.
        """
        scaling, autorange, percentiles = "Linear", True, (0.1, 99.9)
        viewer = self._viewer
        if viewer is None:
            return scaling, autorange, percentiles
        action = viewer.vimg_scaling_group.checkedAction()
        if action is not None:
            scaling = action.text().replace("&", "")
        autorange = viewer.realspace_rescale_button.latched
        # Set by set_real_space_autoscale_range during menu setup; fall back to
        # the browser default if it was never assigned (a custom saved value
        # that matches no preset leaves the attribute unset).
        percentiles = tuple(
            getattr(viewer, "real_space_autoscale_percentiles", (0.1, 99.9))
        )
        return scaling, autorange, percentiles

    def _build_controls(self) -> QWidget:
        """
        Build the strip of display controls (Scaling / Autorange / Range), a
        centered row of labelled controls mirroring the volume tabs' layout.

        This widget is shown/hidden as a unit via :attr:`controls_visible`; the
        button that drives it lives in the page corner (``_controls_toggle``),
        so this widget holds no button of its own. The controls start hidden.

        The widgets are constructed with their initial values **before** any
        handler is connected, so populating them does not trigger a re-render.
        """
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()

        layout.addWidget(QLabel("Scaling"))
        self._scaling_combo = QComboBox()
        self._scaling_combo.addItems(SCALING_MODES)
        self._scaling_combo.setCurrentText(self._scaling)
        layout.addWidget(self._scaling_combo)

        self._autorange_button = LatchingButton("Autorange", latched=self._autorange)
        layout.addWidget(self._autorange_button)

        layout.addWidget(QLabel("Range"))
        self._range_low_spin = QDoubleSpinBox()
        self._range_high_spin = QDoubleSpinBox()
        for spin in (self._range_low_spin, self._range_high_spin):
            spin.setRange(0.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
        self._range_low_spin.setValue(self._autorange_percentiles[0])
        self._range_high_spin.setValue(self._autorange_percentiles[1])
        layout.addWidget(self._range_low_spin)
        layout.addWidget(self._range_high_spin)

        layout.addStretch()

        # Wire the handlers now that the widgets hold their initial values.
        self._scaling_combo.currentTextChanged.connect(self._on_scaling_changed)
        self._autorange_button.activated.connect(self._on_autorange_changed)
        self._range_low_spin.valueChanged.connect(self._on_percentiles_changed)
        self._range_high_spin.valueChanged.connect(self._on_percentiles_changed)

        return panel

    ########## read-only surface ##########

    @property
    def widget(self):
        """
        The QTabWidget page backing this tab: a container holding the render
        view (:attr:`image_view`) on top and the display-control bar below.
        Use :attr:`image_view` to drive the ``pyqtgraph.ImageView`` directly.
        """
        return self._widget

    @property
    def image_view(self):
        """The ``pyqtgraph.ImageView`` that renders this tab's image."""
        return self._image_view

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value: str):
        self._title = value
        if not self._closed and self._viewer is not None:
            self._viewer._update_tab_title(self)

    @property
    def image(self):
        """The raw array last passed to :meth:`set_image` (or ``None``)."""
        return self._image

    @property
    def scale_bar(self):
        """The tab's own :class:`ScaleBar` object (prefer :meth:`set_scalebar`)."""
        return self._scale_bar

    @property
    def colormap(self):
        """
        The tab's active pyqtgraph colormap, a ``pg.ColorMap`` reconstructed
        from the tab's histogram gradient (``None`` if the gradient has no
        ticks).

        The histogram gradient is the single source of truth for what the tab
        renders (pyqtgraph's ``ImageView`` wires the image item to it), so the
        colormap is read back from there rather than from the image item's
        one-shot lookup table.

        Each tab keeps its own colormap. A new tab is seeded with the browser's
        default virtual-image colormap (the ``gui/realspace_colormap`` setting,
        falling back to ``"thermal"`` — see ``create_virtual_image_tab``). Set
        this property (a ``pg.ColorMap`` or a colormap name) to change the tab
        independently of the other panes.
        """
        ticks = (
            self._image_view.getHistogramWidget().gradient.saveState().get("ticks")
            or []
        )
        if not ticks:
            return None
        return pg.ColorMap([t[0] for t in ticks], [t[1] for t in ticks])

    @colormap.setter
    def colormap(self, value):
        """
        Set the tab's colormap. ``value`` may be a ``pyqtgraph.ColorMap`` or a
        colormap name (resolved via the browser's colormap lookup). ``None``
        is ignored.
        """
        if self._closed:
            raise RuntimeError("Cannot set the colormap of a closed virtual-image tab.")
        if value is None:
            return
        cmap = value
        if isinstance(cmap, str):
            cmap = try_get_cmap(cmap)
        if cmap is None:
            raise ValueError(f"Unknown colormap: {value!r}")
        # Set the colormap at the ImageView level so the tab's histogram
        # gradient — the single source of pyqtgraph for what the tab renders —
        # is updated. Setting it on the image item alone would leave the
        # gradient on pyqtgraph's default greyscale, and any gradient change
        # (e.g. the user adjusting the strip) would re-push greyscale over
        # the image. This is the same call the built-in real-space pane uses.
        self._image_view.setColorMap(cmap)

    @property
    def rois(self):
        """A copy of the list of ROIs attached via :meth:`add_roi`."""
        return list(self._rois)

    @property
    def annotations(self):
        """A copy of the list of annotations attached via :meth:`add_annotation`."""
        return list(self._annotations)

    ########## display settings ##########

    @property
    def scaling(self):
        """
        The tab's scaling mode: one of :data:`~py4D_browser.utils.SCALING_MODES`
        (``"Linear"``, ``"Log"``, ``"Square Root"``).
        """
        return self._scaling

    @scaling.setter
    def scaling(self, value: str):
        """
        Set the tab's scaling mode (one of ``"Linear"``, ``"Log"``,
        ``"Square Root"``) and re-render. Raises ``ValueError`` for an
        unknown mode.
        """
        if self._closed:
            raise RuntimeError("Cannot set the scaling of a closed virtual-image tab.")
        if value not in SCALING_MODES:
            raise ValueError(
                f"Unknown scaling mode: {value!r}; expected one of "
                f"{', '.join(repr(m) for m in SCALING_MODES)}"
            )
        self._scaling = value
        # Reflect in the combo without re-firing its handler (which would
        # render a second time).
        self._scaling_combo.blockSignals(True)
        self._scaling_combo.setCurrentText(value)
        self._scaling_combo.blockSignals(False)
        if self._image is not None:
            self._render(reset=True)

    @property
    def autorange(self):
        """
        Whether the tab's display range auto-re-derives from the data (the
        Autorange latch). When off, the range is frozen at whatever the last
        re-derive (or :meth:`set_levels`) produced.
        """
        return self._autorange

    @autorange.setter
    def autorange(self, value: bool):
        """
        Set the Autorange latch. Turning it on re-derives the range from the
        (scaled) data at the tab's percentiles; turning it off freezes the
        current range.
        """
        if self._closed:
            raise RuntimeError(
                "Cannot set the autorange of a closed virtual-image tab."
            )
        value = bool(value)
        self._autorange = value
        # Set the button's state directly: ``setChecked`` does not run the
        # button's click handler, so this does not double-render.
        self._autorange_button.latched = value
        self._autorange_button.setChecked(value)
        if self._image is not None and value:
            self._render(reset=False, auto_level=True)

    @property
    def autorange_percentiles(self):
        """
        The autoscale percentiles ``(low, high)`` in 0–100 used to re-derive
        the range when :attr:`autorange` is on.
        """
        return self._autorange_percentiles

    @autorange_percentiles.setter
    def autorange_percentiles(self, value):
        """
        Set the autoscale percentiles ``(low, high)`` (each in 0–100). If
        :attr:`autorange` is on, the range is re-derived from the scaled data
        at the new percentiles.
        """
        if self._closed:
            raise RuntimeError(
                "Cannot set the percentiles of a closed virtual-image tab."
            )
        try:
            low, high = float(value[0]), float(value[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                f"autorange_percentiles must be a (low, high) pair, " f"got {value!r}"
            ) from exc
        self._autorange_percentiles = (low, high)
        # Reflect in the spinboxes without re-firing their handlers.
        for spin, v in (
            (self._range_low_spin, low),
            (self._range_high_spin, high),
        ):
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
        if self._image is not None and self._autorange:
            self._render(reset=False, auto_level=True)

    @property
    def levels(self):
        """
        The current ``(low, high)`` display level range, or ``None`` before an
        image has been set.
        """
        if self._image is None:
            return None
        low, high = self._image_view.getHistogramWidget().getLevels()
        return (float(low), float(high))

    def set_levels(self, low: float, high: float):
        """
        Set the display level range (the mapping of the scaled data onto the
        colormap) and update the view. This freezes the range even while
        :attr:`autorange` is on (it overrides the next re-derive).
        """
        if self._closed:
            raise RuntimeError("Cannot set the levels of a closed virtual-image tab.")
        if self._image is None:
            raise RuntimeError("Cannot set the levels before an image has been set.")
        self._image_view.setLevels(float(low), float(high))

    ########## control bar ##########

    @property
    def controls_visible(self):
        """
        Whether this tab's bottom display-control bar (the Scaling / Autorange
        / Range row) is shown. ``False`` by default — the bar starts hidden,
        so a fresh tab's view owns the full height. When ``False`` the whole
        bar is out of the way and only the corner toggle button remains, so the
        bar can always be brought back. Each tab remembers its own state
        independently.
        """
        return self._controls_visible

    @controls_visible.setter
    def controls_visible(self, value: bool):
        """
        Show or hide this tab's bottom display-control bar. Hiding it removes
        the whole bar (the view reclaims the space), leaving only the corner
        toggle button, so the bar can always be brought back (by the user, or
        by setting this back to ``True``).
        """
        if self._closed:
            raise RuntimeError(
                "Cannot change the controls of a closed virtual-image tab."
            )
        value = bool(value)
        if value == self._controls_visible:
            return
        self._controls_visible = value
        self._controls.setVisible(value)
        # The arrow points where the controls will go on the next click: up to
        # collapse (they are shown), down to expand (they are hidden).
        if value:
            self._controls_toggle.setArrowType(QtCore.Qt.UpArrow)
            self._controls_toggle.setToolTip("Hide display controls")
            # The bar is back in front; keep the button above it.
            self._controls_toggle.raise_()
        else:
            self._controls_toggle.setArrowType(QtCore.Qt.DownArrow)
            self._controls_toggle.setToolTip("Show display controls")
        # Slide the button between the corner (collapsed) and the bar's left
        # edge (shown) to match the new state.
        self._position_toggle()

    def _toggle_controls(self):
        """Flip the display-control bar between shown and collapsed."""
        if self._closed:
            return
        self.controls_visible = not self._controls_visible

    def _position_toggle(self):
        """
        Anchor the toggle button to the page's bottom-left corner — flush
        against the pane frame — while the strip is collapsed, or to the
        strip's left edge (vertically centered on it, so it reads as the
        strip's handle) while the strip is shown.
        """
        size = self._controls_toggle.sizeHint()
        self._controls_toggle.resize(size)
        if self._controls_visible:
            # Part of the bar: its left edge, centered on the bar's height.
            bar = self._controls
            x = max(0, bar.x())
            y = max(0, int(bar.y() + (bar.height() - size.height()) / 2))
        else:
            # Alone in the page's bottom-left corner, hugging the frame.
            x = 0
            y = max(0, self._widget.height() - size.height())
        self._controls_toggle.move(x, y)

    def eventFilter(self, obj, event):
        # Re-anchor the corner button whenever the page is resized. (Must be
        # named exactly ``eventFilter`` — Qt dispatches the C++ virtual by
        # that name; an underscore-prefixed method is never called.)
        if obj is self._widget and event.type() == QtCore.QEvent.Resize:
            self._position_toggle()
        return super().eventFilter(obj, event)

    ########## display ##########

    def set_image(
        self,
        image,
        reset: bool = True,
        pixel_size: Optional[float] = None,
        pixel_units: Optional[str] = None,
    ):
        """
        Set the raw image displayed in this tab.

        Parameters
        ----------
        image : array_like
            A real 2D array, or a complex 2D array (rendered as an RGB
            Lab-derived image, mirroring the built-in virtual image).
        reset : bool
            If True, re-derive the display level range from the (scaled) data
            at the tab's percentiles and reset the view. If False, the range
            re-derives only when :attr:`autorange` is on; otherwise the
            current range is kept.
        pixel_size, pixel_units : optional
            If given, also update the tab's scale bar.
        """
        if self._closed:
            raise RuntimeError("Cannot set the image of a closed virtual-image tab.")

        self._image = image
        self._render(reset=reset)
        if pixel_size is not None or pixel_units is not None:
            self.set_scalebar(pixel_size, pixel_units)

        self.signal_data_changed.emit(self)

        # If this tab is the one currently on display, tell the result
        # machinery (and any listeners) that the visible virtual image
        # changed. The built-in path emits the same signal in
        # ``set_virtual_image``; this is the tab's equivalent.
        if self._viewer is not None and self._viewer.is_virtual_image_tab_visible(self):
            self._viewer.signal_current_virtual_image_changed.emit()

    def _render(self, reset: bool = False, auto_level: Optional[bool] = None):
        """
        Re-render the tab's image through its own scaling, with the display
        range re-derived (``auto_level``) or frozen (``auto_level`` is False).
        Mirrors the built-in ``_render_virtual_image`` (update_views.py),
        except it uses the tab's own :attr:`scaling` and
        :attr:`autorange_percentiles` instead of the viewer's controls.
        """
        image = self._image
        if image is None:
            return
        if auto_level is None:
            auto_level = reset or self._autorange

        # Complex images are rendered as a fixed-range Lab/RGB image; the
        # scaling modes never apply (same as the built-in).
        if np.iscomplexobj(image):
            new_view = complex_to_Lab(image)
            self._image_view.setImage(
                np.transpose(new_view, (1, 0, 2)),  # flip x/y but keep RGB ordering
                autoLevels=False,
                levels=(0, 1),
                autoRange=bool(reset),
            )
            return

        # Shared with the built-in pane (see ``scale_image`` in utils.py) so
        # the two can never drift apart.
        new_view = scale_image(np.asarray(image), self._scaling)
        levels = None
        if auto_level:
            # Percentiles are applied to the *scaled* data, matching the
            # built-in. (Guard against low > high so a mis-ordered pair still
            # renders.)
            p0, p1 = self._autorange_percentiles
            levels = (
                float(np.percentile(new_view, min(p0, p1))),
                float(np.percentile(new_view, max(p0, p1))),
            )
        self._image_view.setImage(
            new_view.T, autoLevels=False, levels=levels, autoRange=bool(reset)
        )

    def set_scalebar(
        self, pixel_size: Optional[float] = None, units: Optional[str] = None
    ):
        """
        Update this tab's scale bar. Either ``pixel_size`` and/or ``units``
        may be ``None`` to leave that field unchanged.
        """
        if self._closed:
            raise RuntimeError(
                "Cannot update the scale bar of a closed virtual-image tab."
            )
        if pixel_size is not None:
            self._scale_bar.pixel_size = pixel_size
        if units is not None:
            # Translate ASCII unit strings (e.g. "A") to Unicode for display,
            # matching the built-in scale bars.
            self._scale_bar.units = format_unit(units)
        if pixel_size is not None or units is not None:
            self._scale_bar.updateBar()
            if self._viewer.is_virtual_image_tab_visible(self):
                self._viewer.refresh_result_scale_bar()

    ########## bottom-bar handlers ##########

    def _on_scaling_changed(self, text: str):
        if self._closed or self._image is None:
            return
        if text == self._scaling:
            return
        self._scaling = text
        # A scaling change re-renders and re-derives, matching the built-in's
        # scaling actions (which render with ``reset=True``).
        self._render(reset=True)

    def _on_autorange_changed(self):
        if self._closed or self._image is None:
            return
        self._autorange = self._autorange_button.latched
        # The latch only re-derives when it turns on; unlatching freezes the
        # current range (no re-render), matching the built-in rescale button.
        if self._autorange:
            self._render(reset=False, auto_level=True)

    def _on_percentiles_changed(self, _value=None):
        if self._closed or self._image is None:
            return
        self._autorange_percentiles = (
            self._range_low_spin.value(),
            self._range_high_spin.value(),
        )
        # Percentiles only affect the range while autorange is latched.
        if self._autorange:
            self._render(reset=False, auto_level=True)

    ########## ROIs and annotations ##########

    def add_roi(self, roi):
        """
        Attach a pre-built pyqtgraph ROI (e.g. ``pg.RectROI``) to this tab's
        view, and track it so it is detached when the tab is closed. Returns
        the ROI for convenience.
        """
        if self._closed:
            raise RuntimeError("Cannot add a ROI to a closed virtual-image tab.")
        self._image_view.getView().addItem(roi)
        self._rois.append(roi)
        return roi

    def add_annotation(self, annotation):
        """
        Attach any pyqtgraph ``QGraphicsItem`` (line, text, path, ...) as an
        annotation on this tab's view, and track it so it is detached when the
        tab is closed. Returns the item for convenience.
        """
        if self._closed:
            raise RuntimeError(
                "Cannot add an annotation to a closed virtual-image tab."
            )
        self._image_view.getView().addItem(annotation)
        self._annotations.append(annotation)
        return annotation

    def detach_items(self):
        """
        Detach any tracked ROIs/annotations from this tab's view and stop
        tracking them. The items themselves are **not** deleted, so a plugin
        that wants to reuse one can do so. Safe to call more than once.
        """
        for item in list(self._rois) + list(self._annotations):
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)
            item.setParentItem(None)
        self._rois = []
        self._annotations = []

    ########## lifecycle ##########

    @property
    def closed(self):
        """True once :meth:`close` has run (or the tab has been closed)."""
        return self._closed

    def close(self):
        """
        Close this tab, cleaning up its ROIs, annotations, and widget.

        Delegates to the viewer's teardown path, which detaches the tab's
        tracked items, removes it from the pane, and frees the widget. Safe to
        call more than once (a no-op after the first close).
        """
        if self._closed:
            return
        self._viewer.close_virtual_image_tab(self)
