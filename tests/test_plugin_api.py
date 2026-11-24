"""
Tests for the versioned plugin API (py4D_browser.plugin_api).

The pure version-resolution logic is tested without Qt; the API surface
and behavior tests use the session-scoped DataViewer fixture from
conftest.py.
"""

import numpy as np
import pytest
from PyQt5.QtWidgets import QWidget

from py4D_browser.plugin_api import (
    API_VERSION,
    PluginAPI,
    PluginAPI1,
    PluginAPIVersionError,
    SUPPORTED_API_VERSIONS,
    resolve_api_class,
)

########## Version resolution (pure logic, no Qt) ##########


@pytest.mark.parametrize(
    "requested",
    [
        (1, 3),  # older minor: still compatible (additive within a major)
        (1, 6),  # the latest 1.x
        (1, 3, 0),  # extra entries beyond (major, minor) are ignored
        [1, 3],  # any indexable (major, minor) pair works
    ],
)
def test_resolve_compatible_versions(requested):
    assert resolve_api_class(requested) is PluginAPI1
    assert resolve_api_class((1, 3)) is PluginAPI  # latest alias


@pytest.mark.parametrize(
    "requested",
    [
        (1, 7),  # minor ahead of the latest 1.x
        (1, 9),
        (2, 0),  # major not provided at all
        (99, 0),
    ],
)
def test_resolve_unsupported_versions_raises(requested):
    with pytest.raises(PluginAPIVersionError):
        resolve_api_class(requested)


@pytest.mark.parametrize(
    "requested",
    [
        1,
        (1,),
        (),
        "1.0",
        None,
        (None, None),
        {},
    ],
)
def test_resolve_bad_shapes_raise(requested):
    with pytest.raises(PluginAPIVersionError, match="api_version must be"):
        resolve_api_class(requested)


def test_error_message_names_both_versions():
    with pytest.raises(PluginAPIVersionError, match=r"v1\.9") as excinfo:
        resolve_api_class((1, 9))
    assert "v1.6" in str(excinfo.value)  # the version the browser provides


def test_registry_and_api_version_agree():
    assert API_VERSION == (1, 6)
    assert PluginAPI is SUPPORTED_API_VERSIONS[1][1]
    assert PluginAPI1.api_version == (1, 6)


########## API surface ##########


def test_api_exposes_v1_surface(viewer):
    api = PluginAPI(viewer)

    assert api.api_version == (1, 6)

    # datacube is readable (None: no dataset loaded)
    assert api.datacube is viewer.datacube

    # The API's signals are the viewer's underlying signals: connecting
    # through the API must fire when the viewer's signal is emitted.
    # (PyQt5 returns a fresh bound-signal proxy per attribute access, so
    # identity comparison is not meaningful here; check behavior instead.)
    for name in (
        "signal_diffraction_data_changed",
        "signal_virtual_image_data_changed",
        "signal_current_virtual_image_changed",
        "signal_datacube_changed",
    ):
        fired = []

        def handler():  # signals are no-arg; a plain function, not list.append
            fired.append(True)

        api_sig = getattr(api, name)
        viewer_sig = getattr(viewer, name)
        api_sig.connect(handler)
        try:
            viewer_sig.emit()
            assert fired == [True], name
        finally:
            api_sig.disconnect(handler)

    # pane setters, datacube replacement, result registration, view ROI
    # attach/detach, and detector getters are callable
    for attr in (
        "set_virtual_image",
        "set_diffraction_image",
        "set_result_image",
        "set_scalebar",
        "set_datacube",
        "register_result_callback",
        "add_diffraction_roi",
        "remove_diffraction_roi",
        "add_real_space_roi",
        "remove_real_space_roi",
        "get_diffraction_detector",
        "get_virtual_image_detector",
        "create_virtual_image_tab",
        "close_virtual_image_tab",
    ):
        assert callable(getattr(api, attr)), attr

    # v1.6 tab surface: no plugin tabs open yet, and the "current" virtual
    # image is the (empty) built-in one.
    assert api.virtual_image_tabs == []
    assert api.current_virtual_image is viewer.current_virtual_image
    # the settings record is None until a 2D view has data on display
    assert api.current_virtual_image_settings is None
    assert api.current_diffraction_image_settings is None

    # display-image shape accessors: None until an image is on display (no
    # dataset loaded), and never expose the raw arrays themselves.
    assert api.diffraction_image_shape is None
    assert api.real_space_image_shape is None

    # Qt plumbing
    assert api.qtapp is viewer.qtapp
    # qt_window is a dialog-parent widget, NOT the viewer itself: a general
    # plugin must not reach the viewer's full state through it.
    assert isinstance(api.qt_window, QWidget)
    assert api.qt_window is not viewer
    # ...but it is still a child of the main window.
    assert api.qt_window.window() is viewer
    assert api.status_bar is viewer.statusBar()
    assert api.settings is viewer.settings


def test_plugin_api_contract_is_stable(viewer):
    """
    The v1.6 compatibility contract, asserted by behavior, not by identity.

    This is the gate that protects existing 1.6 plugins. It checks that
    every name a 1.6 plugin may touch is present and usable — methods are
    callable, signals expose connect/disconnect/emit, the data and Qt
    accessors are readable, ``api_version`` is a concrete ``(major, minor)``
    pair, and ``datacube`` is read-only — but it deliberately makes NO
    identity or concrete-class assertions. It depends only on ``api`` (never
    on the viewer's internal attribute layout), so a future release may
    replace the implementation behind these names with a broker object that
    bridges the 1.6 surface to new internals, and this test must keep
    passing in that case.
    """
    api = PluginAPI(viewer)

    # api_version is a concrete (major, minor) pair of integers.
    assert isinstance(api.api_version, tuple)
    assert len(api.api_version) == 2
    assert all(isinstance(v, int) for v in api.api_version)

    # Every 1.6 method must be present and callable.
    methods = (
        "set_virtual_image",
        "set_diffraction_image",
        "set_result_image",
        "set_scalebar",
        "refresh_result_scale_bar",
        "set_datacube",
        "register_result_callback",
        "add_diffraction_roi",
        "remove_diffraction_roi",
        "add_real_space_roi",
        "remove_real_space_roi",
        "get_diffraction_detector",
        "get_virtual_image_detector",
        "create_virtual_image_tab",
        "close_virtual_image_tab",
        "create_volume_tab",
    )
    for name in methods:
        assert hasattr(api, name), f"missing method: {name}"
        assert callable(getattr(api, name)), f"method not callable: {name}"

    # Every 1.6 signal must expose the standard PyQt connection surface.
    signals = (
        "signal_diffraction_data_changed",
        "signal_virtual_image_data_changed",
        "signal_current_virtual_image_changed",
        "signal_datacube_changed",
    )
    for name in signals:
        sig = getattr(api, name)
        for op in ("connect", "disconnect", "emit"):
            assert callable(getattr(sig, op)), f"{name}.{op} not callable"

    # Data accessors and Qt plumbing must be present and readable
    # (reading must not raise, with or without a dataset loaded).
    for name in (
        "datacube",
        "virtual_image_tabs",
        "current_virtual_image",
        "current_virtual_image_settings",
        "current_diffraction_image_settings",
        "current_diffraction_image",
        "diffraction_image_shape",
        "real_space_image_shape",
        "qt_window",
        "qtapp",
        "status_bar",
        "settings",
    ):
        assert hasattr(api, name), f"missing attribute: {name}"
        getattr(api, name)  # reading must not raise

    # Behavioral invariants a 1.6 plugin relies on (duck-typed, no identity).
    assert isinstance(api.virtual_image_tabs, list)
    assert isinstance(api.qt_window, QWidget)
    with pytest.raises(AttributeError):
        api.datacube = object()


def test_datacube_assignment_is_rejected(viewer):
    api = PluginAPI(viewer)
    with pytest.raises(AttributeError, match="cannot be assigned"):
        api.datacube = object()


########## Datacube replacement (set_datacube) ##########


def _make_datacube():
    import py4DSTEM

    return py4DSTEM.DataCube(np.zeros((2, 2, 4, 4), dtype=np.float32))


def test_set_datacube_replaces_and_refreshes_by_default(viewer):
    api = PluginAPI(viewer)

    original = viewer.datacube
    fired = []
    handler = lambda: fired.append(True)

    cube = _make_datacube()
    api.signal_datacube_changed.connect(handler)
    try:
        api.set_datacube(cube)
        assert viewer.datacube is cube
        # refresh=True is the default: the load machinery ran, which
        # includes emitting signal_datacube_changed.
        assert fired == [True]
    finally:
        api.signal_datacube_changed.disconnect(handler)
        viewer.datacube = original


def test_set_datacube_refresh_false_skips_machinery(viewer):
    api = PluginAPI(viewer)

    original = viewer.datacube
    fired = []
    handler = lambda: fired.append(True)

    cube = _make_datacube()
    api.signal_datacube_changed.connect(handler)
    try:
        api.set_datacube(cube, refresh=False)
        # the cube is still swapped in ...
        assert viewer.datacube is cube
        # ... but the refresh machinery (view resets + signal) did not run.
        assert fired == []
    finally:
        api.signal_datacube_changed.disconnect(handler)
        viewer.datacube = original


########## Pane setters (behavior through the API object) ##########


def test_set_virtual_image(viewer):
    api = PluginAPI(viewer)
    vimg = np.arange(64, dtype=np.float32).reshape(8, 8)

    api.set_virtual_image(vimg, reset=True, pixel_size=0.5, pixel_units="nm")

    assert viewer.unscaled_realspace_image is vimg
    assert viewer.real_space_scale_bar.pixel_size == 0.5
    assert viewer.real_space_scale_bar.units == "nm"


def test_set_diffraction_image(viewer):
    api = PluginAPI(viewer)
    DP = np.arange(64, dtype=np.float32).reshape(8, 8)

    api.set_diffraction_image(DP, reset=True, pixel_size=0.25, pixel_units="mrad")

    assert viewer.unscaled_diffraction_image is DP
    assert viewer.diffraction_scale_bar.pixel_size == 0.25
    assert viewer.diffraction_scale_bar.units == "mrad"


def test_set_result_image(viewer):
    api = PluginAPI(viewer)
    fft = np.arange(64, dtype=np.float32).reshape(8, 8)

    api.set_result_image(
        fft, reset=True, pixel_size=2.0, pixel_units="nm^-1", title="test result"
    )

    assert viewer.unscaled_fft_image is fft
    assert viewer.fft_scale_bar.pixel_size == 2.0
    assert viewer.fft_scale_bar.units == "nm^-1"
    assert viewer.fft_widget_text.toPlainText() == "test result"


def test_set_scalebar_updates_each_view(viewer):
    api = PluginAPI(viewer)

    cases = [
        ("diffraction", viewer.diffraction_scale_bar),
        ("real_space", viewer.real_space_scale_bar),
        ("result", viewer.fft_scale_bar),
    ]
    for view_name, scale_bar in cases:
        api.set_scalebar(view_name, 1.5, "nm")
        assert scale_bar.pixel_size == 1.5
        assert scale_bar.units == "nm"


def test_set_scalebar_unknown_view_raises(viewer):
    api = PluginAPI(viewer)
    with pytest.raises(ValueError, match="Unknown view"):
        api.set_scalebar("not_a_view", 1.0, "nm")


########## View ROI attach/detach and image-shape accessors ##########


def test_image_shape_accessors_track_displayed_images(viewer):
    api = PluginAPI(viewer)

    dp = np.zeros((5, 7), dtype=np.float32)
    vimg = np.zeros((9, 3), dtype=np.float32)
    api.set_diffraction_image(dp, reset=True, pixel_size=1.0, pixel_units="")
    api.set_virtual_image(vimg, reset=True, pixel_size=1.0, pixel_units="")

    # The accessors expose the *shape* of the raw stored arrays, not the
    # arrays themselves (a plugin sizes ROIs against these; it must not
    # retain the large arrays).
    assert api.diffraction_image_shape == (5, 7)
    assert api.real_space_image_shape == (9, 3)


def test_add_remove_roi_on_built_in_views(viewer):
    import pyqtgraph as pg

    api = PluginAPI(viewer)

    diff_view = viewer.diffraction_space_widget.getView()
    rs_view = viewer.real_space_widget.getView()

    diff_item = pg.InfiniteLine(pos=2.0, pen=pg.mkPen("r"))
    rs_item = pg.InfiniteLine(pos=3.0, pen=pg.mkPen("g"))

    # attach: the API returns the (same) item, and it lands in the correct
    # built-in view's scene.
    assert api.add_diffraction_roi(diff_item) is diff_item
    assert diff_view.scene().items().count(diff_item) == 1
    assert api.add_real_space_roi(rs_item) is rs_item
    assert rs_view.scene().items().count(rs_item) == 1

    # each item is in its own view, not the other's.
    assert rs_view.scene().items().count(diff_item) == 0
    assert diff_view.scene().items().count(rs_item) == 0

    # detach: removed from the scene, but the item itself survives (not
    # deleted), so a plugin can re-attach it later.
    assert api.remove_diffraction_roi(diff_item) is diff_item
    assert diff_view.scene().items().count(diff_item) == 0
    assert api.remove_real_space_roi(rs_item) is rs_item
    assert rs_view.scene().items().count(rs_item) == 0

    # re-attach still works after a detach.
    assert api.add_diffraction_roi(diff_item) is diff_item
    assert diff_view.scene().items().count(diff_item) == 1
    api.remove_diffraction_roi(diff_item)


########## Autoscale (configured percentile range) ##########


@pytest.mark.parametrize(
    "set_image,set_range,rescale_button,widget,scaling_group",
    [
        (
            "set_virtual_image",
            "set_real_space_autoscale_range",
            "realspace_rescale_button",
            "real_space_widget",
            "vimg_scaling_group",
        ),
        (
            "set_diffraction_image",
            "set_diffraction_autoscale_range",
            "diffraction_rescale_button",
            "diffraction_space_widget",
            "diff_scaling_group",
        ),
        (
            "set_result_image",
            "set_result_autoscale_range",
            "result_rescale_button",
            "fft_widget",
            "result_scaling_group",
        ),
    ],
)
def test_autoscale_click_respects_configured_percentile_range(
    viewer, set_image, set_range, rescale_button, widget, scaling_group
):
    # Both autoscale entry points — the Autorange menu (via the
    # ``set_*_autoscale_range`` setters) and the status-bar "Autorange"
    # buttons — must apply the user's configured percentile range on that
    # click, not pyqtgraph's min/max autorange.
    api = PluginAPI(viewer)
    button = getattr(viewer, rescale_button)
    item = getattr(viewer, widget).getImageItem()

    # the display levels are derived from the *displayed* (possibly
    # log/sqrt-scaled) view; use Linear so the expected levels are the raw
    # percentiles of the data
    for action in getattr(viewer, scaling_group).actions():
        if action.text().replace("&", "") == "Linear":
            action.setChecked(True)
            break

    def make_image(scale=1.0):
        # a smooth ramp: different percentiles give different ranges, and
        # none of them coincide with the data min/max
        im = np.arange(256, dtype=np.float32).reshape(16, 16) * scale
        return im

    p0, p1 = 5.0, 95.0
    im = make_image()
    getattr(api, set_image)(im, reset=True)

    # (1) choosing a new range in the Autorange menu applies it immediately,
    # even though the rescale button is not latched
    button.latched = False
    getattr(viewer, set_range)((p0, p1))
    expected = (float(np.percentile(im, p0)), float(np.percentile(im, p1)))
    levels = np.asarray(item.levels)
    assert np.allclose(levels, expected)
    # not the data min/max (what pyqtgraph's autoLevels would pick) ...
    assert not np.allclose(levels, (float(im.min()), float(im.max())))
    # ... and not the factory-default (0.1, 99.9) range either
    default = (float(np.percentile(im, 0.1)), float(np.percentile(im, 99.9)))
    assert not np.allclose(levels, default)

    # (2) a new image displayed with reset=False does not re-derive the
    # levels (a momentary state change is not an autoscale request)
    im2 = make_image(scale=10.0)
    getattr(api, set_image)(im2, reset=False)
    assert np.allclose(np.asarray(item.levels), expected)

    # (3) pressing the "Autorange" button re-derives the levels from the
    # configured range, even on a momentary (un-latched) press
    button.activated.emit()
    expected2 = (
        float(np.percentile(im2, p0)),
        float(np.percentile(im2, p1)),
    )
    assert np.allclose(np.asarray(item.levels), expected2)


########## Signal wiring through the API object ##########


def test_signals_fire_through_api(viewer):
    api = PluginAPI(viewer)

    fired = []

    def handler():
        fired.append(True)

    api.signal_virtual_image_data_changed.connect(handler)
    try:
        api.set_virtual_image(np.zeros((4, 4), dtype=np.float32), reset=True)
        assert fired == [True]
    finally:
        api.signal_virtual_image_data_changed.disconnect(handler)


########## Virtual-image tabs (v1.6) ##########


def test_create_virtual_image_tab(viewer):
    from PyQt5.QtWidgets import QTabBar

    api = PluginAPI(viewer)
    tab_widget = viewer.virtual_image_tab_widget

    # before: only the default tab exists and the tab bar is hidden
    assert tab_widget.count() == 1
    assert tab_widget.tabBar().isHidden()

    tab = api.create_virtual_image_tab("Test")
    try:
        # default tab (index 0) + the new plugin tab (index 1)
        assert tab_widget.count() == 2
        assert tab_widget.widget(0) is viewer.real_space_widget
        assert tab_widget.widget(1) is tab.widget

        # the tab is labeled with its title. (Guards against the macOS
        # native-tab rendering path dropping the *label* of a tab that also
        # carries a custom close button — see the stylesheet applied to the
        # tab bar in ``setup_views``.)
        assert tab_widget.tabText(1) == "Test"

        # more than one tab -> the tab bar becomes visible
        assert not tab_widget.tabBar().isHidden()

        # the (unclosable) default tab has no close button ...
        assert tab_widget.tabBar().tabButton(0, QTabBar.RightSide) is None
        # ... but the plugin tab does
        assert tab_widget.tabBar().tabButton(1, QTabBar.RightSide) is not None

        # the tab is tracked by the API, but not auto-selected: the default
        # tab stays on display until the user switches to the new tab.
        assert api.virtual_image_tabs == [tab]
        assert tab_widget.currentIndex() == 0
    finally:
        api.close_virtual_image_tab(tab)

    # after closing: back to a single (default) tab, tab bar hidden again
    assert tab_widget.count() == 1
    assert tab_widget.tabBar().isHidden()
    assert api.virtual_image_tabs == []


def test_default_tab_cannot_be_closed(viewer):
    api = PluginAPI(viewer)
    tab_widget = viewer.virtual_image_tab_widget

    assert tab_widget.count() == 1
    assert api.virtual_image_tabs == []

    # a close request on the default tab (index 0) is ignored
    viewer._on_tab_close_requested(0)

    assert tab_widget.count() == 1
    assert tab_widget.widget(0) is viewer.real_space_widget
    # and the default tab never appears in the plugin-facing list
    assert api.virtual_image_tabs == []


def test_ctrl_tab_switches_virtual_image_tabs(viewer):
    api = PluginAPI(viewer)
    tab_widget = viewer.virtual_image_tab_widget

    # the shortcuts are bound to Ctrl+Tab / Ctrl+Shift+Tab
    from PyQt5 import QtGui

    assert viewer.virtual_image_tab_next_shortcut.key() == QtGui.QKeySequence(
        "Ctrl+Tab"
    )
    assert viewer.virtual_image_tab_prev_shortcut.key() == QtGui.QKeySequence(
        "Ctrl+Shift+Tab"
    )

    # with only the default tab, the shortcuts are a no-op
    assert tab_widget.currentIndex() == 0
    viewer.virtual_image_tab_next_shortcut.activated.emit()
    viewer.virtual_image_tab_prev_shortcut.activated.emit()
    assert tab_widget.currentIndex() == 0

    tabs = [api.create_virtual_image_tab(f"Tab {i}") for i in range(2)]
    try:
        # Ctrl+Tab: forward, wrapping from the last tab back to the first
        tab_widget.setCurrentIndex(0)
        viewer.virtual_image_tab_next_shortcut.activated.emit()
        assert tab_widget.currentIndex() == 1
        viewer.virtual_image_tab_next_shortcut.activated.emit()
        assert tab_widget.currentIndex() == 2
        viewer.virtual_image_tab_next_shortcut.activated.emit()
        assert tab_widget.currentIndex() == 0

        # Ctrl+Shift+Tab: backward, wrapping from the first tab to the last
        viewer.virtual_image_tab_prev_shortcut.activated.emit()
        assert tab_widget.currentIndex() == 2
        viewer.virtual_image_tab_prev_shortcut.activated.emit()
        assert tab_widget.currentIndex() == 1
        viewer.virtual_image_tab_prev_shortcut.activated.emit()
        assert tab_widget.currentIndex() == 0
    finally:
        for t in tabs:
            api.close_virtual_image_tab(t)


def test_cmd_w_closes_window_and_bare_w_nudges(viewer, monkeypatch):
    from PyQt5 import QtCore, QtGui

    # Ctrl+W (Command+W on a Mac) is bound to close the window.
    assert viewer.close_shortcut.key() == QtGui.QKeySequence("Ctrl+W")

    # The W/A/S/D nudge bindings must only fire on the bare key: with a
    # modifier held (e.g. the Command in Cmd+W) the press is left alone
    # instead of nudging the diffraction selector.
    nudged = []
    monkeypatch.setattr(
        viewer, "nudge_diffraction_selector", lambda dx, dy: nudged.append((dx, dy))
    )

    def press(key, modifiers):
        viewer.keyPressEvent(QtGui.QKeyEvent(QtGui.QKeyEvent.KeyPress, key, modifiers))

    press(QtCore.Qt.Key_W, QtCore.Qt.MetaModifier)  # Cmd+W: close, not nudge
    assert nudged == []

    press(QtCore.Qt.Key_W, QtCore.Qt.NoModifier)  # bare W: 1 px nudge
    assert nudged == [(-1, 0)]

    press(QtCore.Qt.Key_A, QtCore.Qt.ShiftModifier)  # Shift+WASD: 5 px nudge
    assert nudged == [(-1, 0), (0, -5)]


def test_virtual_image_tab_image_and_scalebar(viewer):
    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        im = np.ones((4, 4), dtype=np.float32)
        tab.set_image(im, pixel_size=0.5, pixel_units="nm")

        assert tab.image is im
        assert tab.scale_bar.pixel_size == 0.5
        assert tab.scale_bar.units == "nm"

        # set_scalebar can update either field independently
        tab.set_scalebar(pixel_size=2.0)
        assert tab.scale_bar.pixel_size == 2.0
        assert tab.scale_bar.units == "nm"
    finally:
        api.close_virtual_image_tab(tab)


def test_tab_autoscale_matches_built_in_percentile_range(viewer):
    # A new tab's display levels must be derived from the same autoscale
    # percentiles the built-in pane uses (the user's configured range), not
    # pyqtgraph's default min/max autorange.
    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        # data with outliers, so min/max and percentile ranges differ
        im = np.full((16, 16), 1.0, dtype=np.float32)
        im[0, 0] = 1e6  # outlier high
        im[-1, -1] = 0.0  # outlier low
        tab.set_image(im, reset=True)

        p0, p1 = viewer.real_space_autoscale_percentiles
        expected = (
            float(np.percentile(im, p0)),
            float(np.percentile(im, p1)),
        )
        levels = np.asarray(tab.image_view.getImageItem().levels)
        assert np.allclose(levels, expected)

        # the min/max of the data (what autoLevels=True would pick) must NOT
        # be the tab's levels
        assert not np.allclose(levels, (float(im.min()), float(im.max())))

        # With autorange off, reset=False leaves the existing levels untouched
        # (a fresh tab defaults to the built-in's latched autorange, which would
        # otherwise re-derive on every reset=False set_image).
        tab.autorange = False
        levels_before = levels.copy()
        tab.set_image(im * 1000.0, reset=False)
        assert np.allclose(tab.image_view.getImageItem().levels, levels_before)
    finally:
        api.close_virtual_image_tab(tab)


def _set_builtin_vimg_scaling(viewer, mode):
    """Trigger the built-in Virtual Image scaling action named ``mode``."""
    for action in viewer.vimg_scaling_group.actions():
        if action.text().replace("&", "") == mode:
            action.trigger()
            return
    raise AssertionError(f"no built-in scaling action for {mode!r}")


def _set_builtin_diff_scaling(viewer, mode):
    """Trigger the built-in Diffraction scaling action named ``mode``."""
    for action in viewer.diff_scaling_group.actions():
        if action.text().replace("&", "") == mode:
            action.trigger()
            return
    raise AssertionError(f"no built-in diffraction scaling action for {mode!r}")


def test_tab_scaling_inherits_builtin_and_is_settable(viewer):
    # A fresh tab starts with the built-in pane's scaling; the setter then
    # changes it independently of the built-in.
    api = PluginAPI(viewer)
    orig = viewer.vimg_scaling_group.checkedAction()
    try:
        _set_builtin_vimg_scaling(viewer, "Log")
        tab = api.create_virtual_image_tab("Test")
        try:
            assert tab.scaling == "Log"
            # the setter updates the tab without touching the built-in
            tab.scaling = "Square Root"
            assert tab.scaling == "Square Root"
            assert (
                viewer.vimg_scaling_group.checkedAction().text().replace("&", "")
                == "Log"
            )
        finally:
            api.close_virtual_image_tab(tab)
    finally:
        orig.trigger()


def test_tab_scaling_setter_rejects_unknown_mode(viewer):
    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        with pytest.raises(ValueError, match="Unknown scaling mode"):
            tab.scaling = "Bogus"
    finally:
        api.close_virtual_image_tab(tab)


def test_tab_autorange_defaults_to_builtin_and_toggles(viewer):
    # A fresh tab starts with the built-in's Autorange latch; the setter then
    # toggles the tab's latch independently.
    api = PluginAPI(viewer)
    orig_latch = viewer.realspace_rescale_button.latched
    try:
        viewer.realspace_rescale_button.latched = True
        tab_on = api.create_virtual_image_tab("On")
        try:
            assert tab_on.autorange is True
            tab_on.autorange = False
            assert tab_on.autorange is False
            tab_on.autorange = True
            assert tab_on.autorange is True
        finally:
            api.close_virtual_image_tab(tab_on)

        viewer.realspace_rescale_button.latched = False
        tab_off = api.create_virtual_image_tab("Off")
        try:
            assert tab_off.autorange is False
        finally:
            api.close_virtual_image_tab(tab_off)
    finally:
        viewer.realspace_rescale_button.latched = orig_latch


def test_tab_autorange_on_rederives_off_freezes(viewer):
    # Autorange on: a non-reset set_image re-derives the range. Off: the range
    # is frozen and a non-reset set_image leaves it untouched.
    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        im = np.linspace(1.0, 100.0, 64).reshape(8, 8).astype(np.float32)
        tab.autorange = True
        tab.set_image(im, reset=True)
        l1 = tab.levels
        tab.set_image(im * 10.0, reset=False)  # re-derives (autorange on)
        l2 = tab.levels
        assert not np.allclose(l1, l2)

        tab.autorange = False
        tab.set_image(im * 100.0, reset=False)  # frozen
        assert np.allclose(tab.levels, l2)
    finally:
        api.close_virtual_image_tab(tab)


def test_tab_levels_and_set_levels(viewer):
    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        # no image yet: levels is None and set_levels refuses
        assert tab.levels is None
        with pytest.raises(RuntimeError):
            tab.set_levels(0.1, 0.9)

        im = np.linspace(1.0, 100.0, 64).reshape(8, 8).astype(np.float32)
        tab.set_image(im, reset=True)
        tab.set_levels(0.25, 0.75)
        assert np.allclose(tab.levels, (0.25, 0.75))
    finally:
        api.close_virtual_image_tab(tab)


def test_tab_autorange_percentiles_rejects_bad_pair(viewer):
    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        with pytest.raises(ValueError, match="autorange_percentiles"):
            tab.autorange_percentiles = 5.0
    finally:
        api.close_virtual_image_tab(tab)


def test_tab_percentile_change_follows_autorange(viewer):
    # While autorange is on, changing the percentiles re-derives the range;
    # while off, the range is frozen regardless of the percentiles.
    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        im = np.linspace(1.0, 100.0, 64).reshape(8, 8).astype(np.float32)
        tab.set_image(im, reset=True)

        tab.autorange = True
        tab.autorange_percentiles = (10.0, 90.0)
        l_wide = tab.levels
        tab.autorange_percentiles = (50.0, 60.0)
        l_narrow = tab.levels
        assert not np.allclose(l_wide, l_narrow)

        tab.autorange = False
        frozen = tab.levels
        tab.autorange_percentiles = (10.0, 90.0)
        assert np.allclose(tab.levels, frozen)
    finally:
        api.close_virtual_image_tab(tab)


def test_tab_independent_of_builtin_and_other_tabs(viewer):
    # Mutating one tab's display state leaves the built-in pane and every
    # other tab untouched.
    api = PluginAPI(viewer)
    orig = viewer.vimg_scaling_group.checkedAction()
    orig_latch = viewer.realspace_rescale_button.latched
    orig_pcts = tuple(viewer.real_space_autoscale_percentiles)
    tab_a = api.create_virtual_image_tab("A")
    tab_b = api.create_virtual_image_tab("B")
    try:
        im = np.linspace(1.0, 100.0, 64).reshape(8, 8).astype(np.float32)
        tab_a.set_image(im, reset=True)
        tab_b.set_image(im, reset=True)

        before_b = (
            tab_b.scaling,
            tab_b.autorange,
            tab_b.autorange_percentiles,
            tab_b.levels,
        )

        tab_a.scaling = "Log"
        tab_a.autorange = not tab_a.autorange
        tab_a.autorange_percentiles = (3.0, 97.0)
        tab_a.set_levels(0.3, 0.7)

        # tab B is untouched
        assert (
            tab_b.scaling,
            tab_b.autorange,
            tab_b.autorange_percentiles,
            tab_b.levels,
        ) == before_b
        # the built-in pane is untouched
        assert (
            viewer.vimg_scaling_group.checkedAction().text().replace("&", ""),
            viewer.realspace_rescale_button.latched,
            tuple(viewer.real_space_autoscale_percentiles),
        ) == (orig.text().replace("&", ""), orig_latch, orig_pcts)
    finally:
        api.close_virtual_image_tab(tab_a)
        api.close_virtual_image_tab(tab_b)
        viewer.real_space_autoscale_percentiles = orig_pcts
        viewer.realspace_rescale_button.latched = orig_latch
        orig.trigger()


def test_tab_levels_parity_with_builtin_under_log(viewer):
    # For the same image + Log scaling + percentiles, the tab's re-derived
    # levels must match the built-in pane's (both route through scale_image).
    api = PluginAPI(viewer)
    orig = viewer.vimg_scaling_group.checkedAction()
    orig_pcts = tuple(viewer.real_space_autoscale_percentiles)
    im = np.linspace(1.0, 1000.0, 256).reshape(16, 16).astype(np.float32)
    try:
        # drive the built-in to Log + (2, 98), then render
        _set_builtin_vimg_scaling(viewer, "Log")
        viewer.set_real_space_autoscale_range((2.0, 98.0), redraw=False)
        viewer.set_virtual_image(im, reset=True)
        built_levels = viewer.real_space_widget.getHistogramWidget().getLevels()

        # a fresh tab created now inherits Log + (2, 98)
        tab = api.create_virtual_image_tab("Test")
        try:
            assert tab.scaling == "Log"
            assert tuple(tab.autorange_percentiles) == (2.0, 98.0)
            tab.set_image(im, reset=True)
            tab_levels = tab.image_view.getHistogramWidget().getLevels()
            assert np.allclose(built_levels, tab_levels)
        finally:
            api.close_virtual_image_tab(tab)
    finally:
        viewer.set_virtual_image(None, reset=False)
        viewer.set_real_space_autoscale_range(orig_pcts, redraw=False)
        orig.trigger()


def test_tab_complex_image_ignores_scaling(viewer):
    # Complex images render as a fixed (0,1) Lab/RGB image; scaling is ignored.
    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        im = np.zeros((8, 8), dtype=np.complex64)
        im.real = np.linspace(0.0, 1.0, 64).reshape(8, 8)
        im.imag = np.linspace(1.0, 0.0, 64).reshape(8, 8)
        tab.scaling = "Log"
        tab.set_image(im, reset=True)
        assert np.allclose(tab.levels, (0.0, 1.0))
    finally:
        api.close_virtual_image_tab(tab)


def test_tab_widget_is_container_with_bottom_bar(viewer):
    # widget is the QTabWidget page container; the render view is image_view
    # (child 0), with the display-control bar below it (child 1).
    import pyqtgraph as pg

    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        assert isinstance(tab.widget, QWidget)
        assert tab.widget is not tab.image_view
        assert isinstance(tab.image_view, pg.ImageView)

        layout = tab.widget.layout()
        assert layout.itemAt(0).widget() is tab.image_view
        bottom = layout.itemAt(1).widget()
        assert bottom is not None
        assert bottom is not tab.image_view

        # the page registered in the QTabWidget is the container
        assert viewer.virtual_image_tab_widget.widget(1) is tab.widget
    finally:
        api.close_virtual_image_tab(tab)


def test_controls_visible_toggles_bottom_bar(viewer):
    # controls_visible (default False) shows/hides the whole display-control bar;
    # the toggle button lives in the page (independent of the bar) and always
    # stays visible, so the bar can be brought back; a closed tab rejects the
    # setter.
    from PyQt5 import QtCore

    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        # The toggle button is a child of the page, not of the control bar, so
        # hiding the bar (above) does not hide the button.
        assert tab._controls_toggle.parentWidget() is tab.widget

        # Defaults: bar hidden (so the view owns the full height), with a
        # "down" (expand) arrow offering to bring the bar in. (isHidden()
        # reflects the explicit setVisible() state; isVisible() would be False
        # for any widget whose ancestors are never shown, as in the offscreen
        # fixture.)
        assert tab.controls_visible is False
        assert tab._controls.isHidden()
        assert not tab._controls_toggle.isHidden()
        assert tab._controls_toggle.arrowType() == QtCore.Qt.DownArrow
        assert tab._controls_toggle.toolTip() == "Show display controls"

        # Expanding shows the whole bar and flips the arrow to "up" (collapse).
        tab.controls_visible = True
        assert tab.controls_visible is True
        assert not tab._controls.isHidden()
        assert not tab._controls_toggle.isHidden()
        assert tab._controls_toggle.arrowType() == QtCore.Qt.UpArrow
        assert tab._controls_toggle.toolTip() == "Hide display controls"

        # Collapsing again (e.g. via the toggle button's click) hides the bar.
        tab._controls_toggle.click()
        assert tab.controls_visible is False
        assert tab._controls.isHidden()
        assert tab._controls_toggle.arrowType() == QtCore.Qt.DownArrow

        # Setting the same value again is a no-op, not an error.
        tab.controls_visible = False
        tab.controls_visible = False
        assert tab.controls_visible is False
    finally:
        api.close_virtual_image_tab(tab)

    # A closed tab rejects the setter.
    with pytest.raises(RuntimeError):
        tab.controls_visible = True


def test_toggle_button_follows_page_resize(viewer):
    # The toggle button must sit in the page's bottom-left corner (flush with
    # the frame) for the page's *current* size, tracking it as the page grows
    # — not stranded at the init-time corner until the bar is toggled. (This
    # regressed when the resize hook was misnamed ``_eventFilter`` instead of
    # the C++ virtual ``eventFilter`` Qt actually dispatches to.)
    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    tab.set_image(np.random.rand(64, 64), reset=True)
    try:
        page = tab.widget
        btn = tab._controls_toggle
        # Show the page the way the QTabWidget does in the real GUI (the
        # offscreen fixture never shows it, and a resize on a hidden widget
        # delivers no resize event), then grow it — the button must track the
        # new bottom-left corner, not the initial one.
        page.show()
        page.resize(800, 600)
        viewer.qtapp.processEvents()

        expected = (0, page.height() - btn.height())
        assert (btn.x(), btn.y()) == expected
        # Growing the page must not have un-hidden the collapsed bar.
        assert tab._controls.isHidden()
    finally:
        api.close_virtual_image_tab(tab)


def test_current_virtual_image_settings_tracks_visible_view(viewer):
    # current_virtual_image_settings reflects whichever view is on screen:
    # the built-in pane, an image tab, or None when there is no 2D data.
    from py4D_browser.utils import VirtualImageSettings

    api = PluginAPI(viewer)
    im = np.linspace(1.0, 100.0, 64).reshape(8, 8).astype(np.float32)
    tab = api.create_virtual_image_tab("Test")

    def show_builtin():
        viewer.virtual_image_tab_widget.setCurrentIndex(0)

    def show(t):
        viewer.virtual_image_tab_widget.setCurrentIndex(
            viewer.virtual_image_tabs.index(t) + 1
        )

    try:
        show_builtin()
        viewer.set_virtual_image(None, reset=False)
        assert api.current_virtual_image_settings is None

        # built-in, with data -> the built-in pane's settings
        viewer.set_virtual_image(im, reset=True)
        s = api.current_virtual_image_settings
        assert isinstance(s, VirtualImageSettings)
        assert s.scaling == viewer.vimg_scaling_group.checkedAction().text().replace(
            "&", ""
        )
        assert s.autorange is viewer.realspace_rescale_button.latched
        assert s.percentiles == tuple(viewer.real_space_autoscale_percentiles)

        # image tab, with its own data -> the tab's independent settings
        tab.set_image(im, reset=True)
        tab.scaling = "Square Root"
        show(tab)
        s = api.current_virtual_image_settings
        assert isinstance(s, VirtualImageSettings)
        assert s.scaling == "Square Root"
        assert s.autorange is tab.autorange
        assert s.percentiles == tuple(tab.autorange_percentiles)

        # image tab with no data -> None
        empty = api.create_virtual_image_tab("Empty")
        try:
            show(empty)
            assert api.current_virtual_image_settings is None
        finally:
            api.close_virtual_image_tab(empty)
    finally:
        show_builtin()
        api.close_virtual_image_tab(tab)
        viewer.set_virtual_image(None, reset=False)


def test_current_virtual_image_settings_none_for_volume_tab(viewer):
    # A volume tab on display has no 2D display settings.
    pytest.importorskip("OpenGL")
    api = PluginAPI(viewer)
    vol = api.create_volume_tab("Vol")
    try:
        viewer.virtual_image_tab_widget.setCurrentIndex(
            viewer.virtual_image_tabs.index(vol) + 1
        )
        assert api.current_virtual_image_settings is None
    finally:
        viewer.virtual_image_tab_widget.setCurrentIndex(0)
        api.close_virtual_image_tab(vol)


def test_current_diffraction_image_settings_tracks_dp_pane(viewer):
    # current_diffraction_image_settings reflects the DP pane's scaling /
    # Autorange / autoscale percentiles / levels, and is None when the pane is
    # empty.
    from py4D_browser.utils import VirtualImageSettings

    api = PluginAPI(viewer)
    dp = np.linspace(1.0, 1000.0, 64).reshape(8, 8).astype(np.float32)
    orig = viewer.diff_scaling_group.checkedAction()
    try:
        # empty DP pane -> None (clear first: a prior test may have left a DP on display)
        viewer.set_diffraction_image(None, reset=False)
        assert api.current_diffraction_image_settings is None

        # with a DP on display -> the DP pane's settings
        _set_builtin_diff_scaling(viewer, "Square Root")
        viewer.set_diffraction_autoscale_range((3.0, 97.0), redraw=False)
        viewer.diffraction_rescale_button.latched = True
        viewer.set_diffraction_image(dp, reset=True)

        s = api.current_diffraction_image_settings
        assert isinstance(s, VirtualImageSettings)
        assert s.scaling == "Square Root"
        assert s.autorange is viewer.diffraction_rescale_button.latched
        assert s.percentiles == tuple(viewer.diffraction_autoscale_percentiles)
        assert np.allclose(
            s.levels,
            viewer.diffraction_space_widget.getHistogramWidget().getLevels(),
        )
    finally:
        viewer.set_diffraction_image(None, reset=False)
        orig.trigger()


def test_copy_to_tab_mirrors_builtin_settings(viewer):
    # The copy action mirrors the built-in pane's scaling + range (not its
    # colormap), whether autorange is on (re-derived) or off (manually set).
    viewer.load_plugins()
    entry = next(
        e for e in viewer.loaded_plugins if e["id"] == "py4DGUI.internal.copy_to_tab"
    )
    assert type(entry["api"]) is PluginAPI1
    copy_vi = entry["menu"].actions()[0]

    orig = viewer.vimg_scaling_group.checkedAction()
    orig_latch = viewer.realspace_rescale_button.latched
    orig_pcts = tuple(viewer.real_space_autoscale_percentiles)
    im = np.linspace(1.0, 1000.0, 256).reshape(16, 16).astype(np.float32)
    try:
        _set_builtin_vimg_scaling(viewer, "Log")

        # --- autorange on: range re-derived at the built-in percentiles ---
        viewer.set_real_space_autoscale_range((5.0, 95.0), redraw=False)
        viewer.realspace_rescale_button.latched = True
        viewer.set_virtual_image(im, reset=True)
        built_levels = viewer.real_space_widget.getHistogramWidget().getLevels()

        copy_vi.trigger()
        tab = viewer.virtual_image_tabs[-1]
        assert tab.scaling == "Log"
        assert tab.autorange is True
        assert np.allclose(tab.autorange_percentiles, (5.0, 95.0))
        assert np.allclose(tab.levels, built_levels)
        viewer.close_virtual_image_tab(tab)

        # --- autorange off: range is the built-in's manually-set range ---
        viewer.realspace_rescale_button.latched = False
        viewer.set_virtual_image(im, reset=False)
        viewer.real_space_widget.setLevels(10.0, 500.0)

        copy_vi.trigger()
        tab = viewer.virtual_image_tabs[-1]
        assert tab.scaling == "Log"
        assert tab.autorange is False
        assert np.allclose(tab.levels, (10.0, 500.0))
        viewer.close_virtual_image_tab(tab)
    finally:
        viewer.set_virtual_image(None, reset=False)
        viewer.realspace_rescale_button.latched = orig_latch
        viewer.set_real_space_autoscale_range(orig_pcts, redraw=False)
        orig.trigger()


def test_copy_diffraction_to_tab_mirrors_builtin_settings(viewer):
    # "Copy Diffraction Pattern to Tab" mirrors the DP pane's scaling + range
    # (not its colormap), whether autorange is on (re-derived) or off
    # (manually set) — so a copy of a sqrt-scaled DP stays sqrt-scaled instead
    # of falling back to the real-space default.
    viewer.load_plugins()
    entry = next(
        e for e in viewer.loaded_plugins if e["id"] == "py4DGUI.internal.copy_to_tab"
    )
    assert type(entry["api"]) is PluginAPI1
    copy_dp = entry["menu"].actions()[1]

    orig = viewer.diff_scaling_group.checkedAction()
    orig_latch = viewer.diffraction_rescale_button.latched
    orig_pcts = tuple(viewer.diffraction_autoscale_percentiles)
    dp = np.linspace(1.0, 1000.0, 256).reshape(16, 16).astype(np.float32)
    try:
        _set_builtin_diff_scaling(viewer, "Square Root")

        # --- autorange on: range re-derived at the DP pane's percentiles ---
        viewer.set_diffraction_autoscale_range((5.0, 95.0), redraw=False)
        viewer.diffraction_rescale_button.latched = True
        viewer.set_diffraction_image(dp, reset=True)
        built_levels = viewer.diffraction_space_widget.getHistogramWidget().getLevels()

        copy_dp.trigger()
        tab = viewer.virtual_image_tabs[-1]
        assert tab.scaling == "Square Root"
        assert tab.autorange is True
        assert np.allclose(tab.autorange_percentiles, (5.0, 95.0))
        assert np.allclose(tab.levels, built_levels)
        viewer.close_virtual_image_tab(tab)

        # --- autorange off: range is the DP pane's manually-set range ---
        viewer.diffraction_rescale_button.latched = False
        viewer.set_diffraction_image(dp, reset=False)
        viewer.diffraction_space_widget.setLevels(10.0, 500.0)

        copy_dp.trigger()
        tab = viewer.virtual_image_tabs[-1]
        assert tab.scaling == "Square Root"
        assert tab.autorange is False
        assert np.allclose(tab.levels, (10.0, 500.0))
        viewer.close_virtual_image_tab(tab)
    finally:
        viewer.set_diffraction_image(None, reset=False)
        viewer.diffraction_rescale_button.latched = orig_latch
        viewer.set_diffraction_autoscale_range(orig_pcts, redraw=False)
        orig.trigger()


def test_tab_scalebar_uses_unicode_angstrom(viewer):
    # Units like "A" must be translated to "Å" for display, matching the
    # built-in scale bars.
    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        tab.set_scalebar(pixel_size=0.1, units="A")
        assert tab.scale_bar.units == "Å"

        # and through set_image's pixel_units
        tab2 = api.create_virtual_image_tab("Test 2")
        tab2.set_image(np.ones((4, 4), dtype=np.float32), pixel_units="A")
        assert tab2.scale_bar.units == "Å"
        api.close_virtual_image_tab(tab2)
    finally:
        api.close_virtual_image_tab(tab)


def test_tab_colormap_defaults_to_settings_and_is_independent(viewer):
    from py4D_browser.utils import try_get_cmap

    api = PluginAPI(viewer)
    source_item = viewer.real_space_widget.getImageItem()

    def lut_of(cmap_or_lut):
        # Resolve a pg.ColorMap, or a pane's (possibly live-callable) LUT, to
        # a concrete 256-entry RGB array so the two can be compared directly.
        if hasattr(cmap_or_lut, "getLookupTable"):
            table = cmap_or_lut.getLookupTable(0, 1, 256)
        else:
            table = cmap_or_lut
            if callable(table):
                table = table(None, 256)
        return np.asarray(table)[:, :3]

    # Pin the default virtual-image colormap to a known value (distinct from
    # the "magma" the source pane is pointed at below) so the assertion is
    # deterministic. Remember the original setting so we can restore it.
    orig_name = viewer.settings.value("gui/realspace_colormap", "thermal")
    viewer.settings.setValue("gui/realspace_colormap", "thermal")
    default_cmap = try_get_cmap("thermal")
    assert default_cmap is not None  # "thermal" is a built-in gradient

    # Point the source pane at a non-default colormap. The new tab's default
    # must come from the settings object, not from whatever the source pane
    # is currently showing.
    viewer.real_space_widget.setColorMap(try_get_cmap("magma"))

    tab = api.create_virtual_image_tab("Test")
    try:
        # (1) the new tab is seeded with the browser's default virtual-image
        #     colormap (from settings: "thermal" here) — not pyqtgraph's
        #     default, and not the source pane's magma.
        assert tab.colormap is not None
        assert np.allclose(lut_of(tab.colormap), lut_of(default_cmap), atol=1)
        assert not np.allclose(
            lut_of(tab.colormap), lut_of(try_get_cmap("magma")), atol=1
        )

        # (2) changing the tab's colormap leaves the source pane untouched,
        #     i.e. each tab keeps its own colormap from then on.
        tab.colormap = "viridis"
        assert np.allclose(
            lut_of(tab.colormap), lut_of(try_get_cmap("viridis")), atol=1
        )
        # the source pane is still showing magma
        assert np.allclose(
            lut_of(source_item.lut), lut_of(try_get_cmap("magma")), atol=1
        )
    finally:
        api.close_virtual_image_tab(tab)
        # restore the setting and the pane so the shared (session-scoped)
        # viewer is left in a clean state for other tests.
        viewer.settings.setValue("gui/realspace_colormap", orig_name)
        orig_cmap = try_get_cmap(orig_name) or try_get_cmap("thermal")
        if orig_cmap is not None:
            viewer.real_space_widget.setColorMap(orig_cmap)


def test_tab_colormap_settable_by_name(viewer):
    from py4D_browser.utils import try_get_cmap

    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        # a name resolves through the browser's colormap lookup. (Compare
        # lookup tables, not identity: try_get_cmap returns a fresh ColorMap
        # object each call.)
        tab.colormap = "magma"
        expected = try_get_cmap("magma")
        assert np.array_equal(tab.colormap.getLookupTable(), expected.getLookupTable())

        # ... and an unknown name raises
        with pytest.raises(ValueError, match="Unknown colormap"):
            tab.colormap = "not_a_colormap"
    finally:
        api.close_virtual_image_tab(tab)


def test_default_tab_has_no_close_button(viewer):
    # The tab bar is in closable mode (so plugin tabs get native close
    # buttons), but the unclosable default tab's close button is suppressed
    # on whichever side the style draws it (RightSide on most platforms,
    # LeftSide on macOS). We resolve that side the same way the viewer does.
    side = viewer._tab_close_button_side()
    tab_widget = viewer.virtual_image_tab_widget
    assert tab_widget.tabsClosable()
    assert tab_widget.tabBar().tabButton(0, side) is None

    # with a plugin tab present, the default tab still has no button while
    # the plugin tab gets a (native) close button on the same side
    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        bar = tab_widget.tabBar()
        assert bar.tabButton(0, side) is None
        assert bar.tabButton(1, side) is not None
        # clicking the plugin tab's close button closes it: the native button
        # emits tabCloseRequested with the tab's index, which the viewer routes
        # to close_virtual_image_tab
        bar.tabCloseRequested.emit(1)
        assert api.virtual_image_tabs == []
        assert tab_widget.count() == 1
    finally:
        if not tab.closed:
            api.close_virtual_image_tab(tab)


def test_virtual_image_tab_rois_and_annotations(viewer):
    import pyqtgraph as pg

    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    try:
        roi = pg.ROI([1, 1], [5, 5])
        line = pg.InfiniteLine(pos=2.0, pen=pg.mkPen("r"))

        # the accessors return the items for chaining, and track them
        assert tab.add_roi(roi) is roi
        assert tab.add_annotation(line) is line
        assert tab.rois == [roi]
        assert tab.annotations == [line]

        # both were attached to the tab's view scene
        scene = tab.image_view.getView().scene()
        assert roi.scene() is scene
        assert line.scene() is scene
    finally:
        api.close_virtual_image_tab(tab)


def test_close_virtual_image_tab_cleans_up(viewer):
    import pyqtgraph as pg

    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    tab.set_image(np.ones((4, 4), dtype=np.float32))
    roi = pg.ROI([1, 1], [5, 5])
    line = pg.InfiniteLine(pos=2.0, pen=pg.mkPen("r"))
    tab.add_roi(roi)
    tab.add_annotation(line)

    try:
        # sanity: the tab is in the pane and tracked
        assert viewer.virtual_image_tab_widget.count() == 2
        assert api.virtual_image_tabs == [tab]

        api.close_virtual_image_tab(tab)

        # the tab is removed from the pane and from the tracked list
        assert viewer.virtual_image_tab_widget.count() == 1
        assert api.virtual_image_tabs == []

        # the ROIs/annotations were detached from the scene (but not deleted,
        # so a plugin could still reuse them)
        assert roi.scene() is None
        assert line.scene() is None

        # the visible image falls back to the built-in one (default tab shown)
        assert viewer.current_virtual_image is viewer.unscaled_realspace_image

        # the tab is marked closed, and mutating it now raises
        assert tab.closed
        with pytest.raises(RuntimeError):
            tab.set_image(np.zeros((2, 2), dtype=np.float32))
    finally:
        # idempotent: a second close is a harmless no-op
        api.close_virtual_image_tab(tab)


def test_current_virtual_image_tracks_visible_tab(viewer):
    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")
    im = np.ones((4, 4), dtype=np.float32)
    tab.set_image(im)

    fired = []
    handler = lambda: fired.append(True)
    try:
        viewer.signal_current_virtual_image_changed.connect(handler)

        # default tab visible: current image is the built-in one
        assert viewer.virtual_image_tab_widget.currentIndex() == 0
        assert viewer.current_virtual_image is viewer.unscaled_realspace_image
        assert api.current_virtual_image is viewer.current_virtual_image

        # switching to the plugin tab exposes its raw array and fires the signal
        viewer.virtual_image_tab_widget.setCurrentIndex(1)
        assert viewer.current_virtual_image is im
        assert api.current_virtual_image is im
        assert fired == [True]

        # set_image on the visible tab also fires the signal
        fired.clear()
        im2 = np.full((4, 4), 7.0, dtype=np.float32)
        tab.set_image(im2)
        assert viewer.current_virtual_image is im2
        assert fired == [True]
    finally:
        viewer.signal_current_virtual_image_changed.disconnect(handler)
        api.close_virtual_image_tab(tab)


########## Virtual-image export (menu items follow the visible tab) ##########


def test_export_virtual_image_uses_visible_tab(viewer, monkeypatch, tmp_path):
    # The "Export" menu items for the virtual-image pane must export whatever
    # is currently visible in the pane (menu_actions.export_virtual_image):
    # the built-in image when the default tab is visible, or a plugin tab's
    # image when that tab is visible.
    import tifffile

    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")

    # different shapes, so exporting the wrong source is detectable
    default_im = np.arange(16, dtype=np.float32).reshape(4, 4)
    tab_im = np.arange(48, dtype=np.float32).reshape(8, 6)

    viewer.set_virtual_image(default_im, reset=True)
    tab.set_image(tab_im)

    # route the save dialog to a fixed path
    out = tmp_path / "vimg_export.tiff"
    monkeypatch.setattr(viewer, "get_savefile_name", lambda fmt: str(out))

    try:
        # (1) default tab visible -> the built-in image is exported
        viewer.virtual_image_tab_widget.setCurrentIndex(0)
        viewer.export_virtual_image("TIFF (raw)", "image")
        exported = tifffile.imread(str(out))
        assert exported.shape == default_im.shape
        assert np.allclose(exported, default_im)

        # (2) plugin tab visible -> the tab's image is exported, not the
        # built-in one
        viewer.virtual_image_tab_widget.setCurrentIndex(1)
        viewer.export_virtual_image("TIFF (raw)", "image")
        exported = tifffile.imread(str(out))
        assert exported.shape == tab_im.shape
        assert np.allclose(exported, tab_im)

        # (3) the "display" format exports the visible pane's rendered view
        # (matplotlib may write grayscale as multi-channel, so compare the
        # spatial dims — 8x6 proves it's the tab's image, not the 4x4
        # built-in one)
        viewer.export_virtual_image("TIFF (display)", "image")
        displayed = tifffile.imread(str(out))
        assert displayed.shape[:2] == (tab_im.shape[0], tab_im.shape[1])
    finally:
        viewer.virtual_image_tab_widget.setCurrentIndex(0)
        api.close_virtual_image_tab(tab)


def test_copy_vimg_to_clipboard_uses_visible_tab(viewer):
    # "Copy virtual image to clipboard" (menu_actions.copy_vimg_to_clipboard)
    # must copy the currently-visible tab's image, not always the built-in
    # one.
    from PyQt5.QtWidgets import QApplication

    api = PluginAPI(viewer)
    tab = api.create_virtual_image_tab("Test")

    # different shapes, so copying the wrong source is detectable
    default_im = np.full((4, 4), 1.0, dtype=np.float32)
    tab_im = np.full((8, 6), 2.0, dtype=np.float32)

    viewer.set_virtual_image(default_im, reset=True)
    tab.set_image(tab_im)

    clipboard = QApplication.clipboard()
    try:
        # default tab visible -> copies the built-in image's dimensions
        viewer.virtual_image_tab_widget.setCurrentIndex(0)
        viewer.copy_vimg_to_clipboard()
        qim = clipboard.image()
        assert (qim.width(), qim.height()) == (default_im.shape[1], default_im.shape[0])

        # plugin tab visible -> copies the tab's image
        viewer.virtual_image_tab_widget.setCurrentIndex(1)
        viewer.copy_vimg_to_clipboard()
        qim = clipboard.image()
        assert (qim.width(), qim.height()) == (tab_im.shape[1], tab_im.shape[0])
    finally:
        viewer.virtual_image_tab_widget.setCurrentIndex(0)
        api.close_virtual_image_tab(tab)


########## Tab-demo plugin (end-to-end through the loader) ##########


def test_tab_demo_plugin_copies_visible_image(viewer):
    # the loader is idempotent; reload with real discovery in case the
    # fake-plugin tests in test_loader.py replaced the loaded set
    viewer.load_plugins()

    entry = next(
        e for e in viewer.loaded_plugins if e["id"] == "py4DGUI.internal.copy_to_tab"
    )
    assert type(entry["api"]) is PluginAPI1
    plugin = entry["plugin"]

    im = np.arange(64, dtype=np.float32).reshape(8, 8)
    viewer.set_virtual_image(im, reset=True)

    # simulate a click on the "Copy Virtual Image to Tab" menu item
    # (first action of the plugin's menu)
    entry["menu"].actions()[0].trigger()
    try:
        assert len(viewer.virtual_image_tabs) == 1
        tab = viewer.virtual_image_tabs[0]
        assert tab.title == "VI Copy 1"
        # the copy is the same raw array the pane was showing
        assert tab.image is im

        # a second click makes a second tab with the next free title
        entry["menu"].actions()[0].trigger()
        assert len(viewer.virtual_image_tabs) == 2
        assert viewer.virtual_image_tabs[1].title == "VI Copy 2"
    finally:
        # iterate a copy: closing a tab mutates the list we're iterating
        for tab in list(viewer.virtual_image_tabs):
            viewer.close_virtual_image_tab(tab)

    assert viewer.virtual_image_tabs == []


########## Shape-aware Virtual Image FFT (3D result pane) ##########


def _setup_dataset(viewer):
    """
    Load a small datacube and known 2D source images. Return the prior state
    for restoration by _teardown_dataset.
    """
    api = PluginAPI(viewer)
    state = (
        viewer.datacube,
        viewer.unscaled_diffraction_image,
        viewer.unscaled_realspace_image,
        viewer.result_source_action_group.checkedAction(),
        viewer.virtual_image_tab_widget.currentIndex(),
    )
    api.set_datacube(_make_datacube())
    viewer.set_diffraction_image(np.random.rand(4, 4).astype(np.float32), reset=False)
    viewer.set_virtual_image(np.random.rand(4, 4).astype(np.float32), reset=False)
    return api, state


def _teardown_dataset(viewer, state):
    original_dc, original_dp, original_vimg, prev_mode, prev_tab = state
    viewer.datacube = original_dc
    viewer.set_diffraction_image(original_dp, reset=False)
    viewer.set_virtual_image(original_vimg, reset=False)
    prev_mode.trigger()
    viewer.virtual_image_tab_widget.setCurrentIndex(prev_tab)


def _select_result_mode(viewer, name):
    for act in viewer.result_source_action_group.actions():
        if act.text().replace("&", "") == name:
            act.trigger()
            return
    raise AssertionError(f"no Result View action named {name!r}")


def test_virtual_image_fft_is_default_mode(viewer):
    # "Virtual Image FFT" is the shape-aware default Result View mode — 2D
    # image -> 2D FFT, 3D volume -> 3D FFT. It is the checked action in the
    # exclusive result-source group and part of the Result View menu.
    act = next(
        a
        for a in viewer.result_source_action_group.actions()
        if a.text().replace("&", "") == "Virtual Image FFT"
    )
    assert act.isCheckable()
    assert act in viewer.result_menu.actions()
    assert act in viewer.result_source_action_group.actions()
    # it is checked by default
    assert viewer.result_source_action_group.checkedAction() is act


def test_virtual_image_fft_with_2d_visible_renders_2d(viewer):
    # "Virtual Image FFT" is shape-aware: with a 2D image visible it renders
    # the 2D FFT into the 2D page (the existing 2D->RGB path) and does not
    # create a 3D volume item.
    pytest.importorskip("OpenGL")
    api, state = _setup_dataset(viewer)
    try:
        _select_result_mode(viewer, "Virtual Image FFT")
        assert viewer._result_stack.currentIndex() == 0
        assert viewer._result_volume_item is None
        # a 2D FFT was rendered into the 2D result path
        assert viewer.unscaled_fft_image is not None
        assert viewer.unscaled_fft_image.ndim == 2
    finally:
        _teardown_dataset(viewer, state)


def test_volume_fft_renders_3d_volume_preserves_2d_path(viewer):
    # "Virtual Image FFT" with a 3D volume tab visible renders the FFT as a
    # genuine 3D volume on the pane's 3D page, without touching the 2D result
    # path (which would have shown a 3D array as an RGB image).
    pytest.importorskip("OpenGL")
    from pyqtgraph import opengl as gl

    api, state = _setup_dataset(viewer)
    vol = api.create_volume_tab("Vol")
    try:
        # start clean so the assertions below exercise the creation path
        viewer._result_volume_item = None
        viewer._result_volume_fft = None

        vol.set_volume(np.random.rand(4, 5, 6).astype(np.float32), reset=True)
        viewer.virtual_image_tab_widget.setCurrentIndex(
            viewer.virtual_image_tabs.index(vol) + 1
        )

        fft_before = viewer.unscaled_fft_image
        _select_result_mode(viewer, "Virtual Image FFT")

        # the pane is on the 3D page, showing a 3D volume of the FFT
        assert viewer._result_stack.currentIndex() == 1
        item = viewer._result_volume_item
        assert isinstance(item, gl.GLVolumeItem)
        assert item.data.shape == (4, 5, 6, 4)
        assert item.data.dtype == np.ubyte
        # The RGBA must not be a solid-white block: the colormap is actually
        # applied, so the RGB channels span more than one color. (Regression
        # guard against a degenerate all-white lookup table, e.g. a
        # ColorMap.getLookupTable called with the point count passed
        # positionally as `start`.)
        rgb = item.data[..., :3]
        assert not np.all(rgb == 255)
        assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 1
        # The opacity transfer function is a power law (gamma=2), so the dim
        # low end is fully transparent (min alpha 0) — a linear ramp with an
        # opaque base (e.g. np.linspace(48, 255, 256)) would instead render
        # the dim bulk at a constant non-zero opacity, hiding the structure.
        assert item.data[..., 3].min() == 0
        assert viewer._result_volume_fft is not None
        # the 2D result path was left untouched
        assert viewer.unscaled_fft_image is fft_before
        # the view-direction lock is active and points at the volume tab
        assert viewer._view_sync._active
        assert viewer._view_sync._volume_view is vol.view

        # re-rendering reuses the same volume item (setData, not recreate)
        viewer.update_fft_view()
        assert viewer._result_volume_item is item
    finally:
        viewer.virtual_image_tab_widget.setCurrentIndex(0)
        api.close_virtual_image_tab(vol)
        _teardown_dataset(viewer, state)


def test_visible_volume_result_exports_only_the_current_fft(
    viewer, tmp_path, monkeypatch
):
    import tifffile
    from PyQt5.QtWidgets import QApplication

    api, state = _setup_dataset(viewer)
    vol = api.create_volume_tab("Volume", select=True)
    path = tmp_path / "fft.tif"
    save_formats = []
    clipboard = QApplication.clipboard()
    previous_clipboard = clipboard.text()

    def save_path(fmt):
        save_formats.append(fmt)
        return str(path)

    monkeypatch.setattr(viewer, "get_savefile_name", save_path)
    try:
        vol.set_volume(np.arange(60, dtype=np.float32).reshape(5, 3, 4))
        _select_result_mode(viewer, "Virtual Image FFT")

        clipboard.setText("unchanged")
        viewer.copy_result_to_clipboard()
        assert clipboard.text() == "unchanged"
        for fmt in ("PNG (display)", "TIFF (display)"):
            viewer.export_virtual_image(fmt, "result")
        assert save_formats == []

        viewer.export_virtual_image("TIFF (raw)", "result")
        assert save_formats == ["TIFF (raw)"]
        with tifffile.TiffFile(path) as tif:
            assert len(tif.pages) == 5
            np.testing.assert_array_equal(tif.asarray(), viewer._result_volume_fft)
    finally:
        clipboard.setText(previous_clipboard)
        vol.close()
        _teardown_dataset(viewer, state)


def test_plugin_result_replaces_visible_volume_fft(viewer):
    api, state = _setup_dataset(viewer)
    vol = api.create_volume_tab("Volume", select=True)
    try:
        vol.set_volume(np.arange(60, dtype=np.float32).reshape(5, 3, 4))
        _select_result_mode(viewer, "Virtual Image FFT")
        assert viewer._result_stack.currentIndex() == 1

        api.register_result_callback("Plugin result", None)
        result = np.arange(35, dtype=np.float32).reshape(5, 7)
        api.set_result_image(result, title="Plugin result", reset=True)
        assert viewer._result_stack.currentIndex() == 0
        assert not viewer._view_sync._active
        np.testing.assert_array_equal(viewer.fft_widget.image, result.T)
    finally:
        vol.close()
        _teardown_dataset(viewer, state)


@pytest.mark.parametrize("mode", ["Virtual Image FFT", "Virtual Image FFT (complex)"])
def test_image_tab_fft_uses_its_own_sampling(viewer, mode):
    api, state = _setup_dataset(viewer)
    tab = api.create_virtual_image_tab("Calibrated image", select=True)
    try:
        tab.set_image(np.arange(35).reshape(5, 7), pixel_size=2.0, pixel_units="nm")
        _select_result_mode(viewer, mode)
        assert viewer.fft_scale_bar.pixel_size == pytest.approx(1 / 14)
        assert viewer.fft_scale_bar.units == "nm⁻¹"

        tab.set_scalebar(pixel_size=3.0, units="A")
        assert viewer.fft_scale_bar.pixel_size == pytest.approx(1 / 21)
        assert viewer.fft_scale_bar.units == "Å⁻¹"

        viewer.datacube = None
        viewer.set_diffraction_image(None)
        tab.set_image(np.arange(45).reshape(5, 9))
        assert viewer.unscaled_fft_image.shape == (5, 9)
        assert viewer.fft_scale_bar.pixel_size == pytest.approx(1 / 27)

        viewer._render_result_image()
        transform = viewer.fft_widget.getImageItem().transform()
        assert transform.m22() == pytest.approx(9 / 5)
    finally:
        tab.close()
        _teardown_dataset(viewer, state)


def test_fft_cursor_reports_dc_and_vertical_frequency(viewer, qapp, monkeypatch):
    from PyQt5.QtCore import QPointF
    import py4D_browser.update_views as update_views

    api, state = _setup_dataset(viewer)
    tab = api.create_virtual_image_tab("Calibrated image", select=True)
    try:
        tab.set_image(np.arange(35).reshape(5, 7), pixel_size=2.0, pixel_units="nm")
        _select_result_mode(viewer, "Virtual Image FFT")
        qapp.processEvents()
        monkeypatch.setattr(viewer, "isActiveWindow", lambda: True)
        graphics_view = viewer.fft_widget.ui.graphicsView
        image_item = viewer.fft_widget.getImageItem()
        for row, expected in [(2, "k=0 nm⁻¹"), (1, "k=0.1 nm⁻¹, d=10 nm")]:
            scene_point = image_item.mapToScene(QPointF(3.5, row + 0.5))
            pos = graphics_view.viewport().mapToGlobal(
                graphics_view.mapFromScene(scene_point)
            )
            monkeypatch.setattr(
                update_views, "QCursor", type("Cursor", (), {"pos": lambda: pos})
            )
            viewer.update_tooltip()
            assert viewer.cursor_value_text.text().startswith(f"[{row},3]")
            assert expected in viewer.cursor_value_text.text()
    finally:
        tab.close()
        _teardown_dataset(viewer, state)


def test_closing_active_tab_notifies_listeners_with_new_image(viewer):
    api = PluginAPI(viewer)
    left = api.create_virtual_image_tab("Left", select=True)
    right = api.create_virtual_image_tab("Right")
    left.set_image(np.ones((3, 3)))
    right_image = np.full((4, 4), 2.0)
    right.set_image(right_image)
    observed = []

    def record():
        observed.append(api.current_virtual_image)

    connection = api.signal_current_virtual_image_changed.connect(record)
    try:
        left.close()
        assert len(observed) == 1
        assert observed[0] is right_image
        assert viewer.realspace_title.text() == "Right"
    finally:
        api.signal_current_virtual_image_changed.disconnect(connection)
        left.close()
        right.close()


@pytest.mark.parametrize("shape", [(5, 7), (8, 5), (7, 8)])
def test_fft_calibration_ring_centers_on_dc_pixel(viewer, shape):
    from PyQt5.QtCore import QPointF
    from py4d_browser_plugin.calibration_plugin.calibration_plugin import (
        CalibrateDialog,
        CalibrateFromPeriodDialog,
    )

    api, state = _setup_dataset(viewer)
    viewer.set_virtual_image(np.arange(np.prod(shape)).reshape(shape))
    parent = CalibrateDialog(api)
    dialog = CalibrateFromPeriodDialog(api, parent)
    try:
        center = dialog.roi.pos() + dialog.roi.size() / 2
        item = dialog.image_widget.getImageItem()
        dc = item.mapFromParent(QPointF(center))
        assert dc.x() == pytest.approx(shape[1] // 2 + 0.5)
        assert dc.y() == pytest.approx(shape[0] // 2 + 0.5)
    finally:
        dialog.deleteLater()
        parent.deleteLater()
        _teardown_dataset(viewer, state)


def test_volume_fft_ignores_3d_rgb_virtual_image(viewer):
    # "Virtual Image FFT" must treat a 3D *RGB* virtual image (shape (H, W, 3),
    # also ndim == 3) as a 2D image, NOT as a genuine 3D volume. The 3D branch
    # keys on the *visible tab being a VolumeTab*, not on ndim alone — so a
    # 3D RGB image (built-in pane or image tab) falls through to the 2D-only
    # path and no-ops, exactly as the pre-3D baseline did. (Without that guard,
    # the channel axis would be FFT'd as a fake z-axis.)
    pytest.importorskip("OpenGL")
    api, state = _setup_dataset(viewer)
    try:
        # start clean so the assertions exercise the "not routed" path
        viewer._result_volume_item = None
        viewer._result_volume_fft = None

        # overwrite the 2D virtual image with a 3D RGB one (H, W, 3) in the
        # built-in (default) virtual-image pane
        viewer.set_virtual_image(np.random.rand(4, 5, 3).astype(np.float32), reset=True)

        fft_before = viewer.unscaled_fft_image
        _select_result_mode(viewer, "Virtual Image FFT")

        # NOT misrouted into the 3D volume FFT
        assert viewer._result_volume_item is None
        assert viewer._result_volume_fft is None
        # the pane is on the 2D page (the 2D-only path no-ops on a 3D image)
        # and the 2D result image was left untouched
        assert viewer._result_stack.currentIndex() == 0
        assert viewer.unscaled_fft_image is fft_before
        # and the RGB virtual image is still a 3D array (rendered as RGB in the
        # pane) — RGB virtual-image support is intact
        assert viewer.current_virtual_image.shape == (4, 5, 3)
    finally:
        _teardown_dataset(viewer, state)


def test_volume_fft_ignores_3d_rgb_image_tab(viewer):
    # Same guard, but the 3D RGB image lives in a *plugin image tab*. A
    # VirtualImageTab is not a VolumeTab, so the 3D volume FFT must not fire
    # either — the discriminator is the visible tab's type, not the array shape.
    pytest.importorskip("OpenGL")
    api, state = _setup_dataset(viewer)
    tab = api.create_virtual_image_tab("RGB")
    try:
        viewer._result_volume_item = None
        viewer._result_volume_fft = None

        tab.set_image(np.random.rand(4, 5, 3).astype(np.float32), reset=True)
        viewer.virtual_image_tab_widget.setCurrentIndex(
            viewer.virtual_image_tabs.index(tab) + 1
        )
        _select_result_mode(viewer, "Virtual Image FFT")

        assert viewer._result_volume_item is None
        assert viewer._result_volume_fft is None
        assert viewer._result_stack.currentIndex() == 0
    finally:
        viewer.virtual_image_tab_widget.setCurrentIndex(0)
        api.close_virtual_image_tab(tab)
        _teardown_dataset(viewer, state)


def test_volume_fft_without_dataset(viewer):
    # Unit input voxels give reciprocal spacing of 1/N on each FFT axis.
    pytest.importorskip("OpenGL")
    from pyqtgraph import opengl as gl

    api = PluginAPI(viewer)
    vol = api.create_volume_tab("Vol")
    prev_dc = viewer.datacube
    prev_mode = viewer.result_source_action_group.checkedAction()
    try:
        viewer.datacube = None  # simulate opening a TIFF stack, no dataset
        vol.set_volume(np.random.rand(4, 5, 6).astype(np.float32), reset=True)
        viewer.virtual_image_tab_widget.setCurrentIndex(
            viewer.virtual_image_tabs.index(vol) + 1
        )
        _select_result_mode(viewer, "Virtual Image FFT")
        assert viewer._result_stack.currentIndex() == 1
        item = viewer._result_volume_item
        assert isinstance(item, gl.GLVolumeItem)
        assert item.data.shape == (4, 5, 6, 4)
        spacing = np.diag(item.transform().matrix(3))[:3]
        assert spacing == pytest.approx([1 / 4, 1 / 5, 1 / 6])
    finally:
        viewer.virtual_image_tab_widget.setCurrentIndex(0)
        api.close_virtual_image_tab(vol)
        viewer.datacube = prev_dc
        prev_mode.trigger()


def test_volume_fft_switching_back_to_2d_mode(viewer):
    # "Virtual Image FFT" with a 3D volume visible shows the 3D page; selecting
    # a 2D-only mode (with the 3D volume still visible) falls back to the 2D
    # page and drops the view lock.
    pytest.importorskip("OpenGL")
    api, state = _setup_dataset(viewer)
    vol = api.create_volume_tab("Vol")
    try:
        vol.set_volume(np.random.rand(4, 5, 6).astype(np.float32), reset=True)
        viewer.virtual_image_tab_widget.setCurrentIndex(
            viewer.virtual_image_tabs.index(vol) + 1
        )

        _select_result_mode(viewer, "Virtual Image FFT")
        assert viewer._result_stack.currentIndex() == 1

        # "Virtual Image FFT (complex)" is 2D-only, so with the 3D volume still
        # visible it can't render and falls back to the 2D page.
        _select_result_mode(viewer, "Virtual Image FFT (complex)")
        assert viewer._result_stack.currentIndex() == 0
        assert not viewer._view_sync._active
    finally:
        viewer.virtual_image_tab_widget.setCurrentIndex(0)
        api.close_virtual_image_tab(vol)
        _teardown_dataset(viewer, state)


def test_volume_fft_updates_reciprocal_z_spacing_live(viewer):
    pytest.importorskip("OpenGL")
    api, state = _setup_dataset(viewer)
    vol = api.create_volume_tab("Vol")
    try:
        vol.set_volume(
            np.random.rand(4, 5, 6).astype(np.float32), voxel_size=(2, 3, 4)
        )
        viewer.virtual_image_tab_widget.setCurrentIndex(
            viewer.virtual_image_tabs.index(vol) + 1
        )
        _select_result_mode(viewer, "Virtual Image FFT")
        item = viewer._result_volume_item
        assert item is not None

        assert np.diag(item.transform().matrix(3))[:3] == pytest.approx(
            [1 / 8, 1 / 15, 1 / 24]
        )

        rgba = item.data
        vol._z_slider.setValue(vol._z_slider.maximum())
        new_z = vol.voxel_size[0]
        assert new_z > 2.0
        assert np.diag(item.transform().matrix(3))[:3] == pytest.approx(
            [1 / (4 * new_z), 1 / 15, 1 / 24]
        )
        assert item.data is rgba

        # selecting a 2D-only mode makes the z-slider a no-op for the result FFT
        transform_2d = np.array(item.transform().matrix(3)).copy()
        _select_result_mode(viewer, "Virtual Image FFT (complex)")
        vol._z_slider.setValue(vol._z_slider.minimum())
        assert np.array_equal(np.array(item.transform().matrix(3)), transform_2d)
    finally:
        viewer.virtual_image_tab_widget.setCurrentIndex(0)
        api.close_virtual_image_tab(vol)
        _teardown_dataset(viewer, state)


def test_view_direction_sync(viewer, qapp):
    # ViewDirectionSync copies elevation/azimuth between the two views in
    # both directions, leaves zoom independent per view, and is a no-op
    # while inactive.
    pytest.importorskip("OpenGL")
    from PyQt5 import QtCore, QtGui
    from pyqtgraph import opengl as gl
    from py4D_browser.result_pane import ViewDirectionSync

    view_a = gl.GLViewWidget()
    view_b = gl.GLViewWidget()
    sync = ViewDirectionSync(view_a)

    def drive(view, etype):
        ev = QtGui.QMouseEvent(
            etype,
            QtCore.QPoint(10, 10),
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
        )
        sync.eventFilter(view, ev)
        qapp.processEvents()

    try:
        sync.set_active(True, view_b)

        # drive A: B's direction follows
        view_a.setCameraParams(elevation=30.0, azimuth=45.0)
        drive(view_a, QtCore.QEvent.MouseButtonPress)
        pa, pb = view_a.cameraParams(), view_b.cameraParams()
        assert pa["elevation"] == pytest.approx(pb["elevation"])
        assert pa["azimuth"] == pytest.approx(pb["azimuth"])

        # drive B: A's direction follows (bidirectional)
        view_b.setCameraParams(elevation=-20.0, azimuth=130.0)
        drive(view_b, QtCore.QEvent.MouseButtonRelease)
        pa, pb = view_a.cameraParams(), view_b.cameraParams()
        assert pa["elevation"] == pytest.approx(pb["elevation"])
        assert pa["azimuth"] == pytest.approx(pb["azimuth"])

        # zoom stays independent per view
        view_b.setCameraParams(distance=view_b.cameraParams()["distance"] * 2.0)
        drive(view_b, QtCore.QEvent.Wheel)
        pa, pb = view_a.cameraParams(), view_b.cameraParams()
        assert pa["elevation"] == pytest.approx(pb["elevation"])
        assert pa["distance"] != pb["distance"]

        # inactive: no propagation
        sync.set_active(False)
        view_a.setCameraParams(elevation=60.0, azimuth=10.0)
        drive(view_a, QtCore.QEvent.MouseButtonPress)
        pb = view_b.cameraParams()
        assert pb["elevation"] == pytest.approx(-20.0)
        assert pb["azimuth"] == pytest.approx(130.0)
    finally:
        sync.deleteLater()
        view_a.deleteLater()
        view_b.deleteLater()
        qapp.processEvents()
