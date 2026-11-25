"""Tests for the TIFF volume-stack viewer plugin and its menu action."""

import numpy as np
import pytest

pytest.importorskip("OpenGL")
pytest.importorskip("tifffile")


PLUGIN_ID = "py4DGUI.internal.volume_viewer"


def _loaded_ids(viewer):
    return {entry["id"] for entry in viewer.loaded_plugins}


def _find_plugin(viewer, plugin_id):
    for entry in viewer.loaded_plugins:
        if entry["id"] == plugin_id:
            return entry
    return None


########## dev-plugin gating ##########


def test_volume_viewer_loads_when_dev_plugins_disabled(viewer):
    viewer.settings.setValue("gui/dev_plugins", "0")
    viewer.load_plugins()

    assert PLUGIN_ID in _loaded_ids(viewer)


def test_volume_viewer_loads_when_dev_plugins_enabled(viewer):
    viewer.settings.setValue("gui/dev_plugins", "1")
    try:
        viewer.load_plugins()
        assert PLUGIN_ID in _loaded_ids(viewer)
    finally:
        # restore the default so the session sees dev plugins off again
        viewer.settings.setValue("gui/dev_plugins", "0")
        viewer.load_plugins()


########## opening a TIFF stack ##########


def test_open_tiff_stack_creates_volume_tab(viewer, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog

    # A small 3D stack, written as a multi-page (one page per z slice) TIFF.
    volume = np.random.default_rng(0).random((6, 8, 4), dtype=np.float32)
    stack_path = str(tmp_path / "stack.tif")
    import tifffile

    tifffile.imwrite(stack_path, volume, photometric="minisblack")

    def fake_get_open_file_name(parent, caption, directory, filter):
        return (stack_path, "TIFF images (*.tif *.tiff);;All files (*)")

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(fake_get_open_file_name)
    )

    opened_tabs = []
    try:
        viewer.load_plugins()
        entry = _find_plugin(viewer, PLUGIN_ID)
        assert entry is not None
        plugin = entry["plugin"]
        action = entry["action"]
        assert action in viewer.processing_menu.actions()

        # Snapshot the count before triggering: viewer.virtual_image_tabs is
        # the live list, so binding it directly would just alias the new tab in.
        count_before = len(viewer.virtual_image_tabs)
        expected_title = plugin._next_title("Volume")
        action.trigger()

        assert len(viewer.virtual_image_tabs) == count_before + 1
        tab = viewer.virtual_image_tabs[-1]
        opened_tabs.append(tab)
        assert tab.title == expected_title
        assert tab.volume is not None
        assert np.array_equal(tab.volume, volume)
        # and it is a volume tab, shown selected. (Index 0 of the pane is the
        # built-in virtual image, so the plugin tabs start at 1.)
        assert hasattr(tab, "set_volume")
        assert viewer.virtual_image_tab_widget.currentIndex() == len(
            viewer.virtual_image_tabs
        )
    finally:
        for tab in opened_tabs:
            if not tab.closed:
                tab.close()


def test_open_2d_tiff_is_rejected_without_creating_a_tab(viewer, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog

    image_2d = np.random.default_rng(0).random((8, 8), dtype=np.float32)
    stack_path = str(tmp_path / "image2d.tif")
    import tifffile

    tifffile.imwrite(stack_path, image_2d)

    def fake_get_open_file_name(parent, caption, directory, filter):
        return (stack_path, "TIFF images (*.tif *.tiff);;All files (*)")

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(fake_get_open_file_name)
    )

    viewer.load_plugins()
    plugin = _find_plugin(viewer, PLUGIN_ID)["plugin"]

    count_before = len(viewer.virtual_image_tabs)
    plugin.open_tiff_stack()
    assert len(viewer.virtual_image_tabs) == count_before
