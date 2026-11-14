"""
A development-only plugin: load a 3D volume from a multi-page TIFF image
stack (via a file picker) and display it in a new 3D volume tab.

Useful for trying out the volume-tab rendering (API v1.6) with arbitrary
volumes without loading a 4D-STEM dataset. It is ``dev_only``: it loads
only when the ``gui/dev_plugins`` setting is enabled.
"""

import numpy as np
import tifffile
from PyQt5.QtWidgets import QFileDialog, QWidget


class VolumeViewerPlugin(QWidget):
    """
    Opens a file picker for a TIFF image stack, reads it as a 3D volume, and
    displays it in a new 3D volume tab.
    """

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "py4DGUI.internal.volume_viewer"
    api_version = (1, 6)
    uses_single_action = True
    display_name = "Open TIFF stack as volume..."
    dev_only = True

    def __init__(self, api, plugin_action, **kwargs):
        super().__init__()
        self.api = api
        plugin_action.triggered.connect(self.open_tiff_stack)

    def close(self):
        pass

    def _next_title(self, prefix):
        # Number tabs by prefix, taking the next free number.
        tabs = self.api.virtual_image_tabs
        n = 1
        while any(tab.title == f"{prefix} {n}" for tab in tabs):
            n += 1
        return f"{prefix} {n}"

    def open_tiff_stack(self):
        api = self.api
        path, _ = QFileDialog.getOpenFileName(
            api.qt_window,
            "Open TIFF stack",
            "",
            "TIFF images (*.tif *.tiff);;All files (*)",
        )
        if not path:
            return

        try:
            volume = np.asarray(tifffile.imread(path))
        except Exception as exc:
            api.status_bar.showMessage(
                f"Failed to read '{path}': {exc}", 5_000
            )
            return

        # Validate before creating the tab, so a bad file doesn't leave an
        # empty volume tab behind.
        if np.iscomplexobj(volume):
            api.status_bar.showMessage(
                f"'{path}' contains complex data, which a volume tab cannot "
                "render.",
                5_000,
            )
            return
        if volume.ndim != 3:
            api.status_bar.showMessage(
                f"'{path}' is a {volume.ndim}D array; a 3D stack (x, y, z) "
                "is required.",
                5_000,
            )
            return

        # Note: we do not keep a reference to the tab we create. A closed
        # tab is not freed by the browser (it still holds its raw volume
        # array in `tab.image`), so retaining it would pin that memory until
        # exit.
        title = self._next_title("Volume")
        tab = api.create_volume_tab(title, select=True)
        tab.set_volume(volume, voxel_size=(5,1,1))
        api.status_bar.showMessage(
            f"Loaded a {volume.shape[0]}×{volume.shape[1]}×{volume.shape[2]} "
            f"volume from '{path}' into the '{title}' tab.",
            5_000,
        )
