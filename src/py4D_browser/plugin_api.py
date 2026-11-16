"""
The versioned plugin API for py4DGUI.

Plugins declare which version of this API they were written against with a
class-level attribute::

    api_version = (1, 6)

The loader resolves that against :data:`SUPPORTED_API_VERSIONS` and passes
the plugin an instance of the matching API class. Within a major version the
API is only ever extended: existing methods keep their signatures and
behavior, so a plugin written against an older minor version of a major
continues to work.

Multiple major versions coexist: when a new major version is introduced, the
old class is kept in the registry so that plugins written against it keep
loading. The API objects are the "glue" between plugins and the DataViewer:
their outgoing interface is stable for the lifetime of the major version,
while their implementation (which calls into the viewer) may be updated as
the GUI evolves.

Plugins must declare ``api_version``; a plugin that does not, or that
requires a version this browser does not provide, is logged and skipped.
Plugins that set ``full_access = True`` additionally receive the raw
``DataViewer`` as ``parent`` (bypassing the versioned surface); everyone
else interacts with the browser solely through the API object passed as
``api``.
"""

from functools import partial
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QWidget

if TYPE_CHECKING:
    from py4D_browser.main_window import DataViewer

__all__ = [
    "API_VERSION",
    "PluginAPI",
    "PluginAPI1",
    "PluginAPIVersionError",
    "SUPPORTED_API_VERSIONS",
    "resolve_api_class",
]


class PluginAPIVersionError(Exception):
    """
    Raised when a plugin declares an API version that this browser does not
    provide, i.e. the plugin is newer than the GUI.
    """


class PluginAPI1:
    """
    The version 1.6 plugin API (major version 1).

    Instances are created by the loader (or by hand for testing) with a
    reference to the DataViewer, and passed to plugins as ``api``. All
    state lives in the viewer; this object is a thin, stable facade over it.
    """

    api_version = (1, 6)

    def __init__(self, viewer: "DataViewer"):
        # Hold a strong reference to the viewer; the API is a facade over it.
        self._viewer = viewer

        # Signals: direct references to the viewer's bound signals, so
        # connecting/disconnecting through the API acts on the same signal.
        self.signal_diffraction_data_changed = (
            viewer.signal_diffraction_data_changed
        )
        self.signal_virtual_image_data_changed = (
            viewer.signal_virtual_image_data_changed
        )
        self.signal_current_virtual_image_changed = (
            viewer.signal_current_virtual_image_changed
        )
        self.signal_datacube_changed = viewer.signal_datacube_changed

        # Pane setters: bound to the viewer; signatures match the
        # corresponding DataViewer methods (see update_views.py).
        self.set_virtual_image = partial(viewer.set_virtual_image)
        self.set_diffraction_image = partial(viewer.set_diffraction_image)
        self.set_result_image = partial(viewer.set_result_image)
        self.set_scalebar = partial(viewer.set_scalebar)
        # Recompute the result-pane scale bar from the current calibration.
        self.refresh_result_scale_bar = partial(viewer.refresh_result_scale_bar)

        # Replace the currently loaded datacube (see set_datacube in
        # menu_actions.py). `refresh` (default True) triggers the normal
        # post-load machinery: view resets + signal_datacube_changed.
        self.set_datacube = partial(viewer.set_datacube)

        # Result-pane registration (see register_result_callback in
        # signals.py): lets a plugin drive the result pane instead of the
        # built-in FFT/EWPC. The registration is a singleton — registering a
        # new callback replaces the previous one (and runs its cleanup).
        self.register_result_callback = partial(viewer.register_result_callback)

        # Attach/detach pyqtgraph items (ROIs, lines, …) to the built-in
        # diffraction and real-space views. These are the supported way for a
        # plugin to place an ROI in a built-in view; do not reach into the
        # viewer's widgets directly.
        self.add_diffraction_roi = partial(viewer.add_diffraction_roi)
        self.remove_diffraction_roi = partial(viewer.remove_diffraction_roi)
        self.add_real_space_roi = partial(viewer.add_real_space_roi)
        self.remove_real_space_roi = partial(viewer.remove_real_space_roi)

        # Detector getters (see "Accessing the detectors" in PLUGINS.md)
        self.get_diffraction_detector = partial(viewer.get_diffraction_detector)
        self.get_virtual_image_detector = partial(viewer.get_virtual_image_detector)

        # Virtual-image tabs (v1.6): let an image-producing plugin show its
        # output in its own tab instead of overwriting the built-in virtual
        # image. See the "Virtual-image tabs" section in PLUGINS.md.
        self.create_virtual_image_tab = partial(viewer.create_virtual_image_tab)
        self.close_virtual_image_tab = partial(viewer.close_virtual_image_tab)

        # 3D volume tabs (v1.6): render a 3D volume in a tab alongside the
        # built-in virtual image, with user-adjustable color/alpha transfer
        # functions. See the "3D volume tabs" section in PLUGINS.md. (Closing
        # works through the same close_virtual_image_tab as image tabs.)
        self.create_volume_tab = partial(viewer.create_volume_tab)

        # Qt plumbing
        self.qtapp = viewer.qtapp
        # A dialog-parent widget (see the `qt_window` property), created
        # lazily so a plugin that never shows a dialog doesn't leave a stray
        # child widget on the viewer.
        self._dialog_parent = None
        self.status_bar = viewer.statusBar()
        # QSettings object; only write under a top-level section named
        # after your plugin_id.
        self.settings = viewer.settings

    @property
    def datacube(self):
        """
        The currently loaded DataCube (read-only through the API).
        Mutating the cube object itself is fine; to replace the whole
        cube, use ``set_datacube``.
        """
        return self._viewer.datacube

    @datacube.setter
    def datacube(self, value):
        raise AttributeError(
            "The datacube cannot be assigned; use "
            "api.set_datacube(datacube) to replace the currently loaded "
            "datacube."
        )

    @property
    def virtual_image_tabs(self):
        """
        A copy of the list of open plugin virtual-image tabs (the default,
        built-in tab is not included). May contain both image tabs
        (:class:`~py4D_browser.virtual_image_tabs.VirtualImageTab`) and
        volume tabs (:class:`~py4D_browser.volume_tabs.VolumeTab`);
        distinguish them with ``isinstance``. Read-only; close a tab with
        :meth:`close_virtual_image_tab` or ``tab.close()``.
        """
        return list(self._viewer.virtual_image_tabs)

    @property
    def current_virtual_image(self):
        """
        The raw array currently on display in the virtual-image pane: the
        built-in virtual image when the default tab is visible, else the
        visible plugin tab's raw array (or ``None`` if that tab has no data).
        This is a 2D image for the built-in tab and image tabs, but a 3D
        volume when a volume tab is visible.
        """
        return self._viewer.current_virtual_image

    @property
    def current_virtual_image_settings(self):
        """
        The display settings of the virtual-image view currently on screen,
        as a :class:`~py4D_browser.utils.VirtualImageSettings` record: the
        built-in pane's scaling / Autorange / autoscale percentiles / levels
        when the default tab is visible, else the visible image tab's.

        Returns ``None`` when the visible view has no 2D display settings —
        an empty built-in pane or image tab, or a volume tab. Use this to read
        a view's current scaling and range (e.g. to mirror them onto a new
        tab); each tab's own ``scaling`` / ``autorange`` /
        ``autorange_percentiles`` / ``levels`` / ``set_levels`` control its
        display independently.
        """
        return self._viewer.current_virtual_image_settings

    @property
    def current_diffraction_image_settings(self):
        """
        The display settings of the built-in diffraction (reciprocal-space)
        pane, as a :class:`~py4D_browser.utils.VirtualImageSettings` record:
        the pane's scaling / Autorange / autoscale percentiles / levels.

        Returns ``None`` when the pane is empty (no diffraction image loaded).
        Use this to read the DP pane's current scaling and range (e.g. to
        mirror them onto a new tab); each tab's own ``scaling`` / ``autorange``
        / ``autorange_percentiles`` / ``levels`` / ``set_levels`` control its
        display independently.
        """
        return self._viewer.current_diffraction_image_settings

    @property
    def current_diffraction_image(self):
        """
        The raw array currently on display in the diffraction pane
        (``None`` if no image is loaded).
        """
        return self._viewer.unscaled_diffraction_image

    @property
    def diffraction_image_shape(self):
        """
        The shape of the raw (unscaled) diffraction image currently on display,
        or ``None`` if no image is loaded. Read-only; shapes are exposed
        (rather than the full arrays) so a plugin can size ROIs against the
        view without retaining a reference to a large array.
        """
        image = self._viewer.unscaled_diffraction_image
        return None if image is None else image.shape

    @property
    def real_space_image_shape(self):
        """
        The shape of the raw (unscaled) real-space (virtual image) image
        currently on display in the built-in pane, or ``None`` if no image is
        loaded. Read-only; shapes are exposed (rather than the full array) so a
        plugin can size ROIs against the view without retaining a reference to
        a large array.
        """
        image = self._viewer.unscaled_realspace_image
        return None if image is None else image.shape

    @property
    def qt_window(self):
        """
        A widget to use as the parent for dialogs, e.g.
        ``QDialog(parent=api.qt_window)``.

        This is deliberately **not** the DataViewer itself. It is a plain
        ``QWidget`` (a child of the main window) whose only job is to serve as
        a dialog parent, so a general plugin cannot reach the viewer's full
        state through it — data access must go through the versioned API
        surface above. A plugin that sets ``full_access = True`` still
        receives the raw DataViewer directly as ``parent``.
        """
        if self._dialog_parent is None:
            self._dialog_parent = QWidget(self._viewer)
        return self._dialog_parent


# Major version -> (highest minor version provided, API class).
# New major versions are added here; older ones are kept so that plugins
# written against them continue to load.
SUPPORTED_API_VERSIONS = {
    1: (6, PluginAPI1),
}

# The newest API version provided by this browser.
API_VERSION = (
    max(SUPPORTED_API_VERSIONS),
    SUPPORTED_API_VERSIONS[max(SUPPORTED_API_VERSIONS)][0],
)

# Convenience alias for the latest API class.
PluginAPI = SUPPORTED_API_VERSIONS[API_VERSION[0]][1]


def resolve_api_class(api_version):
    """
    Return the API class to serve a plugin requiring the ``(major, minor)``
    ``api_version``, or raise :class:`PluginAPIVersionError` if this browser
    does not provide it (major absent, or minor ahead of the latest).

    A minor version at or below the latest for a major is compatible: within
    a major the API is only ever extended.
    """
    try:
        major, minor = int(api_version[0]), int(api_version[1])
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise PluginAPIVersionError(
            f"api_version must be a (major, minor) pair of integers, "
            f"got {api_version!r}"
        ) from exc

    entry = SUPPORTED_API_VERSIONS.get(major)
    if entry is None:
        supported = ", ".join(
            f"v{major}.{minor}"
            for major, (minor, _) in sorted(SUPPORTED_API_VERSIONS.items())
        )
        raise PluginAPIVersionError(
            f"requires API v{major}.{minor}, but this browser only provides "
            f"{supported}. Please upgrade py4D_browser to a version that "
            "provides it."
        )

    latest_minor, cls = entry
    if minor > latest_minor:
        raise PluginAPIVersionError(
            f"requires API v{major}.{minor}, but this browser only provides "
            f"v{major}.{latest_minor}. Please upgrade py4D_browser to a "
            "version that provides it."
        )

    return cls
