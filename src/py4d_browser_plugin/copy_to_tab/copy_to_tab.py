from PyQt5.QtWidgets import QWidget


class CopyToTabPlugin(QWidget):
    """
    Copies the raw array currently on display — the virtual image or the
    diffraction pattern — into a new virtual-image tab, so a user can keep it
    alongside the live display without the plugin overwriting anything.
    """

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "py4DGUI.internal.copy_to_tab"
    api_version = (1, 6)
    uses_plugin_menu = True
    display_name = "Copy Image to Tab"

    def __init__(self, api, plugin_menu, **kwargs):
        super().__init__()
        self.api = api
        # Note: we do not keep a list of the tabs we created. A closed tab
        # is not freed by the browser (it still holds its raw image array
        # in `tab.image`), so retaining it would pin that memory until exit.
        copy_vi_action = plugin_menu.addAction("Copy Virtual Image to Tab")
        copy_vi_action.triggered.connect(self.copy_virtual_image_to_tab)
        copy_dp_action = plugin_menu.addAction("Copy Diffraction Pattern to Tab")
        copy_dp_action.triggered.connect(self.copy_diffraction_pattern_to_tab)

    def close(self):
        pass

    def _next_title(self, prefix):
        # Number copies per prefix (the first "VI Copy" is "VI Copy 1",
        # independently of any "DP Copy" tabs), taking the next free number.
        tabs = self.api.virtual_image_tabs
        n = 1
        while any(tab.title == f"{prefix} {n}" for tab in tabs):
            n += 1
        return f"{prefix} {n}"

    def copy_virtual_image_to_tab(self):
        api = self.api
        image = api.current_virtual_image
        if image is None:
            api.status_bar.showMessage(
                "No virtual image to copy — display one first.", 5_000
            )
            return
        if image.ndim != 2:
            # A 3D volume tab is on display; its array can't go into a 2D
            # virtual-image tab.
            api.status_bar.showMessage(
                "The visible virtual image is a 3D volume; it can't be "
                "copied to an image tab.",
                5_000,
            )
            return
        title = self._next_title("VI Copy")
        tab = api.create_virtual_image_tab(title)

        # Mirror the visible view's scaling + range (not its colormap) so the
        # copy looks like its source. Set before set_image so the tab's first
        # render uses the mirrored scaling and percentiles; the final set_levels
        # (below) then pins it to the source's exact on-screen range.
        settings = api.current_virtual_image_settings
        if settings is not None:
            tab.scaling = settings.scaling
            tab.autorange_percentiles = settings.percentiles
            tab.autorange = settings.autorange

        set_scalebar = {}
        if api.datacube is not None:
            set_scalebar = dict(
                pixel_size=api.datacube.calibration.get_R_pixel_size(),
                pixel_units=api.datacube.calibration.get_R_pixel_units(),
            )
        tab.set_image(image, reset=True, **set_scalebar)
        if settings is not None and settings.levels is not None:
            tab.set_levels(*settings.levels)
        api.status_bar.showMessage(
            f"Copied the current virtual image to the '{title}' tab.", 5_000
        )

    def copy_diffraction_pattern_to_tab(self):
        api = self.api
        image = api.current_diffraction_image
        if image is None:
            api.status_bar.showMessage(
                "No diffraction pattern to copy — display one first.", 5_000
            )
            return
        title = self._next_title("DP Copy")
        tab = api.create_virtual_image_tab(title)

        # Mirror the diffraction pane's scaling + range (not its colormap) so
        # the copy looks like its source. Set before set_image so the tab's
        # first render uses the mirrored scaling and percentiles; the final
        # set_levels (below) then pins it to the source's exact on-screen range.
        settings = api.current_diffraction_image_settings
        if settings is not None:
            tab.scaling = settings.scaling
            tab.autorange_percentiles = settings.percentiles
            tab.autorange = settings.autorange

        set_scalebar = {}
        if api.datacube is not None:
            set_scalebar = dict(
                pixel_size=api.datacube.calibration.get_Q_pixel_size(),
                pixel_units=api.datacube.calibration.get_Q_pixel_units(),
            )
        tab.set_image(image, reset=True, **set_scalebar)
        if settings is not None and settings.levels is not None:
            tab.set_levels(*settings.levels)
        api.status_bar.showMessage(
            f"Copied the current diffraction pattern to the '{title}' tab.",
            5_000,
        )
