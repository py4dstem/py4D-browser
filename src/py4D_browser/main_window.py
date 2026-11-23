from typing import Optional
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QMenu,
    QAction,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QTabBar,
    QStyle,
    QActionGroup,
    QLabel,
    QToolTip,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QStackedLayout,
)

from py4DSTEM import DataCube
import pyqtgraph as pg
from pyqtgraph import opengl as gl
import numpy as np

from functools import partial
from pathlib import Path
import os
import gc
import sys
import platformdirs
from showinfm import show_in_file_manager

from py4D_browser.utils import (
    VLine,
    LatchingButton,
    strtobool,
    try_get_cmap,
    VirtualImageSettings,
)
from py4D_browser.version import __version__
from py4D_browser.scalebar import ScaleBar
from py4D_browser.virtual_image_tabs import VirtualImageTab
from py4D_browser.volume_tabs import VolumeTab


class DataViewer(QMainWindow):
    """
    The class is used by instantiating and then entering the main Qt loop with, e.g.:
        win = DataViewer()
        win.show()
        app.exec_()
    """

    # Canonical value lives in utils.py (used by the shared ``scale_image``
    # helper); re-exported here as a class attribute so the diffraction-space
    # renderers can keep referencing ``self.LOG_SCALE_MIN_VALUE``.
    from py4D_browser.utils import LOG_SCALE_MIN_VALUE

    from py4D_browser.menu_actions import (
        load_file,
        load_data_arina,
        load_data_auto,
        load_data_bin,
        load_data_mmap,
        show_file_dialog,
        get_savefile_name,
        export_datacube,
        export_virtual_image,
        show_keyboard_map,
        reshape_data,
        set_datacube,
        copy_vimg_to_clipboard,
        copy_diff_to_clipboard,
        copy_result_to_clipboard,
    )

    from py4D_browser.update_views import (
        set_virtual_image,
        set_diffraction_image,
        set_scalebar,
        _render_virtual_image,
        _render_diffraction_image,
        update_diffraction_space_view,
        update_real_space_view,
        set_diffraction_autoscale_range,
        set_real_space_autoscale_range,
        update_tooltip,
        update_scalebars,
    )

    from py4D_browser.detectors import (
        get_diffraction_detector,
        get_virtual_image_detector,
        add_diffraction_roi,
        remove_diffraction_roi,
        add_real_space_roi,
        remove_real_space_roi,
        update_realspace_detector,
        update_diffraction_detector,
        nudge_real_space_selector,
        nudge_diffraction_selector,
        center_diffraction_selector_at,
        center_real_space_selector_at,
        on_diffraction_double_click,
        on_real_space_double_click,
        update_annulus_pos,
        update_annulus_radii,
    )

    from py4D_browser.result_pane import (
        set_result_image,
        refresh_result_scale_bar,
        _render_result_image,
        _render_result_volume,
        _update_result_volume_display,
        _result_volume_z_stretch,
        _apply_result_volume_geometry,
        _on_volume_voxel_size_changed,
        _visible_volume_tab,
        _enable_view_lock,
        _disable_view_lock,
        update_fft_view,
        set_result_autoscale_range,
        ViewDirectionSync,
    )

    from py4D_browser.signals import (
        register_result_callback,
        set_internal_result_callback,
    )

    from py4D_browser.plugins import load_plugins, unload_plugins

    signal_diffraction_data_changed = QtCore.pyqtSignal()
    signal_virtual_image_data_changed = QtCore.pyqtSignal()
    # Emitted whenever the *currently visible* virtual image changes: on an
    # update to the built-in (default) tab, and on an update to, or switch to,
    # a plugin tab. This is the single signal the result pane is driven by.
    signal_current_virtual_image_changed = QtCore.pyqtSignal()
    signal_datacube_changed = QtCore.pyqtSignal()

    def __init__(
        self,
        filepath: Optional[str] = None,
        reset_state: bool = False,
        debug_console: bool = False,
    ):
        super().__init__()
        # Define this as the QApplication object
        self.qtapp = QApplication.instance()
        if not self.qtapp:
            self.qtapp = QApplication(sys.argv)

        if sys.platform.startswith("darwin"):
            # macOS's default (legacy) GL context lacks GL_TEXTURE_3D / GLSL
            # 1.4, which pyqtgraph's 3D volume rendering needs. Request a
            # modern Core profile as the process default — this must be set
            # before any QOpenGLWidget (e.g. a volume tab's GLViewWidget)
            # exists. pyqtgraph's own GLVolumeItem example does the same.
            fmt = QtGui.QSurfaceFormat()
            fmt.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)
            fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            fmt.setVersion(4, 1)
            QtGui.QSurfaceFormat.setDefaultFormat(fmt)

        # Load settings from config file
        self.config_path = os.path.join(
            platformdirs.user_config_dir("py4DGUI", "py4DSTEM"), "GUI_config.ini"
        )
        print(f"Loading configuration from {self.config_path}")
        QtCore.QCoreApplication.setOrganizationName("py4DSTEM")
        QtCore.QCoreApplication.setOrganizationDomain("py4DSTEM.com")
        QtCore.QCoreApplication.setApplicationName("py4DGUI")
        self.settings = QtCore.QSettings(
            self.config_path, QtCore.QSettings.Format.IniFormat
        )

        self.setWindowTitle("py4DSTEM")

        alternate_logo = strtobool(self.settings.value("gui/quack", "0"))
        icon = QtGui.QIcon(
            str(
                Path(__file__).parent.absolute()
                / ("logo.png" if not alternate_logo else "logo_alternate.png")
            )
        )
        self.setWindowIcon(icon)
        self.qtapp.setWindowIcon(icon)

        self.setWindowTitle("py4DSTEM")
        self.setAcceptDrops(True)

        self.datacube: Optional[DataCube] = None

        self.unscaled_diffraction_image: Optional[np.ndarray] = None
        self.unscaled_realspace_image: Optional[np.ndarray] = None
        self.unscaled_fft_image: Optional[np.ndarray] = None

        # Result-pane 3D FFT state (see _render_result_volume in update_views.py).
        # Kept separate from the 2D path (unscaled_fft_image / fft_widget);
        # None until a 3D FFT renders for the first time.
        self._result_volume_item = None  # the GLVolumeItem
        self._result_volume_fft = None  # last (unscaled) 3D FFT, for re-render
        self._result_volume_px = 1.0  # its reciprocal-space voxel size
        self._result_volume_shape = None  # rendered shape (for the z-stretch transform)

        # Plugin-created virtual-image tabs (image or volume; see
        # virtual_image_tabs.py / volume_tabs.py). The built-in virtual
        # image lives in the default tab (index 0 of
        # ``virtual_image_tab_widget``) and is NOT listed here; this list only
        # ever holds the plugin tabs, in the same order as the tab bar (offset
        # by one, since the default tab occupies index 0).
        self.virtual_image_tabs: list = []

        # Reset stored state if so asked:
        if reset_state:
            self.settings.remove("last_state")
            print("Cleared saved state, using defaults...")

        self.setup_menus()
        self.setup_views()

        # setup listener for tooltip
        self.tooltip_timer = pg.ThreadsafeTimer()
        self.tooltip_timer.timeout.connect(self.update_tooltip)
        self.tooltip_timer.start(1000 // 30)  # run tooltip at 30 Hz
        font = QtGui.QFont(self.font())
        font.setPointSize(10)
        QToolTip.setFont(font)

        self.resize(
            self.settings.value("last_state/window_size", QtCore.QSize(1000, 800)),
        )

        # (Potentially) load plugins
        self.load_plugins()

        self.show()

        # If a file was passed on the command line, open it
        if filepath is not None:
            self.load_file(filepath)

        # launch pyqtgraph's debug console if requested or environment variable exists
        if debug_console or os.environ.get("PY4DGUI_DEBUG"):
            pg.dbg(namespace={"main_window": self})

    def setup_menus(self):
        self.menu_bar = self.menuBar()

        # File menu
        self.file_menu = QMenu("&File", self)
        self.menu_bar.addMenu(self.file_menu)

        import_label = QAction("Import", self)
        import_label.setDisabled(True)
        self.file_menu.addAction(import_label)

        self.load_auto_action = QAction("&Load Data...", self)
        self.load_auto_action.triggered.connect(self.load_data_auto)
        self.file_menu.addAction(self.load_auto_action)
        self.load_auto_action.setShortcut(QtGui.QKeySequence("Ctrl+O"))

        self.load_mmap_action = QAction("Load &Memory Map...", self)
        self.load_mmap_action.triggered.connect(self.load_data_mmap)
        self.file_menu.addAction(self.load_mmap_action)

        self.load_binned_action = QAction("Load Data &Binned...", self)
        self.load_binned_action.triggered.connect(self.load_data_bin)
        self.file_menu.addAction(self.load_binned_action)

        self.load_arina_action = QAction("Load &Arina Data...", self)
        self.load_arina_action.triggered.connect(self.load_data_arina)
        self.file_menu.addAction(self.load_arina_action)

        self.reshape_data_action = QAction("&Reshape Data...", self)
        self.reshape_data_action.triggered.connect(self.reshape_data)
        self.file_menu.addAction(self.reshape_data_action)

        self.file_menu.addSeparator()

        export_label = QAction("Export", self)
        export_label.setDisabled(True)
        self.file_menu.addAction(export_label)

        # Submenu to export datacube
        datacube_export_menu = QMenu("Export Datacube", self)
        self.file_menu.addMenu(datacube_export_menu)
        for method in ["Raw float32", "py4DSTEM HDF5", "Plain HDF5"]:
            menu_item = datacube_export_menu.addAction(method)
            menu_item.triggered.connect(partial(self.export_datacube, method))
            if method == "py4DSTEM HDF5":
                menu_item.setShortcut(QtGui.QKeySequence("Ctrl+S"))

        # Submenu to export virtual image
        vimg_export_menu = QMenu("Export Virtual Image", self)
        self.file_menu.addMenu(vimg_export_menu)
        menu_item = vimg_export_menu.addAction("To clipboard")
        menu_item.triggered.connect(self.copy_vimg_to_clipboard)
        menu_item.setShortcut(QtGui.QKeySequence("Ctrl+C"))
        for method in ["PNG (display)", "TIFF (display)", "TIFF (raw)"]:
            menu_item = vimg_export_menu.addAction(method)
            menu_item.triggered.connect(
                partial(self.export_virtual_image, method, "image")
            )

        # Submenu to export diffraction
        vdiff_export_menu = QMenu("Export Diffraction Pattern", self)
        self.file_menu.addMenu(vdiff_export_menu)
        menu_item = vdiff_export_menu.addAction("To clipboard")
        menu_item.triggered.connect(self.copy_diff_to_clipboard)
        menu_item.setShortcut(QtGui.QKeySequence("Ctrl+Alt+C"))
        for method in ["PNG (display)", "TIFF (display)", "TIFF (raw)"]:
            menu_item = vdiff_export_menu.addAction(method)
            menu_item.triggered.connect(
                partial(self.export_virtual_image, method, "diffraction")
            )

        result_export_menu = QMenu("Export Result", self)
        self.file_menu.addMenu(result_export_menu)
        menu_item = result_export_menu.addAction("To clipboard")
        menu_item.triggered.connect(self.copy_result_to_clipboard)
        menu_item.setShortcut(QtGui.QKeySequence("Ctrl+Shift+C"))
        for method in ["PNG (display)", "TIFF (display)", "TIFF (raw)"]:
            menu_item = result_export_menu.addAction(method)
            menu_item.triggered.connect(
                partial(self.export_virtual_image, method, "result")
            )

        # Scaling Menu
        self.scaling_menu = QMenu("&Scaling", self)
        self.menu_bar.addMenu(self.scaling_menu)

        # Diffraction scaling
        diff_scaling_group = QActionGroup(self)
        diff_scaling_group.setExclusive(True)
        self.diff_scaling_group = diff_scaling_group
        diff_menu_separator = QAction("Diffraction", self)
        diff_menu_separator.setDisabled(True)
        self.scaling_menu.addAction(diff_menu_separator)

        diff_scale_linear_action = QAction("Linear", self)
        diff_scale_linear_action.setCheckable(True)
        diff_scale_linear_action.triggered.connect(
            partial(self._render_diffraction_image, True)
        )
        diff_scaling_group.addAction(diff_scale_linear_action)
        self.scaling_menu.addAction(diff_scale_linear_action)

        diff_scale_log_action = QAction("Log", self)
        diff_scale_log_action.setCheckable(True)
        diff_scale_log_action.triggered.connect(
            partial(self._render_diffraction_image, True)
        )
        diff_scaling_group.addAction(diff_scale_log_action)
        self.scaling_menu.addAction(diff_scale_log_action)

        diff_scale_sqrt_action = QAction("Square Root", self)
        diff_scale_sqrt_action.setCheckable(True)
        diff_scale_sqrt_action.triggered.connect(
            partial(self._render_diffraction_image, True)
        )
        diff_scaling_group.addAction(diff_scale_sqrt_action)
        diff_scale_sqrt_action.setChecked(True)
        self.scaling_menu.addAction(diff_scale_sqrt_action)

        self.scaling_menu.addSeparator()

        # Real space scaling
        vimg_scaling_group = QActionGroup(self)
        vimg_scaling_group.setExclusive(True)
        self.vimg_scaling_group = vimg_scaling_group

        vimg_menu_separator = QAction("Virtual Image", self)
        vimg_menu_separator.setDisabled(True)
        self.scaling_menu.addAction(vimg_menu_separator)

        vimg_scale_linear_action = QAction("Linear", self)
        self.vimg_scale_linear_action = vimg_scale_linear_action  # Save this one!
        vimg_scale_linear_action.setCheckable(True)
        vimg_scale_linear_action.setChecked(True)
        vimg_scale_linear_action.triggered.connect(
            partial(self._render_virtual_image, True)
        )
        vimg_scaling_group.addAction(vimg_scale_linear_action)
        self.scaling_menu.addAction(vimg_scale_linear_action)

        vimg_scale_log_action = QAction("Log", self)
        vimg_scale_log_action.setCheckable(True)
        vimg_scale_log_action.triggered.connect(
            partial(self._render_virtual_image, True)
        )
        vimg_scaling_group.addAction(vimg_scale_log_action)
        self.scaling_menu.addAction(vimg_scale_log_action)

        vimg_scale_sqrt_action = QAction("Square Root", self)
        vimg_scale_sqrt_action.setCheckable(True)
        vimg_scale_sqrt_action.triggered.connect(
            partial(self._render_virtual_image, True)
        )
        vimg_scaling_group.addAction(vimg_scale_sqrt_action)
        self.scaling_menu.addAction(vimg_scale_sqrt_action)

        self.scaling_menu.addSeparator()

        # Real space scaling
        result_scaling_group = QActionGroup(self)
        result_scaling_group.setExclusive(True)
        self.result_scaling_group = result_scaling_group

        result_menu_separator = QAction("Result", self)
        result_menu_separator.setDisabled(True)
        self.scaling_menu.addAction(result_menu_separator)

        result_scale_linear_action = QAction("Linear", self)
        self.result_scale_linear_action = result_scale_linear_action  # Save this one!
        result_scale_linear_action.setCheckable(True)
        result_scale_linear_action.triggered.connect(
            partial(self._render_result_image, True)
        )
        result_scaling_group.addAction(result_scale_linear_action)
        self.scaling_menu.addAction(result_scale_linear_action)

        result_scale_log_action = QAction("Log", self)
        result_scale_log_action.setCheckable(True)
        result_scale_log_action.triggered.connect(
            partial(self._render_result_image, True)
        )
        result_scaling_group.addAction(result_scale_log_action)
        self.scaling_menu.addAction(result_scale_log_action)

        result_scale_sqrt_action = QAction("Square Root", self)
        result_scale_sqrt_action.setCheckable(True)
        result_scale_sqrt_action.setChecked(True)
        result_scale_sqrt_action.triggered.connect(
            partial(self._render_result_image, True)
        )
        result_scaling_group.addAction(result_scale_sqrt_action)
        self.scaling_menu.addAction(result_scale_sqrt_action)

        # Autorange menu
        self.autorange_menu = QMenu("&Autorange", self)
        self.menu_bar.addMenu(self.autorange_menu)

        diff_autoscale_separator = QAction("Diffraction", self)
        diff_autoscale_separator.setDisabled(True)
        self.autorange_menu.addAction(diff_autoscale_separator)

        diff_range_group = QActionGroup(self)
        diff_range_group.setExclusive(True)

        scale_range_default = self.settings.value(
            "last_state/diffraction_autorange", [0.1, 99.9], type=float
        )
        for scale_range in [(0, 100), (0.1, 99.9), (1, 99), (2, 98), (5, 95)]:
            action = QAction(f"{scale_range[0]}% – {scale_range[1]}%", self)
            diff_range_group.addAction(action)
            self.autorange_menu.addAction(action)
            action.setCheckable(True)
            action.triggered.connect(
                partial(self.set_diffraction_autoscale_range, scale_range)
            )
            # set default
            if (
                scale_range[0] == scale_range_default[0]
                and scale_range[1] == scale_range_default[1]
            ):
                action.setChecked(True)
                self.set_diffraction_autoscale_range(scale_range, redraw=False)

        self.autorange_menu.addSeparator()

        vimg_autoscale_separator = QAction("Virtual Image", self)
        vimg_autoscale_separator.setDisabled(True)
        self.autorange_menu.addAction(vimg_autoscale_separator)

        vimg_range_group = QActionGroup(self)
        vimg_range_group.setExclusive(True)

        scale_range_default = self.settings.value(
            "last_state/realspace_autorange", [0.1, 99.9], type=float
        )
        for scale_range in [(0, 100), (0.1, 99.9), (1, 99), (2, 98), (5, 95)]:
            action = QAction(f"{scale_range[0]}% – {scale_range[1]}%", self)
            vimg_range_group.addAction(action)
            self.autorange_menu.addAction(action)
            action.setCheckable(True)
            action.triggered.connect(
                partial(self.set_real_space_autoscale_range, scale_range)
            )
            # set default
            if (
                scale_range[0] == scale_range_default[0]
                and scale_range[1] == scale_range_default[1]
            ):
                action.setChecked(True)
                self.set_real_space_autoscale_range(scale_range, redraw=False)

        ##
        self.autorange_menu.addSeparator()

        result_autoscale_separator = QAction("Result", self)
        result_autoscale_separator.setDisabled(True)
        self.autorange_menu.addAction(result_autoscale_separator)

        result_range_group = QActionGroup(self)
        result_range_group.setExclusive(True)

        scale_range_default = self.settings.value(
            "last_state/result_autorange", [0.1, 99.9], type=float
        )
        for scale_range in [(0, 100), (0.1, 99.9), (1, 99), (2, 98), (5, 95)]:
            action = QAction(f"{scale_range[0]}% – {scale_range[1]}%", self)
            result_range_group.addAction(action)
            self.autorange_menu.addAction(action)
            action.setCheckable(True)
            action.triggered.connect(
                partial(self.set_result_autoscale_range, scale_range)
            )
            # set default
            if (
                scale_range[0] == scale_range_default[0]
                and scale_range[1] == scale_range_default[1]
            ):
                action.setChecked(True)
                self.set_result_autoscale_range(scale_range, redraw=False)
        ##

        # Detector Response Menu
        self.detector_menu = QMenu("&Detector Response", self)
        self.menu_bar.addMenu(self.detector_menu)

        detector_mode_separator = QAction("Diffraction", self)
        detector_mode_separator.setDisabled(True)
        self.detector_menu.addAction(detector_mode_separator)

        detector_mode_group = QActionGroup(self)
        detector_mode_group.setExclusive(True)
        self.detector_mode_group = detector_mode_group

        detector_integrating_action = QAction("&Integrating", self)
        detector_integrating_action.setCheckable(True)
        detector_integrating_action.setChecked(True)
        detector_integrating_action.triggered.connect(
            partial(self.update_real_space_view, True)
        )
        detector_mode_group.addAction(detector_integrating_action)
        self.detector_menu.addAction(detector_integrating_action)

        detector_maximum_action = QAction("&Maximum", self)
        detector_maximum_action.setCheckable(True)
        detector_maximum_action.triggered.connect(
            partial(self.update_real_space_view, True)
        )
        detector_mode_group.addAction(detector_maximum_action)
        self.detector_menu.addAction(detector_maximum_action)

        detector_average_action = QAction("A&verage", self)
        detector_average_action.setCheckable(True)
        detector_average_action.triggered.connect(
            partial(self.update_real_space_view, True)
        )
        detector_mode_group.addAction(detector_average_action)
        self.detector_menu.addAction(detector_average_action)

        detector_CoM = QAction("C&oM", self)
        detector_CoM.setCheckable(True)
        detector_CoM.triggered.connect(partial(self.update_real_space_view, True))
        detector_mode_group.addAction(detector_CoM)
        self.detector_menu.addAction(detector_CoM)

        detector_CoMx = QAction("CoM &X", self)
        detector_CoMx.setCheckable(True)
        detector_CoMx.triggered.connect(partial(self.update_real_space_view, True))
        detector_mode_group.addAction(detector_CoMx)
        self.detector_menu.addAction(detector_CoMx)

        detector_CoMy = QAction("CoM &Y", self)
        detector_CoMy.setCheckable(True)
        detector_CoMy.triggered.connect(partial(self.update_real_space_view, True))
        detector_mode_group.addAction(detector_CoMy)
        self.detector_menu.addAction(detector_CoMy)

        detector_iCoM = QAction("i&CoM", self)
        detector_iCoM.setCheckable(True)
        detector_iCoM.triggered.connect(partial(self.update_real_space_view, True))
        detector_mode_group.addAction(detector_iCoM)
        self.detector_menu.addAction(detector_iCoM)

        # Detector Response for realspace selector
        self.detector_menu.addSeparator()
        rs_detector_mode_separator = QAction("Virtual Image", self)
        rs_detector_mode_separator.setDisabled(True)
        self.detector_menu.addAction(rs_detector_mode_separator)

        realspace_detector_mode_group = QActionGroup(self)
        realspace_detector_mode_group.setExclusive(True)
        self.realspace_detector_mode_group = realspace_detector_mode_group

        detector_integrating_action = QAction("&Integrating", self)
        detector_integrating_action.setCheckable(True)
        detector_integrating_action.setChecked(True)
        detector_integrating_action.triggered.connect(
            partial(self.update_diffraction_space_view, True)
        )
        realspace_detector_mode_group.addAction(detector_integrating_action)
        self.detector_menu.addAction(detector_integrating_action)

        detector_maximum_action = QAction("&Maximum", self)
        detector_maximum_action.setCheckable(True)
        detector_maximum_action.triggered.connect(
            partial(self.update_diffraction_space_view, True)
        )
        realspace_detector_mode_group.addAction(detector_maximum_action)
        self.detector_menu.addAction(detector_maximum_action)

        detector_average_action = QAction("A&verage", self)
        detector_average_action.setCheckable(True)
        detector_average_action.triggered.connect(
            partial(self.update_diffraction_space_view, True)
        )
        realspace_detector_mode_group.addAction(detector_average_action)
        self.detector_menu.addAction(detector_average_action)

        # Detector Shape Menu
        self.detector_shape_menu = QMenu("Detector &Shape", self)
        self.menu_bar.addMenu(self.detector_shape_menu)

        detector_shape_group = QActionGroup(self)
        detector_shape_group.setExclusive(True)
        self.detector_shape_group = detector_shape_group

        diffraction_detector_separator = QAction("Diffraction", self)
        diffraction_detector_separator.setDisabled(True)
        self.detector_shape_menu.addAction(diffraction_detector_separator)

        detector_point_action = QAction("&Point", self)
        detector_point_action.setCheckable(True)
        detector_point_action.setChecked(True)  # Default
        detector_point_action.triggered.connect(self.update_diffraction_detector)
        detector_shape_group.addAction(detector_point_action)
        self.detector_shape_menu.addAction(detector_point_action)

        detector_rectangle_action = QAction("&Rectangular", self)
        detector_rectangle_action.setCheckable(True)
        # detector_rectangle_action.setChecked(True)
        detector_rectangle_action.triggered.connect(self.update_diffraction_detector)
        detector_shape_group.addAction(detector_rectangle_action)
        self.detector_shape_menu.addAction(detector_rectangle_action)

        detector_circle_action = QAction("&Circle", self)
        detector_circle_action.setCheckable(True)
        detector_circle_action.triggered.connect(self.update_diffraction_detector)
        detector_shape_group.addAction(detector_circle_action)
        self.detector_shape_menu.addAction(detector_circle_action)

        detector_annulus_action = QAction("&Annulus", self)
        detector_annulus_action.setCheckable(True)
        detector_annulus_action.triggered.connect(self.update_diffraction_detector)
        detector_shape_group.addAction(detector_annulus_action)
        self.detector_shape_menu.addAction(detector_annulus_action)

        self.detector_shape_menu.addSeparator()

        diffraction_detector_separator = QAction("Virtual Image", self)
        diffraction_detector_separator.setDisabled(True)
        self.detector_shape_menu.addAction(diffraction_detector_separator)

        rs_detector_shape_group = QActionGroup(self)
        rs_detector_shape_group.setExclusive(True)
        self.rs_detector_shape_group = rs_detector_shape_group

        rs_detector_point_action = QAction("Poin&t", self)
        rs_detector_point_action.setCheckable(True)
        rs_detector_point_action.setChecked(True)  # Default
        rs_detector_point_action.triggered.connect(self.update_realspace_detector)
        rs_detector_shape_group.addAction(rs_detector_point_action)
        self.detector_shape_menu.addAction(rs_detector_point_action)

        detector_rectangle_action = QAction("Rectan&gular", self)
        detector_rectangle_action.setCheckable(True)
        detector_rectangle_action.triggered.connect(self.update_realspace_detector)
        rs_detector_shape_group.addAction(detector_rectangle_action)
        self.detector_shape_menu.addAction(detector_rectangle_action)

        self.result_menu = QMenu("Resul&t View", self)
        self.menu_bar.addMenu(self.result_menu)

        self.result_source_action_group = QActionGroup(self)
        self.result_source_action_group.setExclusive(True)
        img_fft_action = QAction("Virtual Image FFT", self)
        img_fft_action.setCheckable(True)
        img_fft_action.triggered.connect(self.set_internal_result_callback)
        img_fft_action.setChecked(True)
        self.result_menu.addAction(img_fft_action)
        self.result_source_action_group.addAction(img_fft_action)

        img_complex_fft_action = QAction("Virtual Image FFT (complex)", self)
        img_complex_fft_action.setCheckable(True)
        self.result_menu.addAction(img_complex_fft_action)
        self.result_source_action_group.addAction(img_complex_fft_action)
        img_complex_fft_action.triggered.connect(self.set_internal_result_callback)

        img_ewpc_action = QAction("EWPC", self)
        img_ewpc_action.setCheckable(True)
        self.result_menu.addAction(img_ewpc_action)
        self.result_source_action_group.addAction(img_ewpc_action)
        img_ewpc_action.triggered.connect(self.set_internal_result_callback)

        # action only for information purposes, to show when a plugin has taken over
        # the result window
        self.result_other_action = QAction("Plugin", self)
        self.result_other_action.setCheckable(True)
        self.result_other_action.setEnabled(False)
        self.result_menu.addAction(self.result_other_action)
        self.result_source_action_group.addAction(self.result_other_action)

        self.set_internal_result_callback()

        # Plugins menu
        self.processing_menu = QMenu("&Plugins", self)
        self.menu_bar.addMenu(self.processing_menu)

        # Help menu
        self.help_menu = QMenu("&Help", self)
        self.menu_bar.addMenu(self.help_menu)

        self.keyboard_map_action = QAction("Show &Keyboard Map", self)
        self.keyboard_map_action.triggered.connect(self.show_keyboard_map)
        self.help_menu.addAction(self.keyboard_map_action)

        self.show_config_file_action = QAction("Show Configuration File", self)
        self.show_config_file_action.triggered.connect(
            partial(show_in_file_manager, self.config_path)
        )
        self.help_menu.addAction(self.show_config_file_action)

        self.debug_console_action = QAction("&Debug Console", self)
        self.debug_console_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+D"))
        self.debug_console_action.triggered.connect(self._launch_debug_console)
        self.help_menu.addAction(self.debug_console_action)

        self.help_menu.addSeparator()

        self.version_action = QAction(f"py4DGUI v{__version__}", self)
        self.version_action.setEnabled(False)
        self.help_menu.addAction(self.version_action)

    def setup_views(self):
        # Set up the diffraction space window.
        self.diffraction_space_widget = pg.ImageView()
        self.diffraction_space_widget.setImage(np.zeros((128, 128)))

        cmap_name = self.settings.value("gui/diffraction_colormap", "inferno")
        cmap = try_get_cmap(cmap_name)
        if cmap is not None:
            self.diffraction_space_widget.setColorMap(cmap)

        self.diffraction_space_widget.setMouseTracking(True)

        # Create virtual detector ROI selector
        self.update_diffraction_detector()

        # Double-click to center the detector on a data point
        self.diffraction_space_widget.view.scene().sigMouseClicked.connect(
            self.on_diffraction_double_click
        )

        # Scalebar
        self.diffraction_scale_bar = ScaleBar(pixel_size=1, units="px", width=10)
        self.diffraction_scale_bar.setParentItem(
            self.diffraction_space_widget.getView()
        )
        self.diffraction_scale_bar.anchor((1, 1), (1, 1), offset=(-40, -40))

        # Name and return
        self.diffraction_space_widget.setWindowTitle("Diffraction Space")

        # Set up the real space window.
        self.real_space_widget = pg.ImageView()
        self.real_space_widget.setImage(np.zeros((256, 256)))

        cmap_name = self.settings.value("gui/realspace_colormap", "thermal")
        cmap = try_get_cmap(cmap_name)
        if cmap is not None:
            self.real_space_widget.setColorMap(cmap)

        # Add point selector connected to displayed diffraction pattern
        self.update_realspace_detector()

        # Double-click to center the selector on a data point
        self.real_space_widget.view.scene().sigMouseClicked.connect(
            self.on_real_space_double_click
        )

        # Scalebar, None by default
        self.real_space_scale_bar = ScaleBar(pixel_size=1, units="px", width=10)
        self.real_space_scale_bar.setParentItem(self.real_space_widget.getView())
        self.real_space_scale_bar.anchor((1, 1), (1, 1), offset=(-40, -40))

        # Name and return
        self.real_space_widget.setWindowTitle("Virtual Image")

        self.diffraction_space_widget.setAcceptDrops(True)
        self.real_space_widget.setAcceptDrops(True)
        self.diffraction_space_widget.dragEnterEvent = self.dragEnterEvent
        self.real_space_widget.dragEnterEvent = self.dragEnterEvent
        self.diffraction_space_widget.dropEvent = self.dropEvent
        self.real_space_widget.dropEvent = self.dropEvent

        # Set up the FFT window.
        self.fft_widget = pg.ImageView()
        self.fft_widget.setImage(np.zeros((256, 256)))

        cmap_name = self.settings.value("gui/fft_colormap", "yellowy")
        cmap = try_get_cmap(cmap_name)
        if cmap is not None:
            self.fft_widget.setColorMap(cmap)

        # FFT scale bar
        self.fft_scale_bar = ScaleBar(pixel_size=1, units="1/px", width=10)
        self.fft_scale_bar.setParentItem(self.fft_widget.getView())
        self.fft_scale_bar.anchor((1, 1), (1, 1), offset=(-40, -40))

        # Name and return
        self.fft_widget.setWindowTitle("FFT of Virtual Image")
        self.fft_widget_text = pg.TextItem("FFT", (200, 200, 200), None, (0, 1))
        self.fft_widget.addItem(self.fft_widget_text)

        self.fft_widget.setAcceptDrops(True)
        self.fft_widget.dragEnterEvent = self.dragEnterEvent
        self.fft_widget.dropEvent = self.dropEvent

        # 3D result view: renders a volume tab's 3D FFT as a genuine 3D volume
        # (see _render_result_volume). Stacked with the 2D fft_widget (page 1),
        # so the existing 2D→RGB path is untouched. Styled like the volume tabs.
        self.result_volume_view = gl.GLViewWidget()
        self.result_volume_view.addItem(gl.GLAxisItem())
        self.result_volume_view.setBackgroundColor("k")

        # Title strip above the 3D view (a QLabel can't overlay a QOpenGLWidget).
        result_volume_page = QWidget()
        result_volume_layout = QVBoxLayout(result_volume_page)
        result_volume_layout.setContentsMargins(0, 0, 0, 0)
        result_volume_layout.setSpacing(0)
        self._result_title_label = QLabel("Virtual Image FFT")
        self._result_title_label.setStyleSheet(
            "color: rgb(200, 200, 200); background-color: rgb(0, 0, 0);"
        )
        self._result_title_label.setAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self._result_title_label.setFixedHeight(20)
        result_volume_layout.addWidget(self._result_title_label)
        result_volume_layout.addWidget(self.result_volume_view, 1)

        self.result_pane = QWidget()
        self._result_stack = QStackedLayout(self.result_pane)
        self._result_stack.addWidget(self.fft_widget)  # page 0: 2D result
        self._result_stack.addWidget(result_volume_page)  # page 1: 3D volume FFT

        # Bidirectional view-direction lock: result 3D view <-> visible volume
        # tab (see ViewDirectionSync). Active only while the 3D page shows.
        self._view_sync = self.ViewDirectionSync(self.result_volume_view)

        layout = QHBoxLayout()
        layout.addWidget(self.diffraction_space_widget, 1)

        # add a resizeable layout for the vimg and FFT
        #
        # The virtual-image pane is a QTabWidget. Tab 0 is the built-in
        # (default) virtual image — the existing real_space_widget — which is
        # not closable and not part of the plugin tab list. Plugins add
        # additional tabs (create_virtual_image_tab) which the user can close.
        # The tab bar is hidden until a second tab exists (see _refresh_tab_bar).
        #
        # Plugin tabs are closable via the tab bar's *native* close buttons
        # (setTabsClosable). The style lays each close button out alongside the
        # tab text, so a long title elides to make room for the "X" rather than
        # running underneath a separately-placed button widget. The default tab
        # is kept unclosable by suppressing its close button in _refresh_tab_bar
        # (setTabButton(0, ..., None)), so it has no "X".
        self.virtual_image_tab_widget = QTabWidget()
        self.virtual_image_tab_widget.addTab(self.real_space_widget, "Virtual Image")
        self.virtual_image_tab_widget.setTabsClosable(True)
        self.virtual_image_tab_widget.tabBar().tabCloseRequested.connect(
            self._on_tab_close_requested
        )
        self.virtual_image_tab_widget.currentChanged.connect(
            self._on_visible_virtual_image_changed
        )
        self._refresh_tab_bar()

        # Cycle through the tabs with Tab + a modifier (wrapping around at
        # either end), like cycling tabs in a browser.
        #
        # Windows/Linux: the usual Ctrl+Tab / Ctrl+Shift+Tab (bound below).
        #
        # macOS: also bind the *Option* (Alt) modifier. On this keyboard the
        # Control/Command keys deliver the Tab press as a codepoint rather than
        # Qt.Key_Tab, so a Key_Tab-based QShortcut bound to them does not match;
        # Option+Tab is the combination the user presses for this and arrives as
        # a clean Key_Tab.
        self.virtual_image_tab_next_shortcut = QShortcut(
            QtGui.QKeySequence(QtCore.Qt.ControlModifier + QtCore.Qt.Key_Tab),
            self,
        )
        self.virtual_image_tab_next_shortcut.activated.connect(
            partial(self._switch_virtual_image_tab, 1)
        )
        self.virtual_image_tab_prev_shortcut = QShortcut(
            QtGui.QKeySequence(
                QtCore.Qt.ControlModifier + QtCore.Qt.ShiftModifier + QtCore.Qt.Key_Tab
            ),
            self,
        )
        self.virtual_image_tab_prev_shortcut.activated.connect(
            partial(self._switch_virtual_image_tab, -1)
        )

        if sys.platform.startswith("darwin"):
            # Option (Alt) + Tab: the modifier the user presses for this on
            # their keyboard; it arrives as a clean Key_Tab, so the
            # Key_Tab-based QShortcut matches it (unlike the Control/Command
            # keys, which deliver the press as a codepoint).
            self.virtual_image_tab_next_option_shortcut = QShortcut(
                QtGui.QKeySequence(QtCore.Qt.AltModifier + QtCore.Qt.Key_Tab),
                self,
            )
            self.virtual_image_tab_next_option_shortcut.activated.connect(
                partial(self._switch_virtual_image_tab, 1)
            )
            self.virtual_image_tab_prev_option_shortcut = QShortcut(
                QtGui.QKeySequence(
                    QtCore.Qt.AltModifier + QtCore.Qt.ShiftModifier + QtCore.Qt.Key_Tab
                ),
                self,
            )
            self.virtual_image_tab_prev_option_shortcut.activated.connect(
                partial(self._switch_virtual_image_tab, -1)
            )

        # Close the window with Ctrl+W (Command+W on a Mac), like any other
        # macOS window.
        self.close_shortcut = QShortcut(QtGui.QKeySequence("Ctrl+W"), self)
        self.close_shortcut.activated.connect(self.close)

        rightside = QSplitter()
        rightside.addWidget(self.virtual_image_tab_widget)
        # The result pane (2D + 3D views), not the bare fft_widget, so a 3D
        # FFT can swap to the 3D page.
        rightside.addWidget(self.result_pane)
        rightside.setOrientation(QtCore.Qt.Vertical)
        # set a sensible ratio for the sizes
        full_height = (
            self.real_space_widget.size().height() + self.result_pane.size().height()
        )
        rightside.setSizes([int(full_height * 2 / 3), int(full_height / 3)])

        layout.addWidget(rightside, 1)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.diffraction_space_widget.getView().setMenuEnabled(False)
        self.real_space_widget.getView().setMenuEnabled(False)
        self.fft_widget.getView().setMenuEnabled(False)

        # Setup Status Bar
        self.stats_button = QPushButton("Statistics")
        self.stats_menu = QMenu()

        self.realspace_title = QAction("Virtual Image")
        self.realspace_title.setDisabled(False)
        self.stats_menu.addAction(self.realspace_title)
        self.realspace_statistics_actions = [QAction("") for i in range(5)]
        for a in self.realspace_statistics_actions:
            self.stats_menu.addAction(a)

        self.stats_menu.addSeparator()

        self.diffraction_title = QAction("Diffraction")
        self.diffraction_title.setDisabled(False)
        self.stats_menu.addAction(self.diffraction_title)
        self.diffraction_statistics_actions = [QAction("") for i in range(5)]
        for a in self.diffraction_statistics_actions:
            self.stats_menu.addAction(a)

        self.stats_button.setMenu(self.stats_menu)

        # Keep the "Virtual Image" section of the statistics menu in sync with
        # whichever tab is visible. This fires on tab switches, when the
        # visible tab's data changes, and on built-in virtual-image updates
        # while the default tab is visible.
        self.signal_current_virtual_image_changed.connect(
            self._update_visible_virtual_image_statistics
        )

        self.cursor_value_text = QLabel("")
        self.diffraction_space_view_text = QLabel("Slice")
        self.real_space_view_text = QLabel("Scan Position")

        # self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.cursor_value_text)
        self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.stats_button)
        self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.diffraction_space_view_text)
        self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.real_space_view_text)
        self.statusBar().addPermanentWidget(VLine())

        self.diffraction_rescale_button = LatchingButton(
            "Autorange Diffraction",
            status_bar=self.statusBar(),
            latched=True,
        )
        # apply the configured autoscale (percentile) range on click, rather
        # than pyqtgraph's full-range autoLevels, so the user's chosen range is
        # respected on every press
        self.diffraction_rescale_button.activated.connect(
            partial(self._render_diffraction_image, reset=False, auto_level=True)
        )
        self.statusBar().addPermanentWidget(self.diffraction_rescale_button)

        self.realspace_rescale_button = LatchingButton(
            "Autorange Virtual Image",
            status_bar=self.statusBar(),
            latched=True,
        )
        self.realspace_rescale_button.activated.connect(
            partial(self._render_virtual_image, reset=False, auto_level=True)
        )
        self.statusBar().addPermanentWidget(self.realspace_rescale_button)

        self.result_rescale_button = LatchingButton(
            "Autorange Result",
            status_bar=self.statusBar(),
            latched=True,
        )
        self.result_rescale_button.activated.connect(
            partial(self._render_result_image, reset=False, auto_level=True)
        )
        self.statusBar().addPermanentWidget(self.result_rescale_button)

    ########## virtual-image tabs (plugin API v1.6) ##########

    def create_virtual_image_tab(
        self, title: str, select: bool = False
    ) -> VirtualImageTab:
        """
        Create and show a new virtual-image tab the plugin can drive.

        The returned :class:`~py4D_browser.virtual_image_tabs.VirtualImageTab`
        is added to the virtual-image pane (behind the default, built-in tab)
        and can be driven with ``set_image``, ``set_scalebar``, and
        ``add_roi`` / ``add_annotation``. By default it is **not**
        auto-selected: the built-in virtual image stays visible until the user
        clicks the new tab. Pass ``select=True`` to immediately switch the
        pane to the new tab. Closing the tab (by the user, or via ``close``)
        cleanly detaches any ROIs and annotations and frees the widget.

        The new tab is seeded with the browser's default virtual-image colormap
        (the ``gui/realspace_colormap`` setting, the same default the built-in
        pane starts with), so a fresh tab matches the browser's default look.
        Each tab still keeps its own colormap from then on (see
        :attr:`VirtualImageTab.colormap`).

        A new tab also starts by inheriting the built-in pane's **current**
        display settings — its scaling mode, the Autorange latch, and the
        autoscale percentiles (read once at creation via
        :meth:`VirtualImageTab._builtin_settings`) — so a fresh tab matches
        whatever the user is currently looking at. From then on each tab is
        independent: it keeps its own scaling and range (see
        :attr:`VirtualImageTab.scaling`, :attr:`VirtualImageTab.autorange`, and
        :attr:`VirtualImageTab.autorange_percentiles`), and changing one tab
        never affects the built-in pane or any other tab.
        """
        tab = VirtualImageTab(title, self)

        # Give the tab the browser's default virtual-image colormap.
        self._seed_tab_colormap(tab)

        # With setTabsClosable(True) active, this new tab automatically gets the
        # tab bar's native close button; the default tab's is suppressed by the
        # _refresh_tab_bar() call below.
        self.virtual_image_tab_widget.addTab(tab.widget, title)
        self.virtual_image_tabs.append(tab)
        if select:
            # The new tab was just appended, so it is the last index. Selecting
            # it fires currentChanged, which notifies listeners that the
            # currently-visible virtual image has changed.
            self.virtual_image_tab_widget.setCurrentIndex(
                self.virtual_image_tab_widget.count() - 1
            )
        self._refresh_tab_bar()
        return tab

    def create_volume_tab(self, title: str, select: bool = False) -> VolumeTab:
        """
        Create and show a new 3D volume tab the plugin can drive.

        The returned :class:`~py4D_browser.volume_tabs.VolumeTab` is added to
        the virtual-image pane (behind the default, built-in tab) and renders
        a 3D volume via pyqtgraph's OpenGL volume rendering: the user can
        orbit/pan/zoom with the mouse and adjust the tab's color and alpha
        transfer functions live. Drive it with ``set_volume`` /
        ``set_levels``. By default it is **not** auto-selected: the built-in
        virtual image stays visible until the user clicks the new tab. Pass
        ``select=True`` to immediately switch the pane to the new tab.
        Closing the tab (by the user, or via ``close``) frees its widget and
        GL resources.

        The tab's color transfer function is seeded with the browser's
        default virtual-image colormap (the ``gui/realspace_colormap``
        setting), matching the built-in pane's default look.

        Requires PyOpenGL (a default dependency); constructing the tab raises
        ``RuntimeError`` if it is not installed.
        """
        tab = VolumeTab(title, self)

        # With setTabsClosable(True) active, this new tab automatically gets the
        # tab bar's native close button; the default tab's is suppressed by the
        # _refresh_tab_bar() call below.
        self.virtual_image_tab_widget.addTab(tab.widget, title)
        self.virtual_image_tabs.append(tab)
        if select:
            # The new tab was just appended, so it is the last index. Selecting
            # it fires currentChanged, which notifies listeners that the
            # currently-visible virtual image has changed.
            self.virtual_image_tab_widget.setCurrentIndex(
                self.virtual_image_tab_widget.count() - 1
            )
        self._refresh_tab_bar()
        return tab

    def _seed_tab_colormap(self, tab: VirtualImageTab):
        """
        Seed a new tab's colormap with the browser's default virtual-image
        colormap (the ``gui/realspace_colormap`` setting, default ``"thermal"``),
        so a fresh tab matches the built-in pane's default appearance.

        This mirrors how the built-in real-space pane is given its default
        colormap in ``setup_views``: read the setting, resolve it through the
        browser's colormap lookup, and apply it if it resolves. If the setting
        names a colormap the lookup cannot resolve, fall back to ``"thermal"``
        so a fresh tab never starts on pyqtgraph's default greyscale. Each tab
        keeps its own colormap from then on (see
        :attr:`~py4D_browser.virtual_image_tabs.VirtualImageTab.colormap`), so
        a user who changes one tab's colormap does not move the others.
        """
        cmap_name = self.settings.value("gui/realspace_colormap", "thermal")
        cmap = try_get_cmap(cmap_name)
        if cmap is None:
            cmap = try_get_cmap("thermal")
        if cmap is not None:
            tab.colormap = cmap

    def close_virtual_image_tab(self, tab):
        """
        Close a virtual-image tab (an image or volume tab), detaching any
        ROIs/annotations and freeing its widget. Idempotent: a no-op if the
        tab is already closed.
        """
        if tab is None or tab.closed:
            return

        # Detach any ROIs/annotations the plugin attached so they are freed
        # (or remain reusable by the plugin) independently of the widget we
        # are about to delete. The tab owns this bookkeeping; the items
        # themselves are not deleted.
        tab.detach_items()

        # Remove the tab from the pane and free its widget. The tab bar's native
        # close button for this tab goes away with the tab (no separate widget to
        # free). Removing the currently-visible tab fires currentChanged, which
        # (below) notifies listeners that the visible virtual image changed.
        widx = self.virtual_image_tab_widget.indexOf(tab.widget)
        if widx != -1:
            self.virtual_image_tab_widget.removeTab(widx)
        tab.widget.deleteLater()

        if tab in self.virtual_image_tabs:
            self.virtual_image_tabs.remove(tab)

        tab._closed = True
        self._refresh_tab_bar()
        gc.collect()

    def _on_tab_close_requested(self, index: int):
        # The native close button fires this with the tab's index. Tab 0 is the
        # built-in default tab and has no close button; guard against it anyway.
        # (The index is resolved at click time, so it always refers to the tab
        # whose button was pressed, regardless of other tabs closing first.)
        if index <= 0:
            return
        self.close_virtual_image_tab(self.virtual_image_tabs[index - 1])

    def _tab_close_button_side(self):
        """
        The :class:`QTabBar` side the native close button is drawn on, per the
        active style (RightSide on most platforms, LeftSide on macOS). Used to
        suppress the default tab's close button wherever the style actually
        puts it, so it is hidden on either side.
        """
        tab_widget = self.virtual_image_tab_widget
        tab_bar = tab_widget.tabBar()
        side = tab_widget.style().styleHint(
            QStyle.SH_TabBar_CloseButtonPosition, None, tab_bar
        )
        # SH_TabBar_CloseButtonPosition is QTabBar.RightSide (0) or
        # QTabBar.LeftSide (1).
        return QTabBar.LeftSide if side == QTabBar.LeftSide else QTabBar.RightSide

    def _refresh_tab_bar(self):
        """
        Keep the tab bar's visibility and close-buttons in the desired state:
        hidden when only the default tab exists, and never showing a close
        button on the (unclosable) default tab.
        """
        tab_widget = self.virtual_image_tab_widget
        tab_bar = tab_widget.tabBar()
        tab_bar.setVisible(tab_widget.count() > 1)
        tab_bar.setTabButton(0, self._tab_close_button_side(), None)

    def _switch_virtual_image_tab(self, direction: int):
        """
        Switch to the next (``direction=1``) or previous (``direction=-1``)
        virtual-image tab, wrapping around at either end. A no-op while only
        the default tab exists (the tab bar is hidden in that case), so the
        shortcut never fires on a pane with a single tab.
        """
        tab_widget = self.virtual_image_tab_widget
        count = tab_widget.count()
        if count < 2:
            return
        tab_widget.setCurrentIndex((tab_widget.currentIndex() + direction) % count)

    def _on_visible_virtual_image_changed(self, _index: int):
        # The visible tab changed (user switch, or a visible tab was closed).
        self.signal_current_virtual_image_changed.emit()

    def _update_visible_virtual_image_statistics(self):
        """
        Update the "Virtual Image" section of the statistics menu (the
        ``realspace_title`` and the five ``realspace_statistics_actions``)
        to show the name and summary stats of whichever virtual-image tab is
        currently visible — the built-in virtual image or a plugin's
        image/volume tab.

        Connected to ``signal_current_virtual_image_changed``, so this runs
        on tab switches, when the visible tab's data changes, and on built-in
        virtual-image updates while the default tab is visible. The stats are
        computed from the same raw array ``current_virtual_image`` exposes
        (2D image or 3D volume).
        """
        index = (
            self.virtual_image_tab_widget.currentIndex()
            if getattr(self, "virtual_image_tab_widget", None) is not None
            else 0
        )
        if index <= 0:
            name = "Virtual Image"
        else:
            tabs = self.virtual_image_tabs
            name = tabs[index - 1].title if index - 1 < len(tabs) else "Virtual Image"

        image = self.current_virtual_image
        if image is None:
            stats_text = [""] * 5
        else:
            stats_text = [
                f"Min:\t{image.min():.5g}",
                f"Max:\t{image.max():.5g}",
                f"Mean:\t{image.mean():.5g}",
                f"Sum:\t{image.sum():.5g}",
                f"Std:\t{np.std(image):.5g}",
            ]

        self.realspace_title.setText(name)
        for t, m in zip(stats_text, self.realspace_statistics_actions):
            m.setText(t)

    @property
    def current_virtual_image(self):
        """
        The raw array currently shown in the virtual-image pane: the built-in
        image when the default tab is visible, else the visible plugin tab's
        last-set image — a 2D image from an image tab, or a 3D volume from a
        volume tab.
        """
        # ``virtual_image_tab_widget`` doesn't exist until setup_views();
        # during early init (menus are set up first) treat it as the default
        # tab being visible.
        index = (
            self.virtual_image_tab_widget.currentIndex()
            if getattr(self, "virtual_image_tab_widget", None) is not None
            else 0
        )
        if index <= 0:
            return self.unscaled_realspace_image
        tabs = self.virtual_image_tabs
        if index - 1 < len(tabs):
            return tabs[index - 1].image
        return self.unscaled_realspace_image

    @property
    def current_virtual_image_settings(self):
        """
        The display settings of the virtual-image view currently on screen,
        as a :class:`~py4D_browser.utils.VirtualImageSettings` record: the
        built-in pane's scaling / Autorange / autoscale percentiles / levels
        when the default tab is visible, else the visible image tab's.

        Returns ``None`` when the visible view has no 2D display settings —
        an empty built-in pane or image tab, or a volume tab (which renders a
        3D volume rather than a scaled image).
        """
        # ``virtual_image_tab_widget`` doesn't exist until setup_views();
        # during early init (menus are set up first) treat it as the default
        # tab being visible.
        index = (
            self.virtual_image_tab_widget.currentIndex()
            if getattr(self, "virtual_image_tab_widget", None) is not None
            else 0
        )
        if index <= 0:
            if self.unscaled_realspace_image is None:
                return None
            return VirtualImageSettings(
                scaling=self.vimg_scaling_group.checkedAction().text().replace("&", ""),
                autorange=self.realspace_rescale_button.latched,
                percentiles=tuple(self.real_space_autoscale_percentiles),
                levels=tuple(self.real_space_widget.getHistogramWidget().getLevels()),
            )
        tabs = self.virtual_image_tabs
        if index - 1 >= len(tabs):
            return None
        tab = tabs[index - 1]
        # Volume tabs have no image_view and no 2D display settings.
        if getattr(tab, "image_view", None) is None or tab.image is None:
            return None
        return VirtualImageSettings(
            scaling=tab.scaling,
            autorange=tab.autorange,
            percentiles=tuple(tab.autorange_percentiles),
            levels=tab.levels,
        )

    @property
    def current_diffraction_image_settings(self):
        """
        The display settings of the built-in diffraction (reciprocal-space)
        pane, as a :class:`~py4D_browser.utils.VirtualImageSettings` record:
        its scaling / Autorange / autoscale percentiles / levels.

        Returns ``None`` when the pane is empty (no diffraction image loaded).
        Use this to read the DP pane's current scaling and range (e.g. to
        mirror them onto a new tab copied from the DP).
        """
        if self.unscaled_diffraction_image is None:
            return None
        return VirtualImageSettings(
            scaling=self.diff_scaling_group.checkedAction().text().replace("&", ""),
            autorange=self.diffraction_rescale_button.latched,
            percentiles=tuple(self.diffraction_autoscale_percentiles),
            levels=tuple(
                self.diffraction_space_widget.getHistogramWidget().getLevels()
            ),
        )

    @property
    def _visible_real_space_widget(self):
        """
        The ``pg.ImageView`` currently shown in the virtual-image pane: the
        built-in real-space widget when the default tab is visible, else the
        visible plugin tab's render view.

        Image tabs expose their render view as ``image_view`` (their ``widget``
        is now the page container holding the view + the bottom control bar);
        volume tabs have no ``image_view`` and fall back to their container
        ``widget``, which the ImageView consumers reject as they did before.
        """
        index = (
            self.virtual_image_tab_widget.currentIndex()
            if getattr(self, "virtual_image_tab_widget", None) is not None
            else 0
        )
        if index <= 0:
            return self.real_space_widget
        tabs = self.virtual_image_tabs
        if index - 1 < len(tabs):
            return getattr(tabs[index - 1], "image_view", tabs[index - 1].widget)
        return self.real_space_widget

    def is_virtual_image_tab_visible(self, tab) -> bool:
        """True if ``tab`` is the currently-visible virtual-image tab (an image or volume tab)."""
        index = (
            self.virtual_image_tab_widget.currentIndex()
            if getattr(self, "virtual_image_tab_widget", None) is not None
            else 0
        )
        if index <= 0:
            return False
        tabs = self.virtual_image_tabs
        if index - 1 < len(tabs):
            return tabs[index - 1] is tab
        return False

    def _update_tab_title(self, tab):
        """Update the pane's tab label for ``tab`` to its current title."""
        for i in range(self.virtual_image_tab_widget.count()):
            if self.virtual_image_tab_widget.widget(i) is tab.widget:
                self.virtual_image_tab_widget.setTabText(i, tab.title)
                break
        if self.is_virtual_image_tab_visible(tab):
            # The statistics menu shows the visible tab's name, so refresh
            # that too when a visible tab is renamed.
            self._update_visible_virtual_image_statistics()

    def resizeEvent(self, event):
        # Store window size for next run
        self.settings.setValue("last_state/window_size", event.size())

    def _launch_debug_console(self):
        pg.dbg(namespace={"main_window": self})

    def closeEvent(self, event):
        self.unload_plugins()
        event.accept()

    # Handle dragging and dropping a file on the window
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if len(files) == 1:
            print(f"Reieving dropped file: {files[0]}")
            self.load_file(files[0])

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        # The nudge bindings are bare letter keys: ignore them while any
        # modifier other than Shift is held, so e.g. Cmd+W (close the window)
        # and Ctrl+C (copy) are not swallowed as a detector nudge.
        if modifier & ~QtCore.Qt.ShiftModifier:
            return

        speed = 5 if modifier == QtCore.Qt.ShiftModifier else 1

        if key in [QtCore.Qt.Key_W, QtCore.Qt.Key_A, QtCore.Qt.Key_S, QtCore.Qt.Key_D]:
            self.nudge_diffraction_selector(
                dx=speed
                * (
                    -1 if key == QtCore.Qt.Key_W else 1 if key == QtCore.Qt.Key_S else 0
                ),
                dy=speed
                * (
                    -1 if key == QtCore.Qt.Key_A else 1 if key == QtCore.Qt.Key_D else 0
                ),
            )
        elif key in [
            QtCore.Qt.Key_I,
            QtCore.Qt.Key_J,
            QtCore.Qt.Key_K,
            QtCore.Qt.Key_L,
        ]:
            self.nudge_real_space_selector(
                dx=speed
                * (
                    -1 if key == QtCore.Qt.Key_I else 1 if key == QtCore.Qt.Key_K else 0
                ),
                dy=speed
                * (
                    -1 if key == QtCore.Qt.Key_J else 1 if key == QtCore.Qt.Key_L else 0
                ),
            )
