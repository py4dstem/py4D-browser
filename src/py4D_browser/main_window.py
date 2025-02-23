from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QMenu,
    QAction,
    QHBoxLayout,
    QSplitter,
    QActionGroup,
    QLabel,
    QToolTip,
    QPushButton,
    QShortcut,
)

from matplotlib.backend_bases import tools
import pyqtgraph as pg
import numpy as np

from functools import partial
from pathlib import Path
import importlib
import os, sys
import platformdirs

from py4D_browser.utils import pg_point_roi, VLine, LatchingButton
from py4D_browser.scalebar import ScaleBar


class DataViewer(QMainWindow):
    """
    The class is used by instantiating and then entering the main Qt loop with, e.g.:
        app = DataViewer(sys.argv)
        app.exec_()
    """

    LOG_SCALE_MIN_VALUE = 1e-6

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
        update_scalebars,
    )

    from py4D_browser.update_views import (
        set_virtual_image,
        set_diffraction_image,
        get_diffraction_detector,
        get_virtual_image_detector,
        _render_virtual_image,
        _render_diffraction_image,
        update_diffraction_space_view,
        update_real_space_view,
        update_realspace_detector,
        update_diffraction_detector,
        set_diffraction_autoscale_range,
        set_real_space_autoscale_range,
        nudge_real_space_selector,
        nudge_diffraction_selector,
        update_annulus_pos,
        update_annulus_radii,
        update_tooltip,
    )

    from py4D_browser.plugins import load_plugins

    def __init__(self, argv):
        super().__init__()
        # Define this as the QApplication object
        self.qtapp = QApplication.instance()
        if not self.qtapp:
            self.qtapp = QApplication(argv)

        self.setWindowTitle("py4DSTEM")

        icon = QtGui.QIcon(str(Path(__file__).parent.absolute() / "logo.png"))
        self.setWindowIcon(icon)
        self.qtapp.setWindowIcon(icon)

        self.setWindowTitle("py4DSTEM")
        self.setAcceptDrops(True)

        self.datacube = None

        # Load settings from cofig file
        config_path = os.path.join(
            platformdirs.user_config_dir("py4DGUI", "py4DSTEM"), "GUI_config.ini"
        )
        print(f"Loading configuration from {config_path}")
        QtCore.QCoreApplication.setOrganizationName("py4DSTEM")
        QtCore.QCoreApplication.setOrganizationDomain("py4DSTEM.com")
        QtCore.QCoreApplication.setApplicationName("py4DGUI")
        self.settings = QtCore.QSettings(config_path, QtCore.QSettings.Format.IniFormat)

        # Reset stored state if so asked:
        if os.environ.get("PY4DGUI_RESET"):
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
        if len(argv) > 1:
            self.load_file(argv[1])

        # launch pyqtgraph's debug console if environment variable exists
        if os.environ.get("PY4DGUI_DEBUG"):
            pg.dbg()

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
        for method in ["PNG (display)", "TIFF (display)", "TIFF (raw)"]:
            menu_item = vimg_export_menu.addAction(method)
            menu_item.triggered.connect(
                partial(self.export_virtual_image, method, "image")
            )

        # Submenu to export diffraction
        vdiff_export_menu = QMenu("Export Diffraction Pattern", self)
        self.file_menu.addMenu(vdiff_export_menu)
        for method in ["PNG (display)", "TIFF (display)", "TIFF (raw)"]:
            menu_item = vdiff_export_menu.addAction(method)
            menu_item.triggered.connect(
                partial(self.export_virtual_image, method, "diffraction")
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

        self.fft_menu = QMenu("FF&T View", self)
        self.menu_bar.addMenu(self.fft_menu)

        self.fft_source_action_group = QActionGroup(self)
        self.fft_source_action_group.setExclusive(True)
        img_fft_action = QAction("Virtual Image FFT", self)
        img_fft_action.setCheckable(True)
        img_fft_action.setChecked(True)
        img_fft_action.triggered.connect(partial(self.update_real_space_view, False))
        self.fft_menu.addAction(img_fft_action)
        self.fft_source_action_group.addAction(img_fft_action)

        img_complex_fft_action = QAction("Virtual Image FFT (complex)", self)
        img_complex_fft_action.setCheckable(True)
        self.fft_menu.addAction(img_complex_fft_action)
        self.fft_source_action_group.addAction(img_complex_fft_action)
        img_complex_fft_action.triggered.connect(
            partial(self.update_real_space_view, False)
        )

        img_ewpc_action = QAction("EWPC", self)
        img_ewpc_action.setCheckable(True)
        self.fft_menu.addAction(img_ewpc_action)
        self.fft_source_action_group.addAction(img_ewpc_action)
        img_ewpc_action.triggered.connect(
            partial(self.update_diffraction_space_view, False)
        )

        # Plugins menu
        self.processing_menu = QMenu("&Plugins", self)
        self.menu_bar.addMenu(self.processing_menu)

        # Help menu
        self.help_menu = QMenu("&Help", self)
        self.menu_bar.addMenu(self.help_menu)

        self.keyboard_map_action = QAction("Show &Keyboard Map", self)
        self.keyboard_map_action.triggered.connect(self.show_keyboard_map)
        self.help_menu.addAction(self.keyboard_map_action)

    def setup_views(self):
        # Set up the diffraction space window.
        self.diffraction_space_widget = pg.ImageView()
        self.diffraction_space_widget.setImage(np.zeros((512, 512)))

        self.diffraction_space_widget.setMouseTracking(True)

        # Create virtual detector ROI selector
        self.update_diffraction_detector()

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
        self.real_space_widget.setImage(np.zeros((512, 512)))

        # Add point selector connected to displayed diffraction pattern
        self.update_realspace_detector()

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
        self.fft_widget.setImage(np.zeros((512, 512)))

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

        layout = QHBoxLayout()
        layout.addWidget(self.diffraction_space_widget, 1)

        # add a resizeable layout for the vimg and FFT
        rightside = QSplitter()
        rightside.addWidget(self.real_space_widget)
        rightside.addWidget(self.fft_widget)
        rightside.setOrientation(QtCore.Qt.Vertical)
        # set a sensible ratio for the sizes
        full_height = (
            self.real_space_widget.size().height() + self.fft_widget.size().height()
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
        self.diffraction_rescale_button.activated.connect(
            self.diffraction_space_widget.autoLevels
        )
        self.statusBar().addPermanentWidget(self.diffraction_rescale_button)
        self.realspace_rescale_button = LatchingButton(
            "Autorange Virtual Image",
            status_bar=self.statusBar(),
            latched=True,
        )
        self.realspace_rescale_button.activated.connect(
            self.real_space_widget.autoLevels
        )
        self.statusBar().addPermanentWidget(self.realspace_rescale_button)

    def resizeEvent(self, event):
        # Store window size for next run
        self.settings.setValue("last_state/window_size", event.size())

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
