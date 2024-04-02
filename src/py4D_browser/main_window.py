from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QMenu,
    QAction,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QActionGroup,
    QLabel,
    QPushButton,
    QTabWidget, 
    QDoubleSpinBox,
    QSpinBox, 
    QComboBox,
    QCheckBox,
    QGraphicsItemGroup,
)

import pyqtgraph as pg
import numpy as np

from functools import partial
from pathlib import Path
import importlib

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
    )

    from py4D_browser.update_views import (
        update_diffraction_space_view,
        update_real_space_view,
        update_realspace_detector,
        update_diffraction_detector,
        nudge_real_space_selector,
        nudge_diffraction_selector,
        update_annulus_pos,
        update_annulus_radii,
        update_probe_template_view,
        update_kernel_view, 
        update_disk_detection,
    )

    HAS_EMPAD2 = importlib.util.find_spec("empad2") is not None
    if HAS_EMPAD2:
        from py4D_browser.empad2_reader import (
            set_empad2_sensor,
            load_empad2_background,
            load_empad2_dataset,
        )

    def __init__(self, argv):
        super().__init__()
        # Define this as the QApplication object
        self.qtapp = QApplication.instance()
        if not self.qtapp:
            self.qtapp = QApplication(argv)

        self.setWindowTitle("py4DSTEM")
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        icon = QtGui.QIcon(str(Path(__file__).parent.absolute() / "logo.png"))
        self.setWindowIcon(icon)
        self.qtapp.setWindowIcon(icon)

        self.setWindowTitle("py4DSTEM")
        self.setAcceptDrops(True)

        self.datacube = None
        self.separate_window = None
        self.disk_group = QGraphicsItemGroup()

        self.setup_menus()
        self.setup_views()

        self.resize(1000, 800)

        self.show()

        # If a file was passed on the command line, open it
        if len(argv) > 1:
            self.load_file(argv[1])

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

        # EMPAD2 menu
        if self.HAS_EMPAD2:
            self.empad2_calibrations = None
            self.empad2_background = None

            self.empad2_menu = QMenu("&EMPAD-G2", self)
            self.menu_bar.addMenu(self.empad2_menu)

            sensor_menu = self.empad2_menu.addMenu("&Sensor")
            calibration_action_group = QActionGroup(self)
            calibration_action_group.setExclusive(True)
            from empad2 import SENSORS

            for name, sensor in SENSORS.items():
                menu_item = sensor_menu.addAction(sensor["display-name"])
                calibration_action_group.addAction(menu_item)
                menu_item.setCheckable(True)
                menu_item.triggered.connect(partial(self.set_empad2_sensor, name))

            self.empad2_menu.addAction("Load &Background...").triggered.connect(
                self.load_empad2_background
            )
            self.empad2_menu.addAction("Load &Dataset...").triggered.connect(
                self.load_empad2_dataset
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
            partial(self.update_diffraction_space_view, True)
        )
        diff_scaling_group.addAction(diff_scale_linear_action)
        self.scaling_menu.addAction(diff_scale_linear_action)

        diff_scale_log_action = QAction("Log", self)
        diff_scale_log_action.setCheckable(True)
        diff_scale_log_action.triggered.connect(
            partial(self.update_diffraction_space_view, True)
        )
        diff_scaling_group.addAction(diff_scale_log_action)
        self.scaling_menu.addAction(diff_scale_log_action)

        diff_scale_sqrt_action = QAction("Square Root", self)
        diff_scale_sqrt_action.setCheckable(True)
        diff_scale_sqrt_action.triggered.connect(
            partial(self.update_diffraction_space_view, True)
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
            partial(self.update_real_space_view, True)
        )
        vimg_scaling_group.addAction(vimg_scale_linear_action)
        self.scaling_menu.addAction(vimg_scale_linear_action)

        vimg_scale_log_action = QAction("Log", self)
        vimg_scale_log_action.setCheckable(True)
        vimg_scale_log_action.triggered.connect(
            partial(self.update_real_space_view, True)
        )
        vimg_scaling_group.addAction(vimg_scale_log_action)
        self.scaling_menu.addAction(vimg_scale_log_action)

        vimg_scale_sqrt_action = QAction("Square Root", self)
        vimg_scale_sqrt_action.setCheckable(True)
        vimg_scale_sqrt_action.triggered.connect(
            partial(self.update_real_space_view, True)
        )
        vimg_scaling_group.addAction(vimg_scale_sqrt_action)
        self.scaling_menu.addAction(vimg_scale_sqrt_action)

        # Detector Response Menu
        self.detector_menu = QMenu("&Detector Response", self)
        self.menu_bar.addMenu(self.detector_menu)

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

        detector_CoM_magnitude = QAction("CoM Ma&gnitude", self)
        detector_CoM_magnitude.setCheckable(True)
        detector_CoM_magnitude.triggered.connect(
            partial(self.update_real_space_view, True)
        )
        detector_mode_group.addAction(detector_CoM_magnitude)
        self.detector_menu.addAction(detector_CoM_magnitude)

        detector_CoM_angle = QAction("CoM &Angle", self)
        detector_CoM_angle.setCheckable(True)
        detector_CoM_angle.triggered.connect(partial(self.update_real_space_view, True))
        detector_mode_group.addAction(detector_CoM_angle)
        self.detector_menu.addAction(detector_CoM_angle)

        detector_iCoM = QAction("i&CoM", self)
        detector_iCoM.setCheckable(True)
        detector_iCoM.triggered.connect(partial(self.update_real_space_view, True))
        detector_mode_group.addAction(detector_iCoM)
        self.detector_menu.addAction(detector_iCoM)

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

        diffraction_detector_separator = QAction("Real Space", self)
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
        
        self.help_menu = QMenu("&Help", self)
        self.menu_bar.addMenu(self.help_menu)

        self.keyboard_map_action = QAction("Show &Keyboard Map", self)
        self.keyboard_map_action.triggered.connect(self.show_keyboard_map)
        self.help_menu.addAction(self.keyboard_map_action)

    def setup_views(self):
        # Set up the diffraction space window.
        self.diffraction_space_widget = pg.ImageView()
        self.diffraction_space_widget.setImage(np.zeros((512, 512)))
        self.diffraction_space_view_text = QLabel("Slice")

        # Create virtual detector ROI selector
        self.virtual_detector_point = pg_point_roi(
            self.diffraction_space_widget.getView()
        )
        self.virtual_detector_point.sigRegionChanged.connect(
            partial(self.update_real_space_view, False)
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
        self.real_space_widget.setImage(np.zeros((512, 512)))
        self.real_space_view_text = QLabel("Scan Position")

        # Add point selector connected to displayed diffraction pattern
        self.real_space_point_selector = pg_point_roi(self.real_space_widget.getView())
        self.real_space_point_selector.sigRegionChanged.connect(
            partial(self.update_diffraction_space_view, False)
        )

        # Scalebar, None by default
        self.real_space_scale_bar = ScaleBar(pixel_size=1, units="px", width=10)
        self.real_space_scale_bar.setParentItem(self.real_space_widget.getView())
        self.real_space_scale_bar.anchor((1, 1), (1, 1), offset=(-40, -40))

        # Name and return
        self.real_space_widget.setWindowTitle("Real Space")

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

        # Create a QTabWidget
        self.tab_widget = QTabWidget()

        # Add the ImageView and QLabel to the first tab
        self.tab1 = QWidget()
        
        self.tab1_layout = QHBoxLayout(self.tab1)
        self.tab1_layout.addWidget(self.diffraction_space_widget, 1)

        # add a resizeable layout for the vimg and FFT
        rightside = QSplitter()
        rightside.addWidget(self.real_space_widget)
        
        self.tabs_bottomright = QTabWidget()
        
        self.fft_tab = QWidget()
        self.fft_tab_layout = QVBoxLayout(self.fft_tab)
        self.fft_tab_layout.addWidget(self.fft_widget)
        
        self.probe_view = pg.ImageView()
        self.probe_view.setImage(np.zeros((512, 512)))
        
        self.disk_detect_tab = QWidget()
        self.disk_detect_tab_layout = QVBoxLayout(self.disk_detect_tab)
        
        self.cross_correlation_layout = QHBoxLayout()
        
        self.probe_template_layout = QVBoxLayout()
        self.generate_probe_template_button = QPushButton("Generate probe template...")
        self.generate_probe_template_button.clicked.connect(self.update_probe_template_view)
        self.probe_template_layout.addWidget(self.generate_probe_template_button)
        self.probe_view = pg.ImageView()
        self.probe_view.setImage(np.zeros((512, 512)))
        self.probe_template_layout.addWidget(self.probe_view)
        
        self.kernel_layout = QVBoxLayout()
        self.generate_kernel_button = QPushButton("Generate kernel...")
        self.generate_kernel_button.clicked.connect(self.update_kernel_view)
        self.kernel_layout.addWidget(self.generate_kernel_button)
        self.kernel_view = pg.ImageView()
        self.kernel_view.setImage(np.zeros((512, 512)))
        self.kernel_layout.addWidget(self.kernel_view)
        self.kernel_radius = QDoubleSpinBox()
        self.kernel_radius.setPrefix("Kernel radius multiplier: ")
        self.kernel_radius.setMinimum(2.0)
        self.kernel_radius.setMaximum(10.0)
        self.kernel_radius.valueChanged.connect(self.update_kernel_view)
        self.kernel_layout.addWidget(self.kernel_radius)
        
        self.cross_correlation_layout.addLayout(self.probe_template_layout)
        self.cross_correlation_layout.addLayout(self.kernel_layout)
        
        self.open_window_button = QPushButton("Setup disk detection parameters")
        self.open_window_button.clicked.connect(self.open_separate_window)
                
        self.disk_detect_tab_layout.addLayout(self.cross_correlation_layout)
        self.disk_detect_tab_layout.addWidget(self.open_window_button)
        
                
        self.tabs_bottomright.addTab(self.fft_tab, "FFT")
        self.tabs_bottomright.addTab(self.disk_detect_tab, "Disk Detection")
        
        rightside.addWidget(self.tabs_bottomright)
        rightside.setOrientation(QtCore.Qt.Vertical)
        rightside.setStretchFactor(0, 2)
        
        self.tab1_layout.addWidget(rightside, 1)
        
        self.diffraction_space_widget.getView().setMenuEnabled(False)
        self.real_space_widget.getView().setMenuEnabled(False)
        self.fft_widget.getView().setMenuEnabled(False)

        # Setup Status Bar
        self.realspace_statistics_text = QLabel("Image Stats")
        self.diffraction_statistics_text = QLabel("Diffraction Stats")
        self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.realspace_statistics_text)
        self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.diffraction_statistics_text)
        self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.diffraction_space_view_text)
        self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.real_space_view_text)
        self.statusBar().addPermanentWidget(VLine())
        self.diffraction_rescale_button = LatchingButton(
            "Autoscale Diffraction",
            status_bar=self.statusBar(),
            latched=True,
        )
        self.diffraction_rescale_button.activated.connect(
            self.diffraction_space_widget.autoLevels
        )
        self.statusBar().addPermanentWidget(self.diffraction_rescale_button)
        self.realspace_rescale_button = LatchingButton(
            "Autoscale Real Space",
            status_bar=self.statusBar(),
            latched=True,
        )
        self.realspace_rescale_button.activated.connect(
            self.real_space_widget.autoLevels
        )
        self.statusBar().addPermanentWidget(self.realspace_rescale_button)
        
        # Make a virtual imaging tab, in the future more tabs can be added for
        # different views
        self.tab_widget.addTab(self.tab1, "Virtual Imaging")        
        self.layout.addWidget(self.tab_widget)

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
    
    def open_separate_window(self):
        self.separate_window = SeparateWindow()
        self.separate_window.on_fit_current_clicked(self.update_disk_detection)

class SeparateWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.layout = QVBoxLayout()
        # Create the fields
        label = QLabel("Disk Detection Parameters:")
        self.layout.addWidget(label)
        
        label = QLabel("Minimum absolute intensity:")
        self.layout.addWidget(label)
        self.min_intensity = QSpinBox()
        self.min_intensity.setMinimum(0)
        self.min_intensity.setMaximum(1000)
        self.min_intensity.setValue(0)
        self.layout.addWidget(self.min_intensity)
        
        label = QLabel("Minimum relative intensity:")
        self.layout.addWidget(label)
        self.rel_intensity = QDoubleSpinBox()
        self.rel_intensity.setMinimum(0.0)
        self.rel_intensity.setMaximum(1.0)
        self.rel_intensity.setSingleStep(0.001)
        self.rel_intensity.setValue(0.005)
        self.layout.addWidget(self.rel_intensity)
        
        label = QLabel("Minimum peak spacing (pixels):")
        self.layout.addWidget(label)
        self.min_peak_spacing = QSpinBox()
        self.min_peak_spacing.setMinimum(0)
        self.min_peak_spacing.setValue(60)
        self.layout.addWidget(self.min_peak_spacing)
        
        label = QLabel("Edge Boundary (pixels):")
        self.layout.addWidget(label)
        self.edge_boundary = QSpinBox()
        self.edge_boundary.setMinimum(0)
        self.edge_boundary.setValue(20)
        self.layout.addWidget(self.edge_boundary)
        
        label = QLabel("Sigma:")
        self.layout.addWidget(label)
        self.sigma = QDoubleSpinBox()
        self.sigma.setMinimum(0.0)
        self.layout.addWidget(self.sigma)
        
        label = QLabel("Maximum number of peaks:")
        self.layout.addWidget(label)
        self.max_num_peaks = QSpinBox()
        self.max_num_peaks.setMinimum(0)
        self.max_num_peaks.setValue(70)
        self.layout.addWidget(self.max_num_peaks)
        
        label = QLabel("Correlation power:")
        self.layout.addWidget(label)
        self.corr_power = QDoubleSpinBox()
        self.corr_power.setValue(1.0)
        self.corr_power.setMaximum(1.0)
        self.corr_power.setMinimum(0.0)
        self.layout.addWidget(self.corr_power)
        
        label = QLabel("Subpixel:")
        self.layout.addWidget(label)
        self.subpixel = QComboBox()
        self.subpixel.addItems(["none", "poly", "multicorr"])
        self.subpixel.setCurrentText("multicorr")
        self.layout.addWidget(self.subpixel)
        
        label = QLabel("Check CUDA:")
        self.layout.addWidget(label)
        self.check_cuda = QCheckBox("Checklist")
        self.layout.addWidget(self.check_cuda)
        
        button_layout = QHBoxLayout()
        self.run_current = QPushButton("Fit Current View")
        self.run_all = LatchingButton('Enable/Disable Disk Detection')
        button_layout.addWidget(self.run_current)
        button_layout.addWidget(self.run_all)
        self.layout.addLayout(button_layout)
        
        self.setWindowTitle("Disk Detection Parameters:")

        # Set the layout for the SeparateWindow
        self.setLayout(self.layout)
        self.show()
        
    def on_fit_current_clicked(self, slot):
        self.run_current.clicked.connect(slot)
    
    def get_params_as_dict(self):
        
        params_dict = {
            "minAbsoluteIntensity": self.min_intensity.value(),
            "minRelativeIntensity": self.rel_intensity.value(),
            "minPeakSpacing": self.min_peak_spacing.value(),
            "edgeBoundary": self.edge_boundary.value(),
            "sigma": self.sigma.value(),
            "maxNumPeaks": self.max_num_peaks.value(),
            "corrPower": self.corr_power.value(),
            "subpixel": self.subpixel.currentText(),
            "CUDA": self.check_cuda.isChecked(),
        }
        
        return params_dict
