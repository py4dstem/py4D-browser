from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
    QMenu,
    QAction,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QPushButton,
    QScrollArea,
    QCheckBox,
    QLineEdit,
    QRadioButton,
    QButtonGroup,
    QDesktopWidget,
    QMessageBox,
    QActionGroup,
)
from PyQt5 import QtGui

import pyqtgraph as pg
import numpy as np

from functools import partial
from pathlib import Path

from py4D_browser.utils import pg_point_roi


class DataViewer(QMainWindow):
    """
    The class is used by instantiating and then entering the main Qt loop with, e.g.:
        app = DataViewer(sys.argv)
        app.exec_()
    """

    LOG_SCALE_MIN_VALUE = 1e-6

    from py4D_browser.menu_actions import (
        load_file,
        load_data_auto,
        load_data_bin,
        load_data_mmap,
        show_file_dialog,
    )

    from py4D_browser.update_views import (
        update_diffraction_space_view,
        update_real_space_view,
        update_diffraction_detector,
        update_annulus_pos,
        update_annulus_radii,
    )

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

        self.setup_menus()
        self.setup_views()

        self.resize(800, 400)

        self.show()

    def setup_menus(self):
        self.menu_bar = self.menuBar()

        # File menu
        self.file_menu = QMenu("&File", self)
        self.menu_bar.addMenu(self.file_menu)

        self.load_auto_action = QAction("&Load Data...", self)
        self.load_auto_action.triggered.connect(self.load_data_auto)
        self.file_menu.addAction(self.load_auto_action)

        self.load_mmap_action = QAction("Load &Memory Map...", self)
        self.load_mmap_action.triggered.connect(self.load_data_mmap)
        self.file_menu.addAction(self.load_mmap_action)

        self.load_binned_action = QAction("Load Data &Binned...", self)
        self.load_binned_action.triggered.connect(self.load_data_bin)
        self.file_menu.addAction(self.load_binned_action)

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

        # detector_iCoM = QAction("i&CoM", self)
        # detector_iCoM.setCheckable(True)
        # detector_iCoM.triggered.connect(partial(self.update_real_space_view, True))
        # detector_mode_group.addAction(detector_iCoM)
        # self.detector_menu.addAction(detector_iCoM)

        # Detector Shape Menu
        self.detector_shape_menu = QMenu("Detector &Shape", self)
        self.menu_bar.addMenu(self.detector_shape_menu)

        detector_shape_group = QActionGroup(self)
        detector_shape_group.setExclusive(True)
        self.detector_shape_group = detector_shape_group

        detector_point_action = QAction("&Point", self)
        detector_point_action.setCheckable(True)
        detector_point_action.setChecked(True) # Default
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

    def setup_views(self):
        # Set up the diffraction space window.
        self.diffraction_space_widget = pg.ImageView()
        self.diffraction_space_widget.setImage(np.zeros((512, 512)))
        self.diffraction_space_view_text = pg.TextItem(
            "Slice", (200, 200, 200), None, (0, 1)
        )
        self.diffraction_space_widget.addItem(self.diffraction_space_view_text)

        # Create virtual detector ROI selector
        self.virtual_detector_point = pg_point_roi(self.diffraction_space_widget.getView())
        self.virtual_detector_point.sigRegionChanged.connect(
            self.update_real_space_view
        )
        # self.virtual_detector_roi = pg.RectROI([5, 5], [20, 20], pen=(3, 9))
        # self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        # self.virtual_detector_roi.sigRegionChangeFinished.connect(
        #     partial(self.update_real_space_view, False)
        # )

        # Name and return
        self.diffraction_space_widget.setWindowTitle("Diffraction Space")

        # Set up the real space window.
        self.real_space_widget = pg.ImageView()
        self.real_space_widget.setImage(np.zeros((512, 512)))
        self.real_space_view_text = pg.TextItem(
            "Scan pos.", (200, 200, 200), None, (0, 1)
        )
        self.real_space_widget.addItem(self.real_space_view_text)

        # Add point selector connected to displayed diffraction pattern
        self.real_space_point_selector = pg_point_roi(self.real_space_widget.getView())
        self.real_space_point_selector.sigRegionChanged.connect(
            partial(self.update_diffraction_space_view, False)
        )

        # Name and return
        self.real_space_widget.setWindowTitle("Real Space")

        self.diffraction_space_widget.setAcceptDrops(True)
        self.real_space_widget.setAcceptDrops(True)
        self.diffraction_space_widget.dragEnterEvent = self.dragEnterEvent
        self.real_space_widget.dragEnterEvent = self.dragEnterEvent
        self.diffraction_space_widget.dropEvent = self.dropEvent
        self.real_space_widget.dropEvent = self.dropEvent

        layout = QHBoxLayout()
        layout.addWidget(self.diffraction_space_widget, 1)
        layout.addWidget(self.real_space_widget, 1)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

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

