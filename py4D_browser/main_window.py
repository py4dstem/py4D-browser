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
)
from PyQt5 import QtGui

import pyqtgraph as pg
import numpy as np

from functools import partial
from pathlib import Path

class DataViewer(QMainWindow):
    """
    The class is used by instantiating and then entering the main Qt loop with, e.g.:
        app = DataViewer(sys.argv)
        app.exec_()
    """

    from py4D_browser.menu_actions import (
        load_data_auto,
        load_data_bin,
        load_data_mmap,
        set_diffraction_scaling,
        set_vimg_scaling,
    )

    from py4D_browser.update_views import (
        update_diffraction_space_view,
        update_real_space_view
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
        diff_menu_separator = QAction("Diffraction", self)
        diff_menu_separator.setDisabled(True)
        self.scaling_menu.addAction(diff_menu_separator)

        diff_scale_linear_action = QAction("Linear", self)
        diff_scale_linear_action.triggered.connect(partial(self.set_diffraction_scaling,"linear"))
        self.scaling_menu.addAction(diff_scale_linear_action)

        diff_scale_log_action = QAction("Log", self)
        diff_scale_log_action.triggered.connect(partial(self.set_diffraction_scaling,"log"))
        self.scaling_menu.addAction(diff_scale_log_action)

        diff_scale_sqrt_action = QAction("Square Root", self)
        diff_scale_sqrt_action.triggered.connect(partial(self.set_diffraction_scaling,"sqrt"))
        self.scaling_menu.addAction(diff_scale_sqrt_action)

        self.scaling_menu.addSeparator()

        # Real space scaling
        vimg_menu_separator = QAction("Virtual Image", self)
        vimg_menu_separator.setDisabled(True)
        self.scaling_menu.addAction(vimg_menu_separator)

        vimg_scale_linear_action = QAction("Linear", self)
        vimg_scale_linear_action.triggered.connect(partial(self.set_vimg_scaling,"linear"))
        self.scaling_menu.addAction(vimg_scale_linear_action)

        vimg_scale_log_action = QAction("Log", self)
        vimg_scale_log_action.triggered.connect(partial(self.set_vimg_scaling,"log"))
        self.scaling_menu.addAction(vimg_scale_log_action)

        vimg_scale_sqrt_action = QAction("Square Root", self)
        vimg_scale_sqrt_action.triggered.connect(partial(self.set_vimg_scaling,"sqrt"))
        self.scaling_menu.addAction(vimg_scale_sqrt_action)


    def setup_views(self):
        # Set up the diffraction space window.
        self.diffraction_space_widget = pg.ImageView()
        self.diffraction_space_widget.setImage(np.zeros((512,512)))
        self.diffraction_space_view_text = pg.TextItem('Slice',(200,200,200),None,(0,1))
        self.diffraction_space_widget.addItem(self.diffraction_space_view_text)

        # Create virtual detector ROI selector
        self.virtual_detector_roi = pg.RectROI([256, 256], [50,50], pen=(3,9))
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChangeFinished.connect(self.update_real_space_view)

        # Name and return
        self.diffraction_space_widget.setWindowTitle('Diffraction Space')

        # Set up the real space window.
        self.real_space_widget = pg.ImageView()
        self.real_space_widget.setImage(np.zeros((512,512)))
        self.real_space_view_text = pg.TextItem('Scan pos.',(200,200,200),None,(0,1))
        self.real_space_widget.addItem(self.real_space_view_text)

        # Add point selector connected to displayed diffraction pattern
        self.real_space_point_selector = pg_point_roi(self.real_space_widget.getView())
        self.real_space_point_selector.sigRegionChanged.connect(self.update_diffraction_space_view)

        # Name and return
        self.real_space_widget.setWindowTitle('Real Space')

        layout = QHBoxLayout()
        layout.addWidget(self.diffraction_space_widget,1)
        layout.addWidget(self.real_space_widget,1)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


def pg_point_roi(view_box):
    """
    Point selection.  Based in pyqtgraph, and returns a pyqtgraph CircleROI object.
    This object has a sigRegionChanged.connect() signal method to connect to other functions.
    """
    circ_roi = pg.CircleROI( (-0.5,-0.5), (2,2), movable=True, pen=(0,9))
    h = circ_roi.addTranslateHandle((0.5,0.5))
    h.pen = pg.mkPen('r')
    h.update()
    view_box.addItem(circ_roi)
    circ_roi.removeHandle(0)
    return circ_roi