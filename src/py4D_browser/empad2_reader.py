import empad2
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication
import numpy as np
from py4D_browser.utils import StatusBarWriter


def set_empad2_sensor(self, sensor_name):
    self.empad2_calibrations = empad2.load_calibration_data(sensor=sensor_name)
    self.statusBar().showMessage(f"{sensor_name} calibrations loaded", 5_000)


def load_empad2_background(self):
    if self.empad2_calibrations is not None:
        filename = raw_file_dialog(self)
        self.empad2_background = empad2.load_background(
            filepath=filename, calibration_data=self.empad2_calibrations
        )
        self.statusBar().showMessage("Background data loaded", 5_000)
    else:
        QMessageBox.warning(
            self, "No calibrations loaded!", "Please select a sensor first"
        )


def load_empad2_dataset(self):
    if self.empad2_calibrations is not None:
        dummy_data = False
        if self.empad2_background is None:
            continue_wo_bkg = QMessageBox.question(
                self,
                "Load without background?",
                "Background data has not been loaded. Do you want to continue loading data?",
            )
            if continue_wo_bkg == QMessageBox.No:
                return
            else:
                self.empad2_background = {
                    "even": np.zeros((128, 128), dtype=np.float32),
                    "odd": np.zeros((128, 128), dtype=np.float32),
                }
                dummy_data = True

        filename = raw_file_dialog(self)
        self.datacube = empad2.load_dataset(
            filename,
            self.empad2_background,
            self.empad2_calibrations,
            _tqdm_args={
                "desc": "Loading",
                "file": StatusBarWriter(self.statusBar()),
                "mininterval": 1.0,
            },
        )

        if dummy_data:
            self.empad2_background = None

        self.update_diffraction_space_view(reset=True)
        self.update_real_space_view(reset=True)

        self.setWindowTitle(filename)

    else:
        QMessageBox.warning(
            self, "No calibrations loaded!", "Please select a sensor first"
        )


def raw_file_dialog(browser):
    filename = QFileDialog.getOpenFileName(
        browser,
        "Open EMPAD-G2 Data",
        "",
        "EMPAD-G2 Data (*.raw);;Any file(*)",
    )
    if filename is not None and len(filename[0]) > 0:
        return filename[0]
    else:
        print("File was invalid, or something?")
        raise ValueError("Could not read file")
