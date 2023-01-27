import py4DSTEM
from PyQt5.QtWidgets import QFileDialog


def load_data_auto(self):
    filename = self.show_file_dialog()
    self.load_file(filename)


def load_data_mmap(self):
    filename = self.show_file_dialog()
    self.load_file(filename, mmap=True)


def load_data_bin(self):
    # TODO: Ask user for binning level
    filename = self.show_file_dialog()
    self.load_file(filename, mmap=False, binning=4)


def load_file(self, filepath, mmap=False, binning=1):
    print(f"Loading file {filepath}")

    self.datacube = py4DSTEM.import_file(
        filepath,
        mem="MEMMAP" if mmap else "RAM",
        binfactor=binning,
    )

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(filepath)


def show_file_dialog(self):
    filename = QFileDialog.getOpenFileName(
        self,
        "Open 4D-STEM Data",
        "",
        "4D-STEM Data (*.dm4 *.raw *.mib *.gtg)",
    )
    if filename is not None and len(filename[0]) > 0:
        return filename[0]
    else:
        print("File was invalid, or something?")
        raise ValueError("Could not read file")
