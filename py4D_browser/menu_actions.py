def load_data_auto(self):
    pass


def load_data_mmap(self):
    pass


def load_data_bin(self):
    pass


def load_file(self, filepath, mmap=False, binning=1):
    print(f"Loading file {filepath}")
    self.setWindowTitle(filepath)


def set_diffraction_scaling(self, mode):
    print(mode)


def set_vimg_scaling(self, mode):
    pass
