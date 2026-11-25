from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
)
import logging
from emdfile import Metadata


class MetadataViewer(QWidget):

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "py4DGUI.internal.metadata"

    # the plugin API version this plugin was written against
    api_version = (1, 6)

    uses_single_action = True
    display_name = "Show Metadata..."

    def __init__(self, api, plugin_action, **kwargs):
        super().__init__()

        self.api = api

        plugin_action.triggered.connect(self.launch_dialog)

    def close(self):
        pass

    def launch_dialog(self):
        dialog = MetadataDialog(self.api)
        dialog.open()


class MetadataDialog(QDialog):
    def __init__(self, api):
        # Parented to the API's dialog-parent widget so it shows over the
        # main window; the datacube is read through the API object.
        super().__init__(parent=api.qt_window)

        layout = QVBoxLayout(self)

        print(api.datacube.metadata)

        mdata = api.datacube.metadata | {"calibration": api.datacube.calibration}

        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(tree)

        for name, md in mdata.items():
            top = QTreeWidgetItem(tree)
            top.setText(0, name)

            # TODO: make this recursive to handle nested dicts
            for k in md.keys:
                entry = QTreeWidgetItem(top)
                entry.setText(0, k)
                entry.setText(1, str(md[k]))

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        cancel_button = QPushButton("Done")
        cancel_button.pressed.connect(self.close)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setMinimumSize(400, 600)
