from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QComboBox,
    QGroupBox,
    QWidget,
    QFormLayout,
    QScrollArea,
)
import logging


class LoggingConfigurationPlugin(QWidget):

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "py4DGUI.internal.logging"

    uses_single_action = True
    display_name = "Logger Settings..."

    def __init__(self, parent, plugin_action, **kwargs):
        super().__init__()

        self.parent = parent

        plugin_action.triggered.connect(self.launch_dialog)

    def close(self):
        pass

    def launch_dialog(self):
        dialog = LoggerDialog(parent=self.parent)
        dialog.open()


class LoggerDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)

        self.parent = parent
        self.layout = QVBoxLayout(self)

        ####### LAYOUT ########

        main_box = QGroupBox("Registered Loggers")
        # layout.addWidget(main_box)

        scroll = QScrollArea()
        scroll.setWidget(main_box)
        scroll.setWidgetResizable(True)
        self.layout.addWidget(scroll)

        form = QFormLayout()
        main_box.setLayout(form)

        # get all loggers
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]

        log_levels = ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for lg in loggers:
            selector = QComboBox()
            selector.addItems(log_levels)
            selector.setCurrentText(logging.getLevelName(lg.level))
            selector.currentTextChanged.connect(
                lambda text, lg=lg: self._update_logger_level(lg, text)
            )

            form.addRow(
                lg.name,
                selector,
            )

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        cancel_button = QPushButton("Done")
        cancel_button.clicked.connect(self.close)
        button_layout.addWidget(cancel_button)

        self.layout.addLayout(button_layout)

    def _update_logger_level(self, logger, level_text):
        """Update the logger level based on the selected text."""
        level_map = {
            "NOTSET": logging.NOTSET,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        logger.setLevel(level_map.get(level_text, logging.NOTSET))
