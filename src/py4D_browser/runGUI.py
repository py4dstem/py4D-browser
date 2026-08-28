#!/usr/bin/env python

import click
import sys
from PyQt5.QtWidgets import QApplication

import py4D_browser


@click.command()
@click.argument("filepath", required=False, default=None)
@click.option(
    "--reset-state",
    is_flag=True,
    default=False,
    help="Clear saved window state and use defaults.",
)
@click.option(
    "-dc",
    "--debug-console",
    is_flag=True,
    default=False,
    help="Launch pyqtgraph's debug console. (also via PY4DGUI_DEBUG env var)",
)
def launch(filepath, reset_state, debug_console):
    """Launch the py4DSTEM browser GUI.

    FILEPATH is an optional path to a data file to open on startup.
    """
    app = QApplication(sys.argv)
    win = py4D_browser.DataViewer(
        filepath=filepath,
        reset_state=reset_state,
        debug_console=debug_console,
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch()
