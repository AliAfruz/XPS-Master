"""Desktop application entry point."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox

from ..logging_config import configure_application_logging
from ..metadata import APP_VERSION
from .main_window import MainWindow


def _install_exception_hook(log_path: Path) -> None:
    """Log unexpected GUI-thread failures and show a recoverable report path."""

    previous_hook = sys.excepthook

    def report_exception(exception_type, exception, traceback) -> None:  # type: ignore[no-untyped-def]
        logging.getLogger("xps_afruz").critical(
            "Unhandled application exception",
            exc_info=(exception_type, exception, traceback),
        )
        if QApplication.instance() is not None:
            QMessageBox.critical(
                None,
                "XPS-master encountered an unexpected error",
                "The error was recorded safely. Your source data was not modified.\n\n"
                f"Diagnostic log:\n{log_path}",
            )
        previous_hook(exception_type, exception, traceback)

    sys.excepthook = report_exception


def main() -> int:
    """Start the XPS-master desktop interface."""

    logger, log_path = configure_application_logging()
    logger.info("Starting XPS-master with Python %s", sys.version.split()[0])
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("XPS-master")
    application.setOrganizationName("XPS-master")
    application.setApplicationVersion(APP_VERSION)
    application.setFont(QFont("Segoe UI Variable Text", 10))
    _install_exception_hook(log_path)
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window.offer_autosave_recovery)
    exit_code = application.exec()
    logger.info("XPS-master closed with exit code %s", exit_code)
    return exit_code
