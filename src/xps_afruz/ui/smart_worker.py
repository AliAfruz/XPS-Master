"""Background worker for responsive smart-deconvolution searches."""

from __future__ import annotations

from threading import Event

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from ..core.spectrum import Spectrum
from ..fitting.smart_deconvolution import (
    SmartDeconvolutionCancelled,
    SmartDeconvolutionResult,
    SmartDeconvolutionSettings,
    smart_deconvolve,
)


class SmartDeconvolutionWorker(QObject):
    """Run bounded multi-start fits off the GUI thread and expose clear progress."""

    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(
        self, spectrum: Spectrum, background: np.ndarray, settings: SmartDeconvolutionSettings
    ) -> None:
        super().__init__()
        self._spectrum = spectrum
        self._background = np.asarray(background, dtype=float).copy()
        self._settings = settings
        self._cancel_event = Event()

    def cancel(self) -> None:
        """Request a cooperative stop after the current bounded candidate fit."""

        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = smart_deconvolve(
                self._spectrum,
                self._background,
                self._settings,
                progress_callback=self._report_progress,
                cancellation_callback=self._cancel_event.is_set,
            )
        except SmartDeconvolutionCancelled:
            self.cancelled.emit()
        except Exception as error:  # noqa: BLE001 - report worker failures to the GUI boundary
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)
        finally:
            self.finished.emit()

    def _report_progress(self, completed: int, total: int, message: str) -> None:
        percentage = int(round(100.0 * completed / max(total, 1)))
        self.progress.emit(min(max(percentage, 0), 100), message)
