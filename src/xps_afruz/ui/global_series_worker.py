"""Background worker for project-level simultaneous XPS fitting."""

from __future__ import annotations

from threading import Event

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from ..core.spectrum import Spectrum
from ..fitting.global_series import (
    GlobalSeriesCancelled,
    GlobalSeriesSettings,
    fit_global_series,
)
from ..fitting.model import ParameterTie, PeakComponent


class GlobalSeriesWorker(QObject):
    """Run immutable global-series inputs without blocking the Qt interface."""

    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(
        self,
        spectra: tuple[Spectrum, ...],
        backgrounds: tuple[np.ndarray, ...],
        components: tuple[PeakComponent, ...],
        ties: tuple[ParameterTie, ...],
        workspace_indices: tuple[int, ...],
        settings: GlobalSeriesSettings,
    ) -> None:
        super().__init__()
        self._spectra = spectra
        self._backgrounds = backgrounds
        self._components = components
        self._ties = ties
        self._workspace_indices = workspace_indices
        self._settings = settings
        self._cancelled = Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @Slot()
    def run(self) -> None:
        try:
            result = fit_global_series(
                self._spectra,
                self._backgrounds,
                self._components,
                self._ties,
                workspace_indices=self._workspace_indices,
                settings=self._settings,
                progress_callback=self.progress.emit,
                cancellation_callback=self._cancelled.is_set,
            )
        except GlobalSeriesCancelled:
            self.cancelled.emit()
        except Exception as error:  # Worker boundary emits concise UI failures.
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)
        finally:
            self.finished.emit()
