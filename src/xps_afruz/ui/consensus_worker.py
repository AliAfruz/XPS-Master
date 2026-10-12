"""Background worker for responsive stability-consensus ensembles."""

from __future__ import annotations

from threading import Event

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from ..core.spectrum import Spectrum
from ..fitting.stability_consensus import (
    StabilityConsensusCancelled,
    StabilityConsensusResult,
    StabilityConsensusSettings,
    stability_consensus,
)


class StabilityConsensusWorker(QObject):
    """Run the perturbation ensemble away from the GUI event loop."""

    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(
        self,
        spectrum: Spectrum,
        background: np.ndarray,
        settings: StabilityConsensusSettings,
        *,
        current_background_label: str,
    ) -> None:
        super().__init__()
        self._spectrum = spectrum
        self._background = np.asarray(background, dtype=float).copy()
        self._settings = settings
        self._current_background_label = current_background_label
        self._cancel_event = Event()

    def cancel(self) -> None:
        """Request a cooperative stop after the current bounded candidate."""

        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = stability_consensus(
                self._spectrum,
                self._background,
                self._settings,
                current_background_label=self._current_background_label,
                progress_callback=self._report_progress,
                cancellation_callback=self._cancel_event.is_set,
            )
        except StabilityConsensusCancelled:
            self.cancelled.emit()
        except Exception as error:  # noqa: BLE001 - cross the worker boundary as a message
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)
        finally:
            self.finished.emit()

    def _report_progress(self, completed: float, total: int, message: str) -> None:
        percentage = int(round(100.0 * completed / max(total, 1)))
        self.progress.emit(min(max(percentage, 0), 100), message)
