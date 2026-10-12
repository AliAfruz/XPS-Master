"""GUI lifecycle controller for cancellable stability-consensus analysis."""

from __future__ import annotations

from PySide6.QtCore import QThread

from ..fitting.stability_consensus import StabilityConsensusResult, StabilityConsensusSettings
from .consensus_worker import StabilityConsensusWorker


class ConsensusControllerMixin:
    """Coordinate one consensus worker without owning scientific calculations."""

    def start_stability_consensus(self) -> None:
        """Launch the perturbation ensemble without replacing the retained fit."""

        if self._global_series_thread is not None and self._global_series_thread.isRunning():
            self.show_error("Global fitting is running", "Finish or stop global multi-spectrum fitting first.")
            return
        if self._consensus_thread is not None and self._consensus_thread.isRunning():
            self.statusBar().showMessage("Stability-consensus analysis is already running.")
            return
        if self._smart_thread is not None and self._smart_thread.isRunning():
            self.show_error("Smart search is running", "Finish or stop smart deconvolution before starting consensus analysis.")
            return
        if self._comparison_thread is not None and self._comparison_thread.isRunning():
            self.show_error("Model comparison is running", "Finish or stop model comparison before consensus analysis.")
            return
        if self.spectrum is None or self.background is None:
            self.show_error("No fit-ready spectrum", "Load a spectrum and calculate an inspectable background first.")
            return
        try:
            settings = StabilityConsensusSettings(
                peak_count=int(self.smart_peak_count.value()),
                profile=str(self.smart_profile.currentData()),
                ensemble_runs=int(self.consensus_runs.value()),
                candidate_trials_per_run=int(self.consensus_starts.value()),
                weighting=self.fit_weighting,
                count_scale=self.poisson_count_scale,
                compare_neighbor_counts=self.consensus_compare_neighbors.isChecked(),
            )
        except ValueError as error:
            self.show_error("Consensus settings", str(error))
            return

        self._consensus_workspace_index = self._active_workspace_index
        self._consensus_target_spectrum = self.spectrum
        thread = QThread(self)
        worker = StabilityConsensusWorker(
            self.spectrum,
            self.background,
            settings,
            current_background_label=self._background_description(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._update_consensus_progress)
        worker.completed.connect(self._apply_consensus_result)
        worker.failed.connect(self._consensus_failed)
        worker.cancelled.connect(self._consensus_cancelled)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._consensus_finished)
        self._consensus_thread = thread
        self._consensus_worker = worker
        self._set_consensus_running(True)
        self.consensus_progress.setValue(0)
        self.consensus_progress.setFormat("0%")
        counts = settings.peak_counts
        total_models = settings.ensemble_runs * len(counts)
        total_starts = total_models * settings.candidate_trials_per_run
        self.consensus_status.setText(
            f"Preparing {settings.ensemble_runs} perturbation scenarios × K={','.join(map(str, counts))} × "
            f"up to {settings.candidate_trials_per_run} starts/model (up to {total_starts} bounded candidate fits)."
        )
        self.statusBar().showMessage(
            f"Stability consensus started: {total_models} scenario/model combinations in a background worker.",
            10_000,
        )
        thread.start()

    def cancel_stability_consensus(self) -> None:
        """Stop the ensemble cooperatively without losing its previous report."""

        if self._consensus_worker is None:
            return
        self._consensus_worker.cancel()
        self.consensus_cancel_button.setEnabled(False)
        self.consensus_status.setText("Stopping after the current bounded candidate fit finishes…")

    def _set_consensus_running(self, running: bool) -> None:
        self.consensus_start_button.setEnabled(not running)
        self.consensus_cancel_button.setEnabled(running)
        self.consensus_runs.setEnabled(not running)
        self.consensus_starts.setEnabled(not running)
        self.consensus_compare_neighbors.setEnabled(not running)
        self.smart_start_button.setEnabled(not running)
        self.smart_peak_count.setEnabled(not running)
        self.smart_profile.setEnabled(not running)
        self.smart_trials.setEnabled(not running)
        self.comparison_start_button.setEnabled(not running)

    def _update_consensus_progress(self, percentage: int, message: str) -> None:
        self.consensus_progress.setValue(percentage)
        self.consensus_progress.setFormat(f"{percentage}%")
        self.consensus_status.setText(message)

    def _apply_consensus_result(self, result: StabilityConsensusResult) -> None:
        """Retain a report only on the spectrum that launched its ensemble."""

        if self.spectrum is not self._consensus_target_spectrum or self._active_workspace_index != self._consensus_workspace_index:
            self.consensus_status.setText(
                "Consensus completed after the active spectrum changed; the result was kept out of the wrong card."
            )
            self.statusBar().showMessage("Consensus result was not attached to a different workspace spectrum.", 8_000)
            return
        self._consensus_result = result
        index = self._active_workspace_index
        if index is not None:
            self._workspace_consensus_results[index] = result
        supported = sum(component.identifiability == "STRONG NUMERICAL SUPPORT" for component in result.components)
        self.consensus_progress.setValue(100)
        self.consensus_progress.setFormat("100%")
        self.consensus_status.setText(
            f"Trust Report ready: {supported}/{len(result.components)} requested components have strong numerical "
            f"support; combined BIC/AICc evidence prefers K={result.preferred_peak_count}."
        )
        self.consensus_report_button.setEnabled(True)
        self._refresh_workspace_views()
        self._set_project_dirty()
        self.statusBar().showMessage(
            "Stability-consensus analysis complete. Open the Trust Report; the retained deconvolution was not replaced.",
            10_000,
        )

    def _consensus_failed(self, message: str) -> None:
        self.consensus_progress.setValue(0)
        self.consensus_progress.setFormat("Not completed")
        self.consensus_status.setText(f"Consensus analysis could not complete: {message}")
        self.statusBar().showMessage("Stability-consensus analysis did not complete.", 8_000)

    def _consensus_cancelled(self) -> None:
        self.consensus_progress.setFormat("Stopped")
        if self._consensus_result is None:
            self.consensus_status.setText("Consensus stopped; no partial Trust Report was applied.")
        else:
            self.consensus_status.setText("Consensus stopped; the previous completed Trust Report remains available.")
        self.statusBar().showMessage("Consensus analysis stopped safely.", 8_000)

    def _consensus_finished(self) -> None:
        self._consensus_thread = None
        self._consensus_worker = None
        self._set_consensus_running(False)
