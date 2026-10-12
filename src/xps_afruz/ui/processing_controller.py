"""Non-destructive high-resolution spectrum processing workflow."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from ..core.processing import (
    combine_spectra,
    derivative_spectrum,
    despike_spectrum,
    gaussian_smooth,
    normalize_spectrum,
    savitzky_golay_smooth,
    signal_to_noise,
    subtract_spectra,
)
from ..core.spectrum import Spectrum
from ..io.delimited import read_delimited_spectrum
from ..io.vgd import VGDReadError, read_vgd_spectra


class ProcessingControllerMixin:
    """Create derived spectra while retaining imported data and undo state."""

    def _apply_processed_spectrum(self, spectrum: Spectrum, *, message: str) -> None:
        """Move to a derived spectrum while retaining an in-memory undo step."""

        if self.spectrum is None:
            return
        was_screened = self.survey_analysis is not None
        self._spectrum_undo.append(self.spectrum)
        self.spectrum = spectrum
        self._store_active_workspace_spectrum(spectrum)
        self.fit_result = None
        self.peak_table.setRowCount(0)
        self._ties = []
        self._reset_survey_results(reset_preferences=False)
        self.setWindowTitle(f"XPS-master — {spectrum.label}")
        self.recalculate_background()
        if was_screened:
            self.analyze_survey()
        self.statusBar().showMessage(message, 8_000)

    def _require_spectrum_for_processing(self) -> Spectrum | None:
        if self.spectrum is None:
            self.show_error("No spectrum loaded", "Load a spectrum before applying a processing operation.")
            return None
        return self.spectrum

    def choose_region_of_interest(self) -> None:
        """Create a derived ROI rather than trimming the imported scan in place."""

        spectrum = self._require_spectrum_for_processing()
        if spectrum is None:
            return
        lower, accepted = QInputDialog.getDouble(
            self, "Region of interest", "Lower energy (eV):", spectrum.energy_range[0], spectrum.energy_range[0], spectrum.energy_range[1], 5
        )
        if not accepted:
            return
        upper, accepted = QInputDialog.getDouble(
            self, "Region of interest", "Upper energy (eV):", spectrum.energy_range[1], spectrum.energy_range[0], spectrum.energy_range[1], 5
        )
        if not accepted:
            return
        try:
            region = spectrum.region(lower, upper, label=f"{spectrum.label} ROI")
        except ValueError as error:
            self.show_error("Invalid region", str(error))
            return
        self._apply_processed_spectrum(region, message=f"Created ROI: {min(lower, upper):.3f}–{max(lower, upper):.3f} eV.")

    def use_visible_region_of_interest(self) -> None:
        spectrum = self._require_spectrum_for_processing()
        if spectrum is None:
            return
        lower, upper = self.spectrum_plot.getViewBox().viewRange()[0]
        try:
            region = spectrum.region(lower, upper, label=f"{spectrum.label} visible ROI")
        except ValueError as error:
            self.show_error("Invalid visible region", str(error))
            return
        self._apply_processed_spectrum(region, message="Created ROI from the visible plot range.")

    def process_normalize(self, mode: str) -> None:
        spectrum = self._require_spectrum_for_processing()
        if spectrum is None:
            return
        try:
            result = normalize_spectrum(spectrum, mode=mode)  # type: ignore[arg-type]
        except ValueError as error:
            self.show_error("Normalization failed", str(error))
            return
        self._apply_processed_spectrum(result, message=f"Created {mode}-normalized spectrum.")

    def process_gaussian_smoothing(self) -> None:
        spectrum = self._require_spectrum_for_processing()
        if spectrum is None:
            return
        sigma, accepted = QInputDialog.getDouble(self, "Gaussian smoothing", "Sigma (eV):", 0.30, 0.001, 100.0, 4)
        if not accepted:
            return
        try:
            result = gaussian_smooth(spectrum, sigma_eV=sigma)
        except ValueError as error:
            self.show_error("Gaussian smoothing failed", str(error))
            return
        self._apply_processed_spectrum(result, message=f"Created Gaussian-smoothed spectrum (σ={sigma:g} eV).")

    def process_savitzky_golay(self) -> None:
        spectrum = self._require_spectrum_for_processing()
        if spectrum is None:
            return
        window, accepted = QInputDialog.getDouble(self, "Savitzky–Golay", "Window (eV):", 1.00, 0.001, 100.0, 4)
        if not accepted:
            return
        order, accepted = QInputDialog.getInt(self, "Savitzky–Golay", "Polynomial order:", 3, 1, 9)
        if not accepted:
            return
        try:
            result = savitzky_golay_smooth(spectrum, window_eV=window, polynomial_order=order)
        except ValueError as error:
            self.show_error("Savitzky–Golay filtering failed", str(error))
            return
        self._apply_processed_spectrum(result, message=f"Created Savitzky–Golay spectrum ({window:g} eV, order {order}).")

    def process_derivative(self, order: int) -> None:
        spectrum = self._require_spectrum_for_processing()
        if spectrum is None:
            return
        try:
            result = derivative_spectrum(spectrum, order=order)  # type: ignore[arg-type]
        except ValueError as error:
            self.show_error("Derivative failed", str(error))
            return
        self._apply_processed_spectrum(result, message=f"Created derivative order {order} spectrum.")

    def process_despike(self) -> None:
        spectrum = self._require_spectrum_for_processing()
        if spectrum is None:
            return
        threshold, accepted = QInputDialog.getDouble(self, "Spike detection", "Threshold (robust sigma):", 6.0, 1.0, 100.0, 3)
        if not accepted:
            return
        try:
            result, detection = despike_spectrum(spectrum, threshold_sigma=threshold)
        except ValueError as error:
            self.show_error("Despike failed", str(error))
            return
        self._apply_processed_spectrum(result, message=f"Created despiked spectrum; replaced {len(detection.indices)} point(s).")

    def _load_secondary_spectrum(self) -> Spectrum | None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select spectrum for processing",
            self._last_directory,
            "Supported spectra (*.csv *.txt *.tsv *.dat *.vgd *.VGD);;All files (*)",
        )
        if not path:
            return None
        try:
            if Path(path).suffix.lower() == ".vgd":
                spectra = read_vgd_spectra(path)
                spectrum = self._select_vgd_spectrum(spectra)
            else:
                spectrum = read_delimited_spectrum(path)
        except (OSError, VGDReadError, ValueError) as error:
            self.show_error("Could not open secondary spectrum", str(error))
            return None
        if spectrum is not None:
            self._last_directory = str(Path(path).parent)
        return spectrum

    def process_combine(self, operation: str) -> None:
        spectrum = self._require_spectrum_for_processing()
        if spectrum is None:
            return
        secondary = self._load_secondary_spectrum()
        if secondary is None:
            return
        try:
            if operation == "subtract":
                result = subtract_spectra(spectrum, secondary)
            else:
                result = combine_spectra((spectrum, secondary), operation=operation)  # type: ignore[arg-type]
        except ValueError as error:
            self.show_error("Spectrum combination failed", str(error))
            return
        self._apply_processed_spectrum(result, message=f"Created {operation} spectrum from two inputs.")

    def show_noise_metrics(self) -> None:
        spectrum = self._require_spectrum_for_processing()
        if spectrum is None:
            return
        metrics = signal_to_noise(spectrum)
        QMessageBox.information(
            self,
            "Noise and signal-to-noise",
            f"Robust noise: {metrics.noise:.6g} {spectrum.intensity_unit}\n"
            f"Signal above median: {metrics.signal:.6g} {spectrum.intensity_unit}\n"
            f"SNR: {metrics.signal_to_noise:.4g}\n\n{metrics.method}",
        )

    def undo_processing(self) -> None:
        if not self._spectrum_undo:
            self.statusBar().showMessage("No processing step is available to undo.")
            return
        previous = self._spectrum_undo.pop()
        self.spectrum = previous
        self._store_active_workspace_spectrum(previous)
        self.fit_result = None
        self.peak_table.setRowCount(0)
        self._ties = []
        self._reset_survey_results(reset_preferences=False)
        self.setWindowTitle(f"XPS-master — {previous.label}")
        self.recalculate_background()
        self.statusBar().showMessage("Restored the previous spectrum processing step.")
