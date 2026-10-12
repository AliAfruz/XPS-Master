"""Multi-spectrum workspace state, navigation, and card presentation."""

from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFileDialog, QFrame, QLabel, QListWidgetItem, QVBoxLayout

from ..backgrounds import BackgroundMethod
from ..chemistry.survey import SurveyAnalysis, analyze_survey
from ..core.spectrum import Spectrum
from ..fitting.model import FitResult, PeakComponent
from ..fitting.model_comparison import ModelComparisonResult
from ..fitting.peak_shapes import PeakProfile
from ..fitting.stability_consensus import StabilityConsensusResult
from ..io.delimited import read_delimited_spectrum
from ..io.vgd import VGDReadError, read_vgd_spectra


class WorkspaceControllerMixin:
    """Retain independent spectrum, fit, survey, and Trust state per card."""

    def show_workspace_board(self) -> None:
        """Reveal and foreground the multi-survey workspace board."""

        self.workspace_dock.show()
        self.workspace_dock.raise_()

    def add_workspace_spectra(self, spectra: list[Spectrum], *, activate: bool) -> None:
        """Add imported spectra to the board without discarding existing surveys."""

        if not spectra:
            return
        start_index = len(self._workspace_spectra)
        self._workspace_spectra.extend(spectra)
        self._workspace_analyses.extend([None] * len(spectra))
        self._workspace_background_settings.extend(
            [(BackgroundMethod.LINEAR, self._default_background_parameters(BackgroundMethod.LINEAR)) for _ in spectra]
        )
        self._workspace_fit_results.extend([None] * len(spectra))
        self._workspace_peak_rows.extend([[] for _ in spectra])
        self._workspace_ties.extend([[] for _ in spectra])
        self._workspace_consensus_results.extend([None] * len(spectra))
        self._workspace_model_comparisons.extend([None] * len(spectra))
        if hasattr(self, "global_start_button"):
            running = self._global_series_thread is not None and self._global_series_thread.isRunning()
            self.global_start_button.setEnabled(not running and len(self._workspace_spectra) >= 2)
        self._refresh_workspace_views()
        if len(self._workspace_spectra) > 1:
            self.show_workspace_board()
        if activate:
            self.activate_workspace_spectrum(start_index)
        else:
            self.workspace_summary.setText(
                f"{len(self._workspace_spectra)} spectra loaded. Select a card or navigator row to make it active."
            )
        self._set_project_dirty()

    def load_workspace_surveys(self) -> None:
        """Import several text and VGD surveys into the selectable workspace."""

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load survey spectra into workspace",
            self._last_directory,
            "Supported spectra (*.csv *.txt *.tsv *.dat *.vgd *.VGD);;All files (*)",
        )
        if not paths:
            return
        imported: list[Spectrum] = []
        errors: list[str] = []
        for raw_path in paths:
            path = Path(raw_path)
            try:
                spectra = read_vgd_spectra(path) if path.suffix.lower() == ".vgd" else [read_delimited_spectrum(path)]
            except (OSError, VGDReadError, ValueError) as error:
                errors.append(f"{path.name}: {error}")
                continue
            imported.extend(spectra)
        if not imported:
            self.show_error("No surveys loaded", "\n".join(errors) or "No readable spectrum was selected.")
            return
        self._last_directory = str(Path(paths[0]).parent)
        self.add_workspace_spectra(imported, activate=self.spectrum is None)
        self.show_workspace_board()
        message = f"Added {len(imported)} spectrum(s) to the multi-survey board."
        if errors:
            message += f" {len(errors)} file(s) could not be imported."
        self.statusBar().showMessage(message, 10_000)

    def activate_workspace_spectrum(self, index: int) -> None:
        """Make a board spectrum the one used by all fitting and analysis tools."""

        if self._updating_workspace_selection or not 0 <= index < len(self._workspace_spectra):
            return
        if index == self._active_workspace_index and self.spectrum is not None:
            return
        previous = self._active_workspace_index
        if previous is not None and 0 <= previous < len(self._workspace_background_settings):
            self._workspace_background_settings[previous] = (
                self.background_method,
                dict(self._background_parameters),
            )
        self._store_active_workspace_fit_state(refresh=False, mark_dirty=False)
        self._active_workspace_index = index
        self._updating_workspace_selection = True
        self.workspace_list.setCurrentRow(index)
        self._updating_workspace_selection = False
        method, parameters = self._workspace_background_settings[index]
        self.background_combo.blockSignals(True)
        self.background_combo.setCurrentIndex(self.background_combo.findData(method.value))
        self.background_combo.blockSignals(False)
        self._background_parameters = dict(parameters)
        spectrum = self._workspace_spectra[index]
        self.set_spectrum(spectrum, clear_components=True)
        self._restore_workspace_fit_state(index)
        self._restore_workspace_consensus_state(index)
        self._restore_workspace_model_comparison_state(index)
        stored_analysis = self._workspace_analyses[index]
        if stored_analysis is not None:
            self.survey_analysis = stored_analysis
            self._populate_survey_results(stored_analysis)
            self.recalculate_quantification()
        self.workflow_tabs.setCurrentIndex(0)
        self._refresh_workspace_views()
        self.statusBar().showMessage(
            f"Active workspace spectrum: {spectrum.label}. This is now the target for fitting and processing.", 8_000
        )

    def _store_active_workspace_spectrum(self, spectrum: Spectrum) -> None:
        """Keep non-destructive processing and calibration attached to its board card."""

        index = self._active_workspace_index
        if index is None or not 0 <= index < len(self._workspace_spectra):
            return
        if self._global_series_result is not None and index in self._global_series_result.workspace_indices:
            self._invalidate_global_series("A spectrum in the retained series changed; rerun global fitting.")
        self._workspace_spectra[index] = spectrum
        self._workspace_analyses[index] = None
        self._workspace_fit_results[index] = None
        self._workspace_peak_rows[index] = []
        self._workspace_ties[index] = []
        self._workspace_consensus_results[index] = None
        self._workspace_model_comparisons[index] = None
        self._consensus_result = None
        self._model_comparison_result = None
        if hasattr(self, "consensus_report_button"):
            self.consensus_report_button.setEnabled(False)
            self.consensus_status.setText("Spectrum changed; run stability consensus to create a new Trust Report.")
        if hasattr(self, "comparison_report_button"):
            self.comparison_report_button.setEnabled(False)
            self.comparison_status.setText("Spectrum changed; build a new model comparison.")
        self._refresh_workspace_views()
        self._set_project_dirty()

    def _store_active_workspace_fit_state(self, *, refresh: bool = True, mark_dirty: bool = True) -> None:
        """Save the editable model and last result on the active board card."""

        index = self._active_workspace_index
        if index is None or not 0 <= index < len(self._workspace_spectra):
            return
        try:
            rows = self.peak_rows()
        except ValueError:
            return
        self._workspace_peak_rows[index] = [dict(row) for row in rows]
        self._workspace_ties[index] = list(self._ties)
        self._workspace_fit_results[index] = self.fit_result
        if refresh:
            self._refresh_workspace_views()
        if mark_dirty:
            self._set_project_dirty()

    def _restore_workspace_fit_state(self, index: int) -> None:
        """Restore one card's editable components, constraints, and fit curves."""

        self.peak_table.setRowCount(0)
        for row in self._workspace_peak_rows[index]:
            self.add_peak_row(
                PeakComponent(
                    str(row["name"]),
                    center=float(row["center"]),
                    fwhm=float(row["fwhm"]),
                    area=float(row["area"]),
                    profile=str(row.get("profile", PeakProfile.PSEUDO_VOIGT.value)),
                    eta=float(row.get("eta", 0.5)),
                    asymmetry=float(row.get("asymmetry", 0.0)),
                    instrument_fwhm=float(row.get("instrument_fwhm", 0.0)),
                    vary_eta=bool(row.get("vary_eta", False)),
                    vary_asymmetry=bool(row.get("vary_asymmetry", False)),
                    vary_instrument_fwhm=bool(row.get("vary_instrument_fwhm", False)),
                ),
                seed_center=float(row.get("seed_center", row["center"])),
            )
        self._ties = list(self._workspace_ties[index])
        self.fit_result = self._workspace_fit_results[index]
        if self.fit_result is not None:
            self._update_table_values(self.fit_result.parameter_values)
            self.fit_summary.setText(
                f"<b>Saved fit restored</b><br>"
                f"{len(self.fit_result.components)} component(s) • {self._fit_metric_label(self.fit_result)}<br>"
                f"Background: {self._background_description()}<br>"
                f"{self._fit_quality_lines(self.fit_result)}"
            )
        else:
            self.fit_summary.setText("No fit has been run for this workspace spectrum.")
        self.render_spectrum()

    def _store_active_workspace_analysis(self) -> None:
        index = self._active_workspace_index
        if index is None or not 0 <= index < len(self._workspace_analyses):
            return
        self._workspace_analyses[index] = self.survey_analysis
        self._refresh_workspace_views()
        self._set_project_dirty()

    def _restore_workspace_consensus_state(self, index: int) -> None:
        """Restore the retained Trust Report belonging to one workspace card."""

        self._consensus_result = self._workspace_consensus_results[index]
        available = self._consensus_result is not None
        self.consensus_report_button.setEnabled(available)
        if not available:
            self.consensus_progress.setValue(0)
            self.consensus_progress.setFormat("Ready")
            self.consensus_status.setText(
                "No Trust Report for this spectrum. Run consensus after selecting the intended ROI and background."
            )
            return
        result = self._consensus_result
        supported = sum(component.identifiability == "STRONG NUMERICAL SUPPORT" for component in result.components)
        self.consensus_progress.setValue(100)
        self.consensus_progress.setFormat("Saved")
        self.consensus_status.setText(
            f"Saved Trust Report: {supported}/{len(result.components)} component(s) have strong numerical support; "
            f"preferred numerical count K={result.preferred_peak_count}."
        )

    def _invalidate_active_consensus(self, reason: str) -> None:
        """Remove a stale report after a consensus input changes."""

        index = self._active_workspace_index
        if index is not None and 0 <= index < len(self._workspace_consensus_results):
            self._workspace_consensus_results[index] = None
        self._consensus_result = None
        if hasattr(self, "consensus_report_button"):
            self.consensus_report_button.setEnabled(False)
            self.consensus_progress.setValue(0)
            self.consensus_progress.setFormat("Ready")
            self.consensus_status.setText(reason)

    def _restore_workspace_model_comparison_state(self, index: int) -> None:
        """Restore the retained alternative-model dashboard for one card."""

        self._model_comparison_result = self._workspace_model_comparisons[index]
        available = self._model_comparison_result is not None
        self.comparison_report_button.setEnabled(available)
        if not available:
            self.comparison_progress.setValue(0)
            self.comparison_progress.setFormat("Ready")
            self.comparison_status.setText(
                "No comparison retained. Build alternatives from the active constrained fit."
            )
            return
        result = self._model_comparison_result
        self.comparison_progress.setValue(100)
        self.comparison_progress.setFormat("Saved")
        self.comparison_status.setText(
            f"Saved comparison: {len(result.candidates)} candidates · recommended numerical model "
            f"{result.recommended.candidate.label} ({result.recommended.model_support:.1%} support)."
        )

    def _invalidate_active_model_comparison(self, reason: str) -> None:
        """Remove alternatives made stale by a changed spectrum or fit hypothesis."""

        if getattr(self, "_applying_model_comparison", False):
            return
        index = self._active_workspace_index
        if index is not None and 0 <= index < len(self._workspace_model_comparisons):
            self._workspace_model_comparisons[index] = None
        self._model_comparison_result = None
        if hasattr(self, "comparison_report_button"):
            self.comparison_report_button.setEnabled(False)
            self.comparison_progress.setValue(0)
            self.comparison_progress.setFormat("Ready")
            self.comparison_status.setText(reason)

    def screen_workspace_surveys(self) -> None:
        """Run the survey screen for every loaded spectrum and update all cards."""

        if not self._workspace_spectra:
            self.show_error("No workspace spectra", "Load one or more surveys into the workspace first.")
            return
        if not self._survey_element_scope:
            self.show_error("No elements selected", "Choose at least one element in the Element Search Scope table.")
            return
        tolerance = float(self.survey_tolerance.value())
        screened = 0
        for index, spectrum in enumerate(self._workspace_spectra):
            try:
                self._workspace_analyses[index] = analyze_survey(
                    spectrum,
                    tolerance=tolerance,
                    include_elements=self._survey_element_scope,
                    exclude_elements=self._excluded_survey_elements,
                    target_elements=self._trace_target_scope(),
                    trace_minimum_snr=float(self.trace_snr.value()),
                )
                screened += 1
            except ValueError:
                self._workspace_analyses[index] = None
        if self._active_workspace_index is not None:
            active_analysis = self._workspace_analyses[self._active_workspace_index]
            if active_analysis is not None:
                self.survey_analysis = active_analysis
                self._populate_survey_results(active_analysis)
                self.recalculate_quantification()
        self._refresh_workspace_views()
        self._set_project_dirty()
        self.statusBar().showMessage(f"Screened {screened} workspace spectrum(s) for possible elements.", 8_000)

    def remove_active_workspace_spectrum(self) -> None:
        """Remove the selected board card without touching files on disk."""

        index = self._active_workspace_index
        if index is None or not 0 <= index < len(self._workspace_spectra):
            self.show_error("No active workspace spectrum", "Select a spectrum card or navigator row first.")
            return
        if self._global_series_result is not None:
            self._invalidate_global_series("The workspace membership changed; select and refit the global series.")
        removed = self._workspace_spectra.pop(index)
        self._workspace_analyses.pop(index)
        self._workspace_background_settings.pop(index)
        self._workspace_fit_results.pop(index)
        self._workspace_peak_rows.pop(index)
        self._workspace_ties.pop(index)
        self._workspace_consensus_results.pop(index)
        self._workspace_model_comparisons.pop(index)
        if self._workspace_spectra:
            self._active_workspace_index = None
            self._refresh_workspace_views()
            self.activate_workspace_spectrum(min(index, len(self._workspace_spectra) - 1))
        else:
            self._active_workspace_index = None
            self._refresh_workspace_views()
        self._set_project_dirty()
        if hasattr(self, "global_start_button"):
            running = self._global_series_thread is not None and self._global_series_thread.isRunning()
            self.global_start_button.setEnabled(not running and len(self._workspace_spectra) >= 2)
        self.statusBar().showMessage(f"Removed {removed.label} from the workspace board. Source file was not changed.", 6_000)

    def _refresh_workspace_views(self) -> None:
        """Rebuild compact visual cards and keep navigator selection synchronized."""

        if not hasattr(self, "workspace_grid"):
            return
        if hasattr(self, "global_start_button"):
            running = self._global_series_thread is not None and self._global_series_thread.isRunning()
            self.global_start_button.setEnabled(not running and len(self._workspace_spectra) >= 2)
        self._updating_workspace_selection = True
        self.workspace_list.clear()
        for index, spectrum in enumerate(self._workspace_spectra):
            analysis = self._workspace_analyses[index]
            summary = self._workspace_card_summary(
                analysis,
                self._workspace_fit_results[index],
                self._workspace_consensus_results[index],
                self._workspace_model_comparisons[index],
            )
            item = QListWidgetItem(f"{index + 1}. {spectrum.label}\n{summary}")
            item.setToolTip(f"{spectrum.size} points • {spectrum.energy_range[0]:.3f}–{spectrum.energy_range[1]:.3f} eV")
            self.workspace_list.addItem(item)
        if self._active_workspace_index is not None and self._active_workspace_index < self.workspace_list.count():
            self.workspace_list.setCurrentRow(self._active_workspace_index)
        self._updating_workspace_selection = False

        while self.workspace_grid.count():
            item = self.workspace_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        if not self._workspace_spectra:
            empty = QLabel("Load several surveys to compare them here. Click a card or navigator row to work on one spectrum.")
            empty.setObjectName("surveyNote")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.workspace_grid.addWidget(empty, 0, 0)
            self.workspace_summary.setText("Load one or more survey files to populate the visual board.")
            self._refresh_manuscript_runner()
            return

        active = self._active_workspace_index
        active_text = f" Active: {active + 1}." if active is not None else ""
        self.workspace_summary.setText(
            f"{len(self._workspace_spectra)} spectra loaded.{active_text} Click a plot card to choose it for analysis."
        )
        for index, spectrum in enumerate(self._workspace_spectra):
            card = self._workspace_card(
                index,
                spectrum,
                self._workspace_analyses[index],
                self._workspace_fit_results[index],
                self._workspace_consensus_results[index],
                self._workspace_model_comparisons[index],
            )
            self.workspace_grid.addWidget(card, index // 2, index % 2)
        self._refresh_manuscript_runner()

    def _workspace_card(
        self,
        index: int,
        spectrum: Spectrum,
        analysis: SurveyAnalysis | None,
        fit_result: FitResult | None,
        consensus_result: StabilityConsensusResult | None,
        comparison_result: ModelComparisonResult | None,
    ) -> QFrame:
        """Create one click-to-activate overview with its stored fit when available."""

        card = QFrame()
        card.setObjectName("plotCard")
        card.setMinimumSize(340, 210)
        if index == self._active_workspace_index:
            card.setStyleSheet("QFrame#plotCard { border: 2px solid #d0a158; background: #262a2b; }")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 11, 12, 12)
        layout.setSpacing(6)
        label = QLabel(f"{index + 1}. {spectrum.label}")
        label.setObjectName("plotTitle")
        label.setWordWrap(True)
        layout.addWidget(label)
        candidate_label = QLabel(self._workspace_card_summary(analysis, fit_result, consensus_result, comparison_result))
        candidate_label.setObjectName("surveyNote")
        candidate_label.setWordWrap(True)
        layout.addWidget(candidate_label)
        plot = pg.PlotWidget()
        plot.setMinimumHeight(135)
        self._configure_plot(plot, "Intensity", spectrum.intensity_unit)
        plot.setLabel("bottom", "ENERGY", units=spectrum.energy_unit, color="#c8d3d4")
        plot.plot(spectrum.energy, spectrum.intensity, pen=pg.mkPen("#e7efed", width=1.0), antialias=True)
        has_fit = fit_result is not None and fit_result.fitted_intensity.shape == spectrum.intensity.shape
        if has_fit:
            plot.plot(spectrum.energy, fit_result.background, pen=pg.mkPen("#d6ab62", width=0.8))
            colours = ("#f2a65a", "#ea7d70", "#bb91df", "#62cbb5", "#e7c15e", "#7fa9e5")
            for component_index, curve in enumerate(fit_result.components.values()):
                colour = QColor(colours[component_index % len(colours)])
                lower = plot.plot(spectrum.energy, fit_result.background, pen=pg.mkPen(0, 0, 0, 0))
                upper = plot.plot(spectrum.energy, curve + fit_result.background, pen=pg.mkPen(colour, width=0.8))
                fill = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(colour.red(), colour.green(), colour.blue(), 56))
                plot.addItem(fill)
            plot.plot(spectrum.energy, fit_result.fitted_intensity, pen=pg.mkPen("#fff6df", width=1.25))
        if analysis is not None:
            for match in analysis.matches[:8]:
                plot.addLine(
                    x=match.peak.maximum_energy,
                    pen=pg.mkPen("#8cd4b0", width=0.8, style=Qt.PenStyle.DashLine),
                )
        plot.scene().sigMouseClicked.connect(lambda _event, card_index=index: self.activate_workspace_spectrum(card_index))
        self._install_plot_copy_menu(plot)
        layout.addWidget(plot, stretch=1)
        return card

    @staticmethod
    def _workspace_candidate_summary(analysis: SurveyAnalysis | None) -> str:
        if analysis is None:
            return "Not screened for elements"
        candidates = ", ".join(candidate.symbol for candidate in analysis.candidates[:6])
        return f"Possible elements: {candidates or 'none within tolerance'}"

    @classmethod
    def _workspace_card_summary(
        cls,
        analysis: SurveyAnalysis | None,
        fit_result: FitResult | None,
        consensus_result: StabilityConsensusResult | None = None,
        comparison_result: ModelComparisonResult | None = None,
    ) -> str:
        """Summarize scientific state concisely for the navigator and board card."""

        parts: list[str] = []
        if fit_result is not None:
            component_count = len(fit_result.components)
            parts.append(f"Fit saved: {component_count} component{'s' if component_count != 1 else ''}, {cls._fit_metric_label(fit_result)}")
        if analysis is not None:
            parts.append(cls._workspace_candidate_summary(analysis))
        if consensus_result is not None:
            strong = sum(
                component.identifiability == "STRONG NUMERICAL SUPPORT"
                for component in consensus_result.components
            )
            parts.append(
                f"Trust: {strong}/{len(consensus_result.components)} strong, K={consensus_result.preferred_peak_count}"
            )
        if comparison_result is not None:
            parts.append(
                f"Models: {len(comparison_result.candidates)}, recommended {comparison_result.recommended.candidate.label}"
            )
        return " • ".join(parts) or "Ready for screening or fitting"
