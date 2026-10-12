"""Survey screening, trace recovery, and RSF quantification workflow."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QInputDialog, QTableWidgetItem

from ..chemistry.quantification import CompositionResult, quantify_survey
from ..chemistry.survey import SurveyAnalysis, analyze_survey
from ..fitting.model import PeakComponent


class SurveyControllerMixin:
    """Coordinate cautious elemental screening and composition presentation."""

    def _on_element_scope_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_element_scope:
            return
        symbol = item.text().strip()
        if symbol not in self._element_scope_items:
            return
        if item.checkState() == Qt.CheckState.Checked:
            self._survey_element_scope.add(symbol)
        else:
            self._survey_element_scope.discard(symbol)
        self._element_scope_changed()

    def _set_element_scope(self, symbols: set[str], *, invalidate: bool = True) -> None:
        """Apply a complete search scope and synchronize every selector cell."""

        selected = set(symbols).intersection(self._element_scope_items)
        self._updating_element_scope = True
        for symbol, item in self._element_scope_items.items():
            item.setCheckState(Qt.CheckState.Checked if symbol in selected else Qt.CheckState.Unchecked)
        self._updating_element_scope = False
        self._survey_element_scope = selected
        if invalidate:
            self._element_scope_changed()
        else:
            self._update_element_scope_summary()

    def _select_common_survey_elements(self) -> None:
        common = {
            "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca",
            "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        }
        self._set_element_scope(common)

    def _update_element_scope_summary(self) -> None:
        selected = len(self._survey_element_scope)
        total = len(self._element_scope_items)
        trace_state = "TRACE ON" if self._trace_target_scope() else "TRACE PAUSED"
        self.element_scope_summary.setText(f"●  {selected} OF {total} SELECTED  •  {trace_state}")

    def _trace_target_scope(self) -> set[str]:
        """Return a bounded, deliberate element set for weak-line recovery."""

        if not hasattr(self, "trace_search_checkbox") or not self.trace_search_checkbox.isChecked():
            return set()
        if 0 < len(self._survey_element_scope) <= self.TRACE_TARGET_LIMIT:
            return set(self._survey_element_scope)
        return set()

    def _element_scope_changed(self) -> None:
        """Invalidate stale assignments when the allowed reference set changes."""

        self._update_element_scope_summary()
        if self.survey_analysis is not None:
            self._reset_survey_results(reset_preferences=False)
            self._store_active_workspace_analysis()
            self.render_spectrum()
        else:
            self._set_project_dirty()
        self.statusBar().showMessage(
            f"Survey search scope updated: {len(self._survey_element_scope)} element(s); "
            f"targeted trace recovery {'active' if self._trace_target_scope() else 'paused'}. Run the finder to refresh assignments.",
            6_000,
        )

    def analyze_survey(self) -> None:
        """Detect survey peaks and show cautious element-reference candidates."""

        if self.spectrum is None:
            self.show_error("No spectrum loaded", "Load a survey spectrum before finding possible elements.")
            return
        if not self._survey_element_scope:
            self.show_error("No elements selected", "Choose at least one element in the Element Search Scope table.")
            return
        try:
            self.survey_analysis = analyze_survey(
                self.spectrum,
                tolerance=float(self.survey_tolerance.value()),
                include_elements=self._survey_element_scope,
                exclude_elements=self._excluded_survey_elements,
                target_elements=self._trace_target_scope(),
                trace_minimum_snr=float(self.trace_snr.value()),
            )
        except ValueError as error:
            self.show_error("Survey screening failed", str(error))
            return
        self._populate_survey_results(self.survey_analysis)
        self.recalculate_quantification()
        self.survey_dock.show()
        self.quant_dock.show()
        self.survey_dock.raise_()
        self._store_active_workspace_analysis()
        self.render_spectrum()
        range_width = self.spectrum.energy_range[1] - self.spectrum.energy_range[0]
        scope = " This is a narrow region, so the list is especially provisional." if range_width < 150.0 else ""
        self.statusBar().showMessage(
            f"Survey finder detected {len(self.survey_analysis.peaks)} peaks and {len(self.survey_analysis.candidates)} possible element(s).{scope}",
            10_000,
        )

    def _populate_survey_results(self, analysis: SurveyAnalysis) -> None:
        """Fill the dock in rank order while retaining the match behind each row."""

        self._survey_table_matches = list(analysis.matches[:60])
        candidate_text = ", ".join(
            f"{candidate.symbol} ({candidate.confidence_label})" for candidate in analysis.candidates[:8]
        ) or "none within tolerance"
        scope_text = ", ".join(analysis.included_elements) if len(analysis.included_elements) <= 12 else f"{len(analysis.included_elements)} selected elements"
        excluded_text = f"<br>Excluded: {', '.join(analysis.excluded_elements)}" if analysis.excluded_elements else ""
        if analysis.trace_searched_elements:
            recovered = ", ".join(analysis.trace_recovered_elements) or "none above threshold"
            trace_text = (
                f"<br>Targeted trace pass: {recovered} "
                f"(local SNR ≥ {analysis.trace_minimum_snr:.1f}; review required)"
            )
        else:
            trace_text = "<br>Targeted trace pass: paused — narrow the checked scope to 12 elements or fewer"
        self.survey_summary.setText(
            f"<b>{len(analysis.candidates)} possible element(s)</b> from {len(analysis.peaks)} detected peak(s)<br>"
            f"Search scope: {scope_text}<br>"
            f"Top candidates: {candidate_text}<br>"
            f"{analysis.energy_scale.value.title()} scale · noise {analysis.noise:.3g} · detection proxy {analysis.detection_limit:.3g}<br>"
            f"{len(analysis.unassigned_peaks)} suspicious/unassigned peak(s) · match window ±{analysis.tolerance:.1f} eV"
            f"{trace_text}{excluded_text}"
        )
        self.survey_table.setRowCount(len(self._survey_table_matches))
        for row, match in enumerate(self._survey_table_matches):
            candidate = next((item for item in analysis.candidates if item.symbol == match.reference.symbol), None)
            feature = f"{match.reference.line} ({'Auger' if match.reference.feature_type == 'auger' else 'PE'})"
            if match.reference.screening_context:
                feature += f" · {match.reference.screening_context}"
            if match.peak.targeted:
                feature += " · trace"
            values = (
                f"{match.reference.symbol} — {match.reference.element}",
                feature,
                f"{match.reference_energy:.1f}",
                f"{match.peak.maximum_energy:.1f}",
                f"{match.energy_error:+.1f}",
                f"{match.peak.signal_to_noise:.0f}",
                f"{match.score:.0f}",
            )
            secondaries = ""
            if candidate and candidate.expected_secondary_lines:
                secondaries = "; expected companions: " + ", ".join(
                    f"{line.reference.label} @ {line.expected_energy:.1f} ({'seen' if line.observed else 'not seen'})"
                    for line in candidate.expected_secondary_lines
                )
            overlap = f"; overlap candidates: {', '.join(match.overlapping_elements)}" if match.overlapping_elements else ""
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(
                    f"{match.reference.label}: reference {match.reference_energy:.2f} eV; "
                    f"maximum {match.peak.maximum_energy:.2f} eV; centroid {match.peak.centroid_energy:.2f} eV; "
                    f"ΔE {match.energy_error:+.2f} eV; SNR {match.peak.signal_to_noise:.1f}; screening rank {match.score:.1f}"
                    f"{secondaries}{overlap}."
                )
                self.survey_table.setItem(row, column, item)
        if not self._survey_table_matches:
            self.survey_summary.setText(
                f"<b>No reference line matched</b> the {len(analysis.peaks)} detected peak(s) within ±{analysis.tolerance:.1f} eV.\n"
                f"Searched elements: {scope_text}. Try checking energy calibration or widening the tolerance only when scientifically justified."
            )

    def recalculate_quantification(self) -> None:
        """Refresh the reproducible atomic-percent preview from current candidates."""

        if self.survey_analysis is None:
            self.quantification = None
            return
        try:
            self.quantification = quantify_survey(self.survey_analysis, factor_overrides=self._rsf_overrides)
        except ValueError as error:
            self.show_error("Quantification preview failed", str(error))
            return
        self._populate_quantification(self.quantification)

    def _populate_quantification(self, result: CompositionResult) -> None:
        self._updating_quant_table = True
        try:
            self.quant_table.setRowCount(len(result.records))
            for row, record in enumerate(result.records):
                values = (
                    record.symbol,
                    record.line,
                    f"{record.area:.4g}",
                    f"{record.sensitivity_factor.value:.4g}",
                    f"{record.corrected_intensity:.4g}",
                    f"{'≈' if record.targeted_trace else ''}{record.atomic_percent:.2f} ± {record.atomic_percent_uncertainty:.2f}"
                    f"{' · trace' if record.targeted_trace else ''}",
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column != 3:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    else:
                        item.setData(Qt.ItemDataRole.UserRole, (record.symbol, record.line))
                        item.setToolTip(
                            f"Editable RSF. Current model: {record.sensitivity_factor.model}; "
                            f"transmission × analyzer = {record.sensitivity_factor.transmission_factor * record.sensitivity_factor.analyzer_factor:.4g}; "
                            f"source: {record.sensitivity_factor.source}"
                        )
                    if column == 5 and record.targeted_trace:
                        item.setToolTip(
                            f"Targeted weak-line estimate (local SNR {record.signal_to_noise:.2f}). "
                            "The uncertainty includes inverse-SNR and RSF terms; confirm with a high-resolution region."
                        )
                    self.quant_table.setItem(row, column, item)
            total_text = " + ".join(f"{record.symbol} {record.atomic_percent:.1f}%" for record in result.records) or "no quantifiable selected lines"
            missing = f"<br>Missing RSF: {', '.join(result.unquantified_elements)}" if result.unquantified_elements else ""
            scope_warning = (
                "<br><b>Scope warning:</b> one checked element normalizes to 100%. Check the relevant matrix elements too for a meaningful at.% comparison."
                if len(self._survey_element_scope) == 1 and result.records
                else ""
            )
            self.quant_summary.setText(
                f"<b>Atomic-percent preview</b><br>{total_text}<br>Model: {result.factor_model}{missing}{scope_warning}"
            )
        finally:
            self._updating_quant_table = False

    def _on_quantification_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_quant_table or item.column() != 3:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(key, tuple) or len(key) != 2:
            return
        try:
            value = float(item.text())
            if value <= 0.0:
                raise ValueError
        except ValueError:
            self.show_error("Invalid RSF", "RSF must be a positive number.")
            self.recalculate_quantification()
            return
        self._rsf_overrides[(str(key[0]), str(key[1]))] = value
        self.recalculate_quantification()
        self._set_project_dirty()

    def set_selected_survey_rsf(self) -> None:
        """Attach an explicit analyst-supplied RSF to the selected line."""

        row = self.survey_table.currentRow()
        if row < 0 or row >= len(self._survey_table_matches):
            self.show_error("No survey line selected", "Select a candidate line before entering its RSF.")
            return
        match = self._survey_table_matches[row]
        key = (match.reference.symbol, match.reference.line)
        existing = self._rsf_overrides.get(key)
        if existing is None and self.quantification is not None:
            record = next(
                (
                    record
                    for record in self.quantification.records
                    if record.symbol == key[0] and record.line == key[1]
                ),
                None,
            )
            existing = None if record is None else record.sensitivity_factor.value
        value, accepted = QInputDialog.getDouble(
            self,
            "Instrument sensitivity factor",
            f"RSF for {key[0]} {key[1]}\nUse a validated factor for this analyzer and acquisition setup:",
            1.0 if existing is None else float(existing),
            0.000001,
            1_000_000.0,
            6,
        )
        if not accepted:
            return
        self._rsf_overrides[key] = float(value)
        self.recalculate_quantification()
        self._store_active_workspace_analysis()
        self.statusBar().showMessage(
            f"Applied user-defined RSF {value:g} to {key[0]} {key[1]}; atomic percentages recalculated.",
            8_000,
        )

    def focus_selected_survey_match(self) -> None:
        """Zoom to the selected candidate without changing the spectrum or fit."""

        row = self.survey_table.currentRow()
        if row < 0 or row >= len(self._survey_table_matches) or self.spectrum is None:
            return
        match = self._survey_table_matches[row]
        width = max(12.0, self.survey_analysis.tolerance * 5.0) if self.survey_analysis else 12.0
        lower = max(float(self.spectrum.energy[0]), match.peak.maximum_energy - width)
        upper = min(float(self.spectrum.energy[-1]), match.peak.maximum_energy + width)
        self.spectrum_plot.setXRange(lower, upper, padding=0.04)
        self.statusBar().showMessage(
            f"Viewing {match.reference.label}: detected {match.peak.maximum_energy:.2f} eV, ΔE {match.energy_error:+.2f} eV.",
            8_000,
        )

    def seed_selected_survey_component(self) -> None:
        """Add an editable fit seed for one analyst-selected survey candidate."""

        row = self.survey_table.currentRow()
        if row < 0 or row >= len(self._survey_table_matches) or self.spectrum is None:
            self.show_error("No survey candidate selected", "Select a candidate row before seeding a fit component.")
            return
        match = self._survey_table_matches[row]
        x, y = self.spectrum.energy, self.spectrum.intensity
        step = float(np.median(np.diff(x)))
        nearest = int(np.argmin(np.abs(x - match.peak.maximum_energy)))
        baseline = float(self.background[nearest]) if self.background is not None else float(np.median(y))
        height = max(float(y[nearest] - baseline), float(np.max(y)) * 0.01, 1.0)
        fwhm = max(1.4, step * 2.0)
        area = height * fwhm * np.sqrt(np.pi / (4.0 * np.log(2.0)))
        base_name = match.reference.label
        existing_names = {row_data["name"] for row_data in self.peak_rows()}
        name = base_name
        suffix = 2
        while name in existing_names:
            name = f"{base_name} {suffix}"
            suffix += 1
        self.add_peak_row(
            PeakComponent(
                name,
                center=match.peak.maximum_energy,
                fwhm=fwhm,
                area=float(area),
                center_bounds=(float(x[0]), float(x[-1])),
                fwhm_bounds=(max(step / 2.0, 0.01), float(x[-1] - x[0]) * 2.0),
            )
        )
        self.fit_result = None
        self.fit_summary.setText(
            f"Fit seed added for <b>{match.reference.label}</b> at {match.peak.maximum_energy:.2f} eV. "
            "Review the range, line shape, and any doublet constraints before fitting."
        )
        self.render_spectrum()
        self._store_active_workspace_fit_state()
        self.statusBar().showMessage(f"Added editable fit seed: {name}.")

    def remove_selected_survey_element(self) -> None:
        """Exclude every line of the selected proposed element from this survey."""

        row = self.survey_table.currentRow()
        if row < 0 or row >= len(self._survey_table_matches):
            self.show_error("No survey element selected", "Select a survey row before removing its proposed element.")
            return
        symbol = self._survey_table_matches[row].reference.symbol
        self._excluded_survey_elements.add(symbol)
        self.analyze_survey()
        self.statusBar().showMessage(f"Removed {symbol} from this survey screen and composition preview.")

    def restore_survey_elements(self) -> None:
        """Restore previously excluded candidate elements for the active scan."""

        if not self._excluded_survey_elements:
            self.statusBar().showMessage("No survey elements are currently excluded.")
            return
        self._excluded_survey_elements.clear()
        if self.spectrum is not None:
            self.analyze_survey()
        self.statusBar().showMessage("Restored all survey elements.")

    def _reset_survey_results(self, *, reset_preferences: bool = False) -> None:
        """Forget candidates when a different spectrum replaces the current one."""

        self.survey_analysis = None
        self._survey_table_matches = []
        self.quantification = None
        if reset_preferences:
            self._excluded_survey_elements.clear()
            self._rsf_overrides.clear()
        if hasattr(self, "survey_table"):
            self.survey_table.setRowCount(0)
            self.survey_summary.setText("Run the finder on a survey spectrum to see possible elements.")
        if hasattr(self, "quant_table"):
            self._updating_quant_table = True
            self.quant_table.setRowCount(0)
            self._updating_quant_table = False
            self.quant_summary.setText("Run the survey finder to generate a composition preview.")

    def clear_survey_results(self) -> None:
        """Clear candidate markers and results, keeping the loaded spectrum intact."""

        self._reset_survey_results(reset_preferences=True)
        self._store_active_workspace_analysis()
        self.render_spectrum()
        self.statusBar().showMessage("Survey candidates cleared.")
