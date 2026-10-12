"""Background selection, configuration, diagnostics, and fit weighting."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QInputDialog, QMessageBox

from ..backgrounds import (
    BackgroundMethod,
    BackgroundSettings,
    EdgeStatistic,
    TougaardVariant,
    diagnose_background,
    estimate_background,
)
from ..core.spectrum import Spectrum


class BackgroundControllerMixin:
    """Keep inspectable background choices independent from peak fitting."""

    def recalculate_background(self) -> None:
        if self.spectrum is None:
            return
        refresh_survey = self.survey_analysis is not None
        if self.background_method is BackgroundMethod.CUSTOM and "values" not in self._background_parameters:
            self.background = None
            self.background_diagnostics = None
            self.background_info.setText("Custom background · choose Configure background… to import an aligned trace.")
            self.render_spectrum()
            return
        try:
            self.background = estimate_background(self.spectrum, self.background_method, **self._background_parameters)
            self.background_diagnostics = diagnose_background(self.spectrum, self.background)
        except (TypeError, ValueError) as error:
            self.show_error("Background calculation failed", str(error))
            return
        self.background_info.setText(f"{self._background_description()}\n{self._background_diagnostic_summary()}")
        self.fit_result = None
        self.render_spectrum()
        if refresh_survey:
            self.analyze_survey()

    @property
    def background_method(self) -> BackgroundMethod:
        return BackgroundMethod(str(self.background_combo.currentData()))

    @property
    def fit_weighting(self) -> str:
        return str(self.fit_weighting_combo.currentData())

    @property
    def poisson_count_scale(self) -> float:
        return float(self.poisson_count_scale_spin.value())

    def on_fit_weighting_changed(self, _index: int = 0) -> None:
        """Make the statistical claim visible before launching a fit."""

        poisson = self.fit_weighting == "poisson"
        self.poisson_count_scale_spin.setEnabled(poisson)
        if poisson:
            self.fit_statistics_note.setText(
                "Poisson likelihood requires unprocessed counts. Pearson χ² is an asymptotic diagnostic; confirm the displayed-unit count scale."
            )
        else:
            self.fit_statistics_note.setText(
                "Uniform weighting reports residual RMS and a numerical objective only; it does not report a calibrated statistical χ²."
            )

    def _recommend_fit_weighting_for_spectrum(self, spectrum: Spectrum) -> None:
        """Select a transparent default from imported units and processing history."""

        unit = spectrum.intensity_unit.lower().replace(" ", "")
        count_scale = 1_000.0 if "kcount" in unit else 1_000_000.0 if "mcount" in unit else 1.0
        raw_counts = "count" in unit and not spectrum.processing_history
        self.fit_weighting_combo.blockSignals(True)
        self.fit_weighting_combo.setCurrentIndex(self.fit_weighting_combo.findData("poisson" if raw_counts else "uniform"))
        self.fit_weighting_combo.blockSignals(False)
        self.poisson_count_scale_spin.setValue(count_scale)
        self.on_fit_weighting_changed()

    @staticmethod
    def _default_background_parameters(method: BackgroundMethod) -> dict[str, object]:
        if method is BackgroundMethod.LINEAR:
            return {"edge_fraction": 0.03, "edge_statistic": EdgeStatistic.LOWER_QUANTILE.value}
        if method is BackgroundMethod.SHIRLEY:
            return {
                "edge_fraction": 0.03,
                "edge_statistic": EdgeStatistic.LOWER_QUANTILE.value,
                "max_iterations": 200,
                "tolerance": 1e-7,
            }
        if method is BackgroundMethod.TOUGAARD:
            return {
                "variant": TougaardVariant.U2.value,
                "b": 2866.0,
                "c": 1643.0,
                "d": 0.0,
                "scale": None,
                "edge_fraction": 0.03,
                "edge_statistic": EdgeStatistic.LOWER_QUANTILE.value,
            }
        return {}

    @property
    def background_settings(self) -> BackgroundSettings:
        return BackgroundSettings(self.background_method, self._background_parameters)

    def _background_description(self) -> str:
        if self.background_method is BackgroundMethod.CUSTOM:
            values = self._background_parameters.get("values", [])
            return f"Custom background · {len(values) if isinstance(values, list) else 0} aligned values"
        formatted_values: list[str] = []
        for key, value in self._background_parameters.items():
            if value is None:
                formatted_values.append(f"{key}=Auto")
            elif isinstance(value, float):
                formatted_values.append(f"{key}={value:g}")
            else:
                formatted_values.append(f"{key}={value}")
        values = ", ".join(formatted_values)
        return f"{self.background_method.value.title()} background · {values or 'default settings'}"

    def _background_diagnostic_summary(self) -> str:
        diagnostics = self.background_diagnostics
        if diagnostics is None:
            return "No background quality diagnostics available."
        status = "CAUTION: review background above signal." if diagnostics.is_cautionary else "Quality check: no material over-background."
        return (
            f"{status} Anchors {diagnostics.left_anchor:.4g} → {diagnostics.right_anchor:.4g}; "
            f"minimum net {diagnostics.minimum_net_intensity:.4g}."
        )

    def show_background_diagnostics(self) -> None:
        """Show independent, advisory quality checks for the active curve."""

        if self.spectrum is None or self.background is None:
            self.show_error("No background to inspect", "Load a spectrum and calculate a background first.")
            return
        diagnostics = self.background_diagnostics or diagnose_background(self.spectrum, self.background)
        review = (
            "Review the endpoints, model parameters, and fit region before using this background for quantification."
            if diagnostics.is_cautionary
            else "The background does not materially exceed the measured trace using the current noise estimate."
        )
        QMessageBox.information(
            self,
            "Background diagnostics",
            f"{self._background_description()}\n\n"
            f"Left anchor: {diagnostics.left_anchor:.7g}\n"
            f"Right anchor: {diagnostics.right_anchor:.7g}\n"
            f"Background above measured + 2σ: {diagnostics.fraction_above_signal:.1%}\n"
            f"Minimum net intensity: {diagnostics.minimum_net_intensity:.7g}\n"
            f"Robust noise estimate: {diagnostics.robust_net_noise:.7g}\n\n"
            f"{review}",
        )

    def on_background_method_changed(self, _index: int = 0) -> None:
        self._background_parameters = self._default_background_parameters(self.background_method)
        self._invalidate_active_consensus("Background changed; rerun stability consensus for an updated Trust Report.")
        self._invalidate_active_model_comparison("Background changed; rebuild the model comparison.")
        self._invalidate_global_series("A selected background changed; rerun global fitting.")
        self.recalculate_background()
        self._store_active_workspace_fit_state()

    def configure_background(self) -> None:
        """Prompt for explicit, serializable parameters for the selected model."""

        method = self.background_method
        options = dict(self._background_parameters)
        if method is BackgroundMethod.LINEAR:
            edge, accepted = QInputDialog.getDouble(
                self, "Linear background", "Edge fraction:", float(options.get("edge_fraction", 0.03)), 0.001, 0.5, 4
            )
            if not accepted:
                return
            statistic = self._choose_edge_statistic(options, "Linear background")
            if statistic is None:
                return
            left_text, accepted = QInputDialog.getText(
                self, "Linear background", "Left anchor level (Auto or numeric):", text=str(options.get("left_level", "Auto"))
            )
            if not accepted:
                return
            right_text, accepted = QInputDialog.getText(
                self, "Linear background", "Right anchor level (Auto or numeric):", text=str(options.get("right_level", "Auto"))
            )
            if not accepted:
                return
            options = {"edge_fraction": edge, "edge_statistic": statistic.value}
            for key, text in (("left_level", left_text), ("right_level", right_text)):
                if text.strip().lower() != "auto":
                    try:
                        options[key] = float(text)
                    except ValueError:
                        self.show_error("Invalid linear anchor", f"{key.replace('_', ' ').title()} must be Auto or a number.")
                        return
        elif method is BackgroundMethod.SHIRLEY:
            edge, accepted = QInputDialog.getDouble(
                self, "Shirley background", "Edge fraction:", float(options.get("edge_fraction", 0.03)), 0.001, 0.5, 4
            )
            if not accepted:
                return
            statistic = self._choose_edge_statistic(options, "Shirley background")
            if statistic is None:
                return
            iterations, accepted = QInputDialog.getInt(
                self, "Shirley background", "Maximum iterations:", int(options.get("max_iterations", 200)), 1, 100_000
            )
            if not accepted:
                return
            tolerance, accepted = QInputDialog.getDouble(
                self, "Shirley background", "Convergence tolerance:", float(options.get("tolerance", 1e-7)), 1e-12, 1.0, 10
            )
            if not accepted:
                return
            options = {
                "edge_fraction": edge,
                "edge_statistic": statistic.value,
                "max_iterations": iterations,
                "tolerance": tolerance,
            }
        elif method is BackgroundMethod.TOUGAARD:
            labels = ("U2 — B·T/(C+T²)²", "U3 — U2 plus D·T²")
            current_variant = 1 if str(options.get("variant")) == TougaardVariant.U3.value else 0
            selected, accepted = QInputDialog.getItem(self, "Tougaard background", "Kernel:", labels, current_variant, False)
            if not accepted:
                return
            variant = TougaardVariant.U3.value if selected.startswith("U3") else TougaardVariant.U2.value
            b, accepted = QInputDialog.getDouble(self, "Tougaard background", "B:", float(options.get("b", 2866.0)), 0.000001, 1e9, 6)
            if not accepted:
                return
            c, accepted = QInputDialog.getDouble(self, "Tougaard background", "C:", float(options.get("c", 1643.0)), 0.000001, 1e12, 6)
            if not accepted:
                return
            d, accepted = QInputDialog.getDouble(self, "Tougaard background", "D (U3 only):", float(options.get("d", 0.0)), 0.0, 1e12, 6)
            if not accepted:
                return
            edge, accepted = QInputDialog.getDouble(
                self, "Tougaard background", "Edge fraction:", float(options.get("edge_fraction", 0.03)), 0.001, 0.5, 4
            )
            if not accepted:
                return
            statistic = self._choose_edge_statistic(options, "Tougaard background")
            if statistic is None:
                return
            scale_default = "Auto" if options.get("scale") is None else str(options.get("scale"))
            scale_text, accepted = QInputDialog.getText(
                self, "Tougaard background", "Scale (Auto or numeric):", text=scale_default
            )
            if not accepted:
                return
            left_text, accepted = QInputDialog.getText(
                self, "Tougaard background", "Left anchor level (Auto or numeric):", text=str(options.get("left_level", "Auto"))
            )
            if not accepted:
                return
            right_text, accepted = QInputDialog.getText(
                self, "Tougaard background", "Right anchor level (Auto or numeric):", text=str(options.get("right_level", "Auto"))
            )
            if not accepted:
                return
            try:
                scale = self._auto_or_number(scale_text)
                left = self._auto_or_number(left_text)
                right = self._auto_or_number(right_text)
            except ValueError:
                self.show_error("Invalid Tougaard constraint", "Scale and endpoint levels must be Auto or a number.")
                return
            options = {
                "variant": variant,
                "b": b,
                "c": c,
                "d": d,
                "scale": scale,
                "edge_fraction": edge,
                "edge_statistic": statistic.value,
            }
            if left is not None:
                options["left_level"] = left
            if right is not None:
                options["right_level"] = right
        else:
            if self.spectrum is None:
                return
            custom = self._load_secondary_spectrum()
            if custom is None:
                return
            if custom.size != self.spectrum.size or not np.allclose(custom.energy, self.spectrum.energy, rtol=0.0, atol=1e-9):
                self.show_error("Custom background mismatch", "The imported background trace must use exactly the same energy axis and point count as the active spectrum.")
                return
            options = {"values": custom.intensity.tolist()}
        self._background_parameters = options
        self._invalidate_active_consensus("Background parameters changed; rerun stability consensus for an updated Trust Report.")
        self._invalidate_active_model_comparison("Background parameters changed; rebuild the model comparison.")
        self._invalidate_global_series("Selected background parameters changed; rerun global fitting.")
        self.recalculate_background()
        self._store_active_workspace_fit_state()

    @staticmethod
    def _auto_or_number(text: str) -> float | None:
        if text.strip().lower() == "auto":
            return None
        return float(text)

    @staticmethod
    def _choose_edge_statistic(options: dict[str, object], title: str) -> EdgeStatistic | None:
        labels = ("Peak-safe lower quantile", "Median")
        current = 1 if options.get("edge_statistic") == EdgeStatistic.MEDIAN.value else 0
        selected, accepted = QInputDialog.getItem(None, title, "Automatic edge anchor:", labels, current, False)
        if not accepted:
            return None
        return EdgeStatistic.MEDIAN if selected == "Median" else EdgeStatistic.LOWER_QUANTILE
