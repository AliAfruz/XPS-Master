"""Interactive dashboard for retained global multi-spectrum fits."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.spectrum import Spectrum
from ..fitting.global_series import GlobalSeriesResult

COLOURS = ("#e9a857", "#e97970", "#b993df", "#5dc8b0", "#e6c35e", "#77a8e8", "#df86bc")


def open_global_series_report(
    parent: QWidget,
    spectra: Mapping[int, Spectrum],
    result: GlobalSeriesResult,
    *,
    on_apply: Callable[[], None],
    install_table_menu: Callable[[QTableWidget], None] | None = None,
) -> None:
    """Display global-vs-independent evidence, fits, residuals, and trends."""

    dialog = QDialog(parent)
    dialog.setWindowTitle("Global Multi-Spectrum Fit Dashboard")
    dialog.resize(1_480, 900)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(10)

    title = QLabel("GLOBAL MULTI-SPECTRUM FIT")
    title.setObjectName("sectionTitle")
    layout.addWidget(title)
    summary = QLabel(
        f"<b>{result.preference}</b><br>"
        f"{len(result.spectrum_fits)} spectra · {len(result.component_trends)} matched components · "
        f"global parameters {result.global_parameter_count} vs independent {result.independent_parameter_count}<br>"
        f"ΔAICc (independent − global) {result.delta_aicc:+.3f} · "
        f"ΔBIC {result.delta_bic:+.3f} · global reduced objective {result.reduced_global_objective:.5g}"
    )
    summary.setObjectName("fitSummary")
    summary.setWordWrap(True)
    layout.addWidget(summary)

    spectrum_table = QTableWidget(len(result.spectrum_fits), 7)
    spectrum_table.setHorizontalHeaderLabels(
        ("Workspace", "Spectrum", "Energy shift ± σ (eV)", "Global RMS", "Independent RMS", "Global objective", "Converged")
    )
    spectrum_table.verticalHeader().setVisible(False)
    spectrum_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    spectrum_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    spectrum_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    spectrum_table.setAlternatingRowColors(True)
    spectrum_table.setMaximumHeight(190)
    for row, item in enumerate(result.spectrum_fits):
        shift_error = "—" if item.energy_shift_stderr_eV != item.energy_shift_stderr_eV else f"{item.energy_shift_stderr_eV:.4g}"
        values = (
            item.workspace_index + 1,
            item.spectrum_label,
            f"{item.energy_shift_eV:+.5f} ± {shift_error}",
            f"{item.global_fit.residual_rms:.5g}",
            f"{item.independent_fit.residual_rms:.5g}",
            f"{item.global_fit.chi_square:.5g}",
            "Yes" if item.global_fit.success else "No",
        )
        for column, value in enumerate(values):
            table_item = QTableWidgetItem(str(value))
            if column != 1:
                table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            spectrum_table.setItem(row, column, table_item)
    spectrum_table.resizeColumnsToContents()
    spectrum_table.horizontalHeader().setStretchLastSection(True)
    if install_table_menu is not None:
        install_table_menu(spectrum_table)
    layout.addWidget(spectrum_table)

    lower = QSplitter(Qt.Orientation.Horizontal)
    lower.setChildrenCollapsible(False)
    layout.addWidget(lower, stretch=1)

    fit_panel = QFrame()
    fit_panel.setObjectName("glassPanel")
    fit_layout = QVBoxLayout(fit_panel)
    selector_row = QHBoxLayout()
    selector_row.addWidget(QLabel("Inspect spectrum"))
    selector = QComboBox()
    for item in result.spectrum_fits:
        selector.addItem(f"{item.workspace_index + 1}. {item.spectrum_label}", item.workspace_index)
    selector_row.addWidget(selector, stretch=1)
    fit_layout.addLayout(selector_row)
    fit_plot = pg.PlotWidget()
    residual_plot = pg.PlotWidget()
    for plot in (fit_plot, residual_plot):
        plot.setBackground("#141718")
        plot.showGrid(x=True, y=True, alpha=0.12)
    fit_plot.setLabel("left", "INTENSITY")
    residual_plot.setLabel("left", "FIT − MEASURED")
    residual_plot.setLabel("bottom", "ENERGY", units="eV")
    residual_plot.setXLink(fit_plot)
    plot_menu_installer = getattr(parent, "_install_plot_copy_menu", None)
    if callable(plot_menu_installer):
        plot_menu_installer(fit_plot)
        plot_menu_installer(residual_plot)
    fit_layout.addWidget(fit_plot, stretch=3)
    fit_layout.addWidget(residual_plot, stretch=1)
    lower.addWidget(fit_panel)

    trend_panel = QFrame()
    trend_panel.setObjectName("glassPanel")
    trend_layout = QVBoxLayout(trend_panel)
    trend_title = QLabel("COMPONENT AREA TRENDS")
    trend_title.setObjectName("sectionTitle")
    trend_layout.addWidget(trend_title)
    trend_plot = pg.PlotWidget()
    trend_plot.setBackground("#141718")
    trend_plot.showGrid(x=True, y=True, alpha=0.12)
    trend_plot.setLabel("left", "FITTED AREA")
    trend_plot.setLabel("bottom", "SERIES POSITION")
    trend_plot.addLegend(offset=(10, 10))
    series_x = list(range(1, len(result.spectrum_fits) + 1))
    for index, trend in enumerate(result.component_trends):
        colour = COLOURS[index % len(COLOURS)]
        trend_plot.plot(
            series_x,
            trend.areas,
            pen=pg.mkPen(colour, width=2.0),
            symbol="o",
            symbolBrush=colour,
            symbolSize=7,
            name=trend.name,
        )
    trend_layout.addWidget(trend_plot, stretch=1)

    component_table = QTableWidget(len(result.component_trends), 6)
    component_table.setHorizontalHeaderLabels(
        ("Component", "Profile", "Reference center ± σ", "Mean FWHM", "Area min", "Area max")
    )
    component_table.verticalHeader().setVisible(False)
    component_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    component_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    component_table.setAlternatingRowColors(True)
    for row, trend in enumerate(result.component_trends):
        center_error = "—" if trend.reference_center_stderr_eV != trend.reference_center_stderr_eV else f"{trend.reference_center_stderr_eV:.4g}"
        values = (
            trend.name,
            trend.profile,
            f"{trend.reference_center_eV:.5f} ± {center_error}",
            f"{sum(trend.fwhm_eV) / len(trend.fwhm_eV):.5g}",
            f"{min(trend.areas):.5g}",
            f"{max(trend.areas):.5g}",
        )
        for column, value in enumerate(values):
            component_table.setItem(row, column, QTableWidgetItem(str(value)))
    component_table.resizeColumnsToContents()
    component_table.horizontalHeader().setStretchLastSection(True)
    if install_table_menu is not None:
        install_table_menu(component_table)
    trend_layout.addWidget(component_table)
    lower.addWidget(trend_panel)
    lower.setSizes((850, 570))

    def render(row: int) -> None:
        item = result.spectrum_fits[max(0, min(row, len(result.spectrum_fits) - 1))]
        spectrum = spectra.get(item.workspace_index)
        fit_plot.clear()
        residual_plot.clear()
        if spectrum is None:
            return
        fit_plot.addLegend(offset=(10, 10))
        fit_plot.setLabel("bottom", f"{spectrum.energy_scale.value.upper()} ENERGY", units=spectrum.energy_unit)
        fit_plot.plot(spectrum.energy, spectrum.intensity, pen=pg.mkPen("#eef2ef", width=1.2), name="Measured")
        fit_plot.plot(spectrum.energy, item.global_fit.background, pen=pg.mkPen("#d4a45b", width=1.1), name="Background")
        for component_index, (name, curve) in enumerate(item.global_fit.components.items()):
            colour = QColor(COLOURS[component_index % len(COLOURS)])
            lower_curve = fit_plot.plot(spectrum.energy, item.global_fit.background, pen=pg.mkPen(0, 0, 0, 0))
            upper_curve = fit_plot.plot(
                spectrum.energy,
                curve + item.global_fit.background,
                pen=pg.mkPen(colour, width=1.0),
                name=name,
            )
            fit_plot.addItem(
                pg.FillBetweenItem(
                    upper_curve,
                    lower_curve,
                    brush=pg.mkBrush(colour.red(), colour.green(), colour.blue(), 48),
                )
            )
        fit_plot.plot(spectrum.energy, item.independent_fit.fitted_intensity, pen=pg.mkPen("#86a4aa", width=1.0, style=Qt.PenStyle.DashLine), name="Independent")
        fit_plot.plot(spectrum.energy, item.global_fit.fitted_intensity, pen=pg.mkPen("#fff5dc", width=2.0), name="Global")
        residual_plot.plot(spectrum.energy, item.independent_fit.residual, pen=pg.mkPen("#86a4aa", width=0.9, style=Qt.PenStyle.DashLine))
        residual_plot.plot(spectrum.energy, item.global_fit.residual, pen=pg.mkPen("#e97970", width=1.15))
        residual_plot.addLine(y=0.0, pen=pg.mkPen("#8a9697", width=0.8, style=Qt.PenStyle.DashLine))

    selector.currentIndexChanged.connect(render)
    spectrum_table.currentCellChanged.connect(lambda row, _column, _old_row, _old_column: selector.setCurrentIndex(max(row, 0)))
    spectrum_table.selectRow(0)
    render(0)

    note = QLabel(result.numerical_note)
    note.setObjectName("surveyNote")
    note.setWordWrap(True)
    layout.addWidget(note)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    apply_button = buttons.addButton("Apply global fits to workspace", QDialogButtonBox.ButtonRole.AcceptRole)
    apply_button.setObjectName("primaryButton")

    def apply_result() -> None:
        on_apply()
        dialog.accept()

    apply_button.clicked.connect(apply_result)
    buttons.rejected.connect(dialog.reject)
    buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()
