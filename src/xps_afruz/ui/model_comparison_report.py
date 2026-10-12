"""Interactive dashboard for retained physics-constrained model alternatives."""

from __future__ import annotations

from collections.abc import Callable

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.spectrum import Spectrum
from ..fitting.model_comparison import ModelCandidateEvaluation, ModelComparisonResult

COLOURS = ("#e9a857", "#e97970", "#b993df", "#5dc8b0", "#e6c35e", "#77a8e8", "#df86bc")
HEADERS = (
    "Rank",
    "Candidate",
    "K",
    "Background",
    "Profiles",
    "AICc",
    "BIC",
    "Support",
    "Residual ρ₁",
    "Boundary",
    "Max corr.",
    "Assessment",
)


def open_model_comparison_report(
    parent: QWidget,
    spectrum: Spectrum,
    result: ModelComparisonResult,
    *,
    on_accept: Callable[[str], None],
    install_table_menu: Callable[[QTableWidget], None] | None = None,
) -> None:
    """Show ranked evidence while keeping acceptance an explicit user action."""

    dialog = QDialog(parent)
    dialog.setWindowTitle("Physics-Constrained Model Comparison")
    dialog.resize(1_500, 840)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(11)

    title = QLabel("PHYSICS-CONSTRAINED MODEL COMPARISON")
    title.setObjectName("sectionTitle")
    layout.addWidget(title)
    recommended = result.recommended
    summary = QLabel(
        f"Recommended numerical candidate: <b>{recommended.candidate.label}</b> · "
        f"support {recommended.model_support:.1%} · {recommended.identifiability}. "
        "Selection remains an analyst decision."
    )
    summary.setObjectName("glassPanel")
    summary.setWordWrap(True)
    summary.setMargin(10)
    layout.addWidget(summary)

    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.setChildrenCollapsible(False)
    layout.addWidget(splitter, stretch=1)

    left = QFrame()
    left.setObjectName("glassPanel")
    left_layout = QVBoxLayout(left)
    left_layout.setContentsMargins(10, 10, 10, 10)
    table = QTableWidget(len(result.candidates), len(HEADERS))
    table.setHorizontalHeaderLabels(HEADERS)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    for row, evaluation in enumerate(result.candidates):
        profiles = ", ".join(sorted({component.profile.display_name for component in evaluation.candidate.components}))
        values = (
            evaluation.rank,
            evaluation.candidate.label,
            evaluation.component_count,
            evaluation.candidate.background.method.value.title(),
            profiles,
            f"{evaluation.aicc:.5g}",
            f"{evaluation.bic:.5g}",
            f"{evaluation.model_support:.1%}",
            f"{evaluation.residual_autocorrelation:+.3f}",
            f"{evaluation.boundary_hit_fraction:.0%}",
            f"{evaluation.max_parameter_correlation:.3f}",
            ("Excluded · " if not evaluation.eligible else "") + evaluation.identifiability,
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if column == 0:
                item.setData(Qt.ItemDataRole.UserRole, evaluation.candidate.identifier)
            if not evaluation.eligible:
                item.setForeground(QColor("#e97970"))
            table.setItem(row, column, item)
    table.resizeColumnsToContents()
    table.horizontalHeader().setStretchLastSection(True)
    table.setMinimumWidth(720)
    if install_table_menu is not None:
        install_table_menu(table)
    left_layout.addWidget(table, stretch=1)

    detail = QLabel()
    detail.setObjectName("surveyNote")
    detail.setWordWrap(True)
    detail.setMinimumHeight(100)
    left_layout.addWidget(detail)
    splitter.addWidget(left)

    right = QFrame()
    right.setObjectName("glassPanel")
    right_layout = QVBoxLayout(right)
    right_layout.setContentsMargins(10, 10, 10, 10)
    right_layout.setSpacing(8)
    spectrum_plot = pg.PlotWidget()
    residual_plot = pg.PlotWidget()
    spectrum_plot.setBackground("#141718")
    residual_plot.setBackground("#141718")
    spectrum_plot.showGrid(x=True, y=True, alpha=0.12)
    residual_plot.showGrid(x=True, y=True, alpha=0.12)
    spectrum_plot.setLabel("left", "INTENSITY", units=spectrum.intensity_unit)
    residual_plot.setLabel("left", "RESIDUAL", units=spectrum.intensity_unit)
    residual_plot.setLabel("bottom", f"{spectrum.energy_scale.value.upper()} ENERGY", units=spectrum.energy_unit)
    residual_plot.setXLink(spectrum_plot)
    plot_menu_installer = getattr(parent, "_install_plot_copy_menu", None)
    if callable(plot_menu_installer):
        plot_menu_installer(spectrum_plot)
        plot_menu_installer(residual_plot)
    right_layout.addWidget(spectrum_plot, stretch=3)
    right_layout.addWidget(residual_plot, stretch=2)
    splitter.addWidget(right)
    splitter.setSizes((760, 700))

    def evaluation_for_row(row: int) -> ModelCandidateEvaluation:
        item = table.item(row, 0)
        identifier = str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else result.recommended_candidate_id
        return result.candidate(identifier)

    def render(row: int) -> None:
        selected = evaluation_for_row(row)
        spectrum_plot.clear()
        residual_plot.clear()
        spectrum_plot.addLegend(offset=(10, 10))
        spectrum_plot.plot(spectrum.energy, spectrum.intensity, pen=pg.mkPen("#eef2ef", width=1.25), name="Measured")
        for index, evaluation in enumerate(result.candidates):
            colour = QColor(COLOURS[index % len(COLOURS)])
            alpha = 210 if evaluation is selected else 55
            width = 2.2 if evaluation is selected else 0.8
            spectrum_plot.plot(
                spectrum.energy,
                evaluation.fit_result.fitted_intensity,
                pen=pg.mkPen(colour.red(), colour.green(), colour.blue(), alpha, width=width),
                name=evaluation.candidate.label if evaluation is selected else None,
            )
            residual_plot.plot(
                spectrum.energy,
                evaluation.fit_result.residual,
                pen=pg.mkPen(colour.red(), colour.green(), colour.blue(), alpha, width=width),
            )
        spectrum_plot.plot(
            spectrum.energy,
            selected.fit_result.background,
            pen=pg.mkPen("#d4a45b", width=1.2),
            name="Selected background",
        )
        component_colours = ("#f2a65a", "#ea7d70", "#bb91df", "#62cbb5", "#e7c15e", "#7fa9e5")
        for index, (name, curve) in enumerate(selected.fit_result.components.items()):
            colour = QColor(component_colours[index % len(component_colours)])
            lower = spectrum_plot.plot(spectrum.energy, selected.fit_result.background, pen=pg.mkPen(0, 0, 0, 0))
            upper = spectrum_plot.plot(
                spectrum.energy,
                curve + selected.fit_result.background,
                pen=pg.mkPen(colour, width=1.0),
                name=name,
            )
            spectrum_plot.addItem(
                pg.FillBetweenItem(
                    upper,
                    lower,
                    brush=pg.mkBrush(colour.red(), colour.green(), colour.blue(), 52),
                )
            )
        residual_plot.addLine(y=0.0, pen=pg.mkPen("#8a9697", width=0.8, style=Qt.PenStyle.DashLine))
        warnings = "<br>".join(f"• {warning}" for warning in selected.warnings) or "No automatic warning flags."
        detail.setText(
            f"<b>{selected.candidate.label}</b><br>{selected.candidate.origin}<br>"
            f"Durbin–Watson {selected.durbin_watson:.3f} · background above signal "
            f"{selected.background_diagnostics.fraction_above_signal:.1%}<br>{warnings}"
        )

    table.currentCellChanged.connect(lambda row, _column, _old_row, _old_column: render(max(row, 0)))
    table.selectRow(0)
    render(0)

    note = QLabel(result.numerical_note)
    note.setObjectName("surveyNote")
    note.setWordWrap(True)
    layout.addWidget(note)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    accept_button = QPushButton("Accept selected candidate")
    accept_button.setObjectName("primaryButton")

    def accept_selected() -> None:
        row = max(table.currentRow(), 0)
        if not evaluation_for_row(row).eligible:
            return
        on_accept(evaluation_for_row(row).candidate.identifier)
        dialog.accept()

    accept_button.clicked.connect(accept_selected)
    accept_button.setEnabled(evaluation_for_row(max(table.currentRow(), 0)).eligible)
    table.currentCellChanged.connect(
        lambda row, _column, _old_row, _old_column: accept_button.setEnabled(evaluation_for_row(max(row, 0)).eligible)
    )
    buttons.addButton(accept_button, QDialogButtonBox.ButtonRole.AcceptRole)
    buttons.rejected.connect(dialog.reject)
    buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()
