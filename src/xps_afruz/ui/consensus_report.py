"""Presentation-only dialog for retained stability-consensus evidence."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..fitting.stability_consensus import StabilityConsensusResult

TableMenuInstaller = Callable[[QTableWidget], None]


def open_consensus_report(
    parent: QWidget,
    result: StabilityConsensusResult,
    *,
    install_table_menu: TableMenuInstaller,
) -> None:
    """Open the copyable component table and graphical Trust Map."""

    dialog = QDialog(parent)
    dialog.setWindowTitle(f"Stability Trust Report — {result.spectrum_label}")
    dialog.resize(1_180, 760)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)

    title = QLabel("STABILITY-CONSENSUS TRUST REPORT")
    title.setObjectName("sectionTitle")
    layout.addWidget(title)
    count_support = " · ".join(
        f"K={count}: consensus {probability:.1%}, "
        f"BIC {result.bic_peak_count_probabilities[count]:.1%}, "
        f"AICc {result.aicc_peak_count_probabilities[count]:.1%}"
        for count, probability in sorted(result.peak_count_probabilities.items())
    )
    summary = QLabel(
        f"{result.successful_runs} valid model fits; {result.failed_runs} rejected/failed · "
        f"{result.paired_model_scenarios}/{result.requested_runs} complete paired scenarios · "
        f"requested K={result.requested_peak_count}<br>"
        f"Paired peak-count support (mean normalized BIC/AICc weights): {count_support}<br>"
        f"Backgrounds: {', '.join(result.background_models)}<br>"
        f"Profiles: {', '.join(result.profiles)} · {result.requested_runs} scenarios · "
        f"{result.candidate_trials_per_run} starts/model · bootstrap ×{result.bootstrap_strength:g} · "
        f"calibration σ ×{result.calibration_sigma_scale:g} · seed {result.random_seed}"
    )
    summary.setObjectName("fitSummary")
    summary.setWordWrap(True)
    layout.addWidget(summary)

    columns = (
        "Component",
        "Survival",
        "Model support",
        "Center ± σ (eV)",
        "FWHM ± σ (eV)",
        "Area CV",
        "Background",
        "Calibration",
        "Boundary",
        "Collapse",
        "Max |corr|",
        "Trust",
        "Identifiability",
    )
    table = QTableWidget(len(result.components), len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    table.setMinimumHeight(245)
    for row, component in enumerate(result.components):
        values = (
            component.name,
            f"{component.survival_probability:.1%}",
            f"{component.model_support_probability:.1%}",
            f"{component.center_mean_eV:.4f} ± {component.center_std_eV:.4f}",
            f"{component.fwhm_mean_eV:.4f} ± {component.fwhm_std_eV:.4f}",
            f"{component.area_cv:.1%}",
            component.background_sensitivity_label,
            f"{component.calibration_sensitivity_label} ({component.calibration_contribution_eV:.4g} eV)",
            f"{component.boundary_hit_frequency:.1%}",
            f"{component.collapse_frequency:.1%}",
            f"{component.max_parameter_correlation:.3f}",
            f"{component.trust_score:.1f}%",
            component.identifiability,
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column > 0:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row, column, item)
    install_table_menu(table)
    layout.addWidget(table)

    trust_plot = pg.PlotWidget()
    trust_plot.setBackground("#141718")
    trust_plot.showGrid(x=True, y=True, alpha=0.12)
    trust_plot.setLabel("left", "STABILITY TRUST SCORE", units="%", color="#cdd3d1")
    trust_plot.setLabel("bottom", "MEAN COMPONENT CENTER", units="eV", color="#cdd3d1")
    trust_plot.setYRange(0.0, 100.0, padding=0.03)
    trust_plot.addLine(y=80.0, pen=pg.mkPen("#6dd6c0", width=1.0, style=Qt.PenStyle.DashLine))
    trust_plot.addLine(y=55.0, pen=pg.mkPen("#d7aa62", width=1.0, style=Qt.PenStyle.DashLine))
    for component in result.components:
        if not np.isfinite(component.center_mean_eV):
            continue
        colour = (
            "#6dd6c0"
            if component.identifiability == "STRONG NUMERICAL SUPPORT"
            else "#d7aa62"
            if component.identifiability == "PROVISIONAL"
            else "#ef786d"
        )
        trust_plot.plot(
            [component.center_mean_eV],
            [component.trust_score],
            pen=None,
            symbol="o",
            symbolSize=13,
            symbolBrush=pg.mkBrush(colour),
            symbolPen=pg.mkPen("#f3f2ed", width=1.0),
        )
        trust_plot.addItem(
            pg.ErrorBarItem(
                x=np.asarray([component.center_mean_eV]),
                y=np.asarray([component.trust_score]),
                left=np.asarray([component.center_std_eV]),
                right=np.asarray([component.center_std_eV]),
                beam=0.8,
                pen=pg.mkPen(colour, width=1.3),
            )
        )
    layout.addWidget(trust_plot, stretch=1)

    note = QLabel(result.numerical_note)
    note.setObjectName("surveyNote")
    note.setWordWrap(True)
    layout.addWidget(note)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(dialog.accept)
    layout.addWidget(buttons)
    dialog.exec()
