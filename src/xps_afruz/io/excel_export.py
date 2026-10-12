"""Publication-ready Excel export for complete XPS workspaces.

The workbook is intentionally data-first: every native chart points to the
flat numerical table on the same worksheet, derived columns use visible Excel
formulas, and scientific provenance follows the table instead of being hidden
inside drawing objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import xlsxwriter
from xlsxwriter.exceptions import XlsxWriterException
from xlsxwriter.utility import xl_col_to_name

from ..backgrounds import estimate_background
from ..fitting.model import FitResult
from ..fitting.global_series import GlobalSeriesResult
from .project_io import WorkspaceItemData


EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLUMNS = 16_384
COMPONENT_COLOURS = (
    "#E67E22",
    "#C94C4C",
    "#8E5AA9",
    "#2E9F8F",
    "#D4A017",
    "#5470C6",
    "#C45A8D",
    "#4E8A5B",
)


class ExcelExportError(RuntimeError):
    """Raised when Excel cannot finalize or replace the requested workbook."""


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size > 20:
            return f"[{value.size} aligned numerical values; see exported data columns]"
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray) and value.size > 20:
        return f"{value.size} aligned numerical values (exported in the data table)"
    if isinstance(value, (Mapping, list, tuple, np.ndarray)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _safe_sheet_name(label: str, index: int, existing: set[str]) -> str:
    cleaned = re.sub(r"\s+", " ", re.sub(r"[\[\]:*?/\\]", " ", label)).strip().strip("'") or f"Spectrum {index}"
    base = f"{index:02d} {cleaned}"[:31].rstrip()
    candidate = base
    suffix = 2
    while candidate.casefold() in existing:
        marker = f" {suffix}"
        candidate = f"{base[: 31 - len(marker)].rstrip()}{marker}"
        suffix += 1
    existing.add(candidate.casefold())
    return candidate


def _aligned_fit_result(result: FitResult | None, size: int) -> FitResult | None:
    if result is None:
        return None
    curves = (
        result.background,
        result.fitted_intensity,
        result.residual,
        result.weighted_residual,
        *result.components.values(),
    )
    if any(np.asarray(curve).size != size or not np.all(np.isfinite(curve)) for curve in curves):
        raise ValueError("A retained fit contains non-finite or misaligned curves and cannot be exported.")
    return result


def _export_background(item: WorkspaceItemData, fit_result: FitResult | None) -> tuple[np.ndarray | None, str]:
    if fit_result is not None:
        return np.asarray(fit_result.background, dtype=float), "Retained fitted background"
    try:
        background = estimate_background(
            item.spectrum,
            item.background_method,
            **dict(item.background_parameters),
        )
    except (TypeError, ValueError) as error:
        return None, f"Background unavailable during export: {error}"
    return np.asarray(background, dtype=float), "Calculated from retained background settings"


def _formats(workbook: xlsxwriter.Workbook) -> dict[str, Any]:
    return {
        "title": workbook.add_format(
            {"font_name": "Aptos Display", "font_size": 18, "bold": True, "font_color": "#FFFFFF", "bg_color": "#25292A", "align": "left", "valign": "vcenter"}
        ),
        "subtitle": workbook.add_format(
            {"font_name": "Aptos", "font_size": 10, "font_color": "#E7C589", "bg_color": "#25292A", "align": "left", "valign": "vcenter"}
        ),
        "section": workbook.add_format(
            {"font_name": "Aptos", "font_size": 11, "bold": True, "font_color": "#FFFFFF", "bg_color": "#6A5130", "bottom": 1, "bottom_color": "#C3924C"}
        ),
        "header": workbook.add_format(
            {"font_name": "Aptos", "font_size": 9, "bold": True, "font_color": "#FFFFFF", "bg_color": "#3B4244", "bottom": 1, "bottom_color": "#C3924C", "text_wrap": True, "align": "center", "valign": "vcenter"}
        ),
        "label": workbook.add_format({"font_name": "Aptos", "font_size": 9, "bold": True, "font_color": "#554126", "bg_color": "#F2E8D8", "text_wrap": True, "valign": "vcenter"}),
        "text": workbook.add_format({"font_name": "Aptos", "font_size": 9, "font_color": "#222222", "valign": "top"}),
        "wrapped": workbook.add_format({"font_name": "Aptos", "font_size": 9, "font_color": "#222222", "text_wrap": True, "valign": "top"}),
        "number": workbook.add_format({"font_name": "Aptos", "font_size": 9, "num_format": "0.000000", "font_color": "#222222"}),
        "scientific": workbook.add_format({"font_name": "Aptos", "font_size": 9, "num_format": "0.000000E+00", "font_color": "#222222"}),
        "integer": workbook.add_format({"font_name": "Aptos", "font_size": 9, "num_format": "#,##0", "font_color": "#222222"}),
        "note": workbook.add_format({"font_name": "Aptos", "font_size": 9, "italic": True, "font_color": "#5E5E5E", "bg_color": "#F7F4EE", "text_wrap": True, "valign": "vcenter"}),
        "link": workbook.add_format({"font_name": "Aptos", "font_size": 9, "font_color": "#1F4E78", "underline": True}),
        "good": workbook.add_format({"font_name": "Aptos", "font_size": 9, "bold": True, "font_color": "#24613B", "bg_color": "#E7F2EA"}),
        "caution": workbook.add_format({"font_name": "Aptos", "font_size": 9, "bold": True, "font_color": "#8A451D", "bg_color": "#FBE9D8"}),
    }


def _write_scalar(worksheet: Any, row: int, column: int, value: Any, formats: Mapping[str, Any]) -> None:
    number = _finite_number(value)
    if number is not None and not isinstance(value, (str, bool)):
        number_format = formats["integer"] if isinstance(value, (int, np.integer)) else formats["number"]
        worksheet.write_number(row, column, number, number_format)
    elif value is None or isinstance(value, (float, np.floating)):
        worksheet.write_blank(row, column, None, formats["text"])
    else:
        worksheet.write_string(row, column, _display_value(value), formats["wrapped"])


def _write_key_value_block(
    worksheet: Any,
    start_row: int,
    start_column: int,
    entries: Sequence[tuple[str, Any]],
    formats: Mapping[str, Any],
) -> None:
    for offset, (label, value) in enumerate(entries):
        worksheet.write_string(start_row + offset, start_column, label, formats["label"])
        _write_scalar(worksheet, start_row + offset, start_column + 1, value, formats)


def _write_summary_sheet(
    workbook: xlsxwriter.Workbook,
    items: Sequence[WorkspaceItemData],
    sheet_names: Sequence[str],
    active_index: int,
    project_name: str,
    formats: Mapping[str, Any],
    global_series: GlobalSeriesResult | None = None,
) -> None:
    sheet = workbook.add_worksheet("Export Summary")
    sheet.hide_gridlines(2)
    sheet.set_tab_color("#C3924C")
    sheet.set_zoom(95)
    sheet.set_column("A:A", 17)
    sheet.set_column("B:B", 30)
    sheet.set_column("C:C", 34)
    sheet.set_column("D:F", 18)
    sheet.set_column("G:G", 16)
    sheet.set_column("H:H", 14)
    sheet.set_column("I:I", 20)
    sheet.set_column("J:K", 22)
    sheet.merge_range("A1:K2", "XPS-master — Complete Excel Export", formats["title"])
    sheet.merge_range("A3:K3", "Auditable spectra, fitted components, residuals, global-series evidence, model comparisons, Trust Reports, parameters, and provenance", formats["subtitle"])
    _write_key_value_block(
        sheet,
        4,
        0,
        (
            ("Project", project_name),
            ("Exported UTC", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
            ("Spectra", len(items)),
            ("Active spectrum", active_index + 1),
            (
                "Global series",
                "Not run"
                if global_series is None
                else f"{len(global_series.spectrum_fits)} spectra; {global_series.preference}",
            ),
        ),
        formats,
    )
    sheet.merge_range("D5:K8", "Measured intensity is the active spectrum displayed by the application. The original imported energy axis is preserved separately. Processing, calibration, global-series, candidate-model comparison, and stability-consensus evidence are retained explicitly; original pre-processing intensity is not reconstructed when it is not retained by the spectrum model.", formats["note"])

    headers = ("#", "Worksheet", "Spectrum label", "Energy scale", "Range (eV)", "Points", "Background", "Components", "Fit quality", "Stability consensus", "Model comparison")
    header_row = 10
    for column, header in enumerate(headers):
        sheet.write_string(header_row, column, header, formats["header"])
    for index, (item, sheet_name) in enumerate(zip(items, sheet_names, strict=True)):
        row = header_row + 1 + index
        fit = item.fit_result
        sheet.write_number(row, 0, index + 1, formats["integer"])
        sheet.write_url(row, 1, f"internal:'{sheet_name}'!A1", formats["link"], sheet_name)
        sheet.write_string(row, 2, item.spectrum.label, formats["text"])
        sheet.write_string(row, 3, item.spectrum.energy_scale.value.title(), formats["text"])
        sheet.write_string(row, 4, f"{item.spectrum.energy_range[0]:.4f}–{item.spectrum.energy_range[1]:.4f}", formats["text"])
        sheet.write_number(row, 5, item.spectrum.size, formats["integer"])
        sheet.write_string(row, 6, item.background_method.value.title(), formats["text"])
        sheet.write_number(row, 7, 0 if fit is None else len(fit.components), formats["integer"])
        if fit is None:
            sheet.write_string(row, 8, "Not fitted", formats["caution"])
        elif fit.has_calibrated_chi_square:
            sheet.write_string(row, 8, f"Reduced statistical chi-square {fit.reduced_statistical_chi_square:.5g}", formats["good"])
        else:
            sheet.write_string(row, 8, f"Residual RMS {fit.residual_rms:.5g}", formats["good"])
        consensus = item.consensus_result
        if consensus is None:
            sheet.write_string(row, 9, "Not run", formats["caution"])
        else:
            strong = sum(component.identifiability == "STRONG NUMERICAL SUPPORT" for component in consensus.components)
            sheet.write_string(
                row,
                9,
                f"K={consensus.preferred_peak_count}; {strong}/{len(consensus.components)} strong",
                formats["good"] if strong else formats["caution"],
            )
        comparison = item.model_comparison
        if comparison is None:
            sheet.write_string(row, 10, "Not run", formats["caution"])
        else:
            sheet.write_string(
                row,
                10,
                f"{len(comparison.candidates)} candidates; {comparison.recommended.candidate.label}",
                formats["good"],
            )
    sheet.autofilter(header_row, 0, header_row + len(items), len(headers) - 1)
    sheet.freeze_panes(header_row + 1, 2)

    note_row = header_row + len(items) + 3
    sheet.merge_range(note_row, 0, note_row, 10, "Scientific boundary", formats["section"])
    sheet.merge_range(
        note_row + 1,
        0,
        note_row + 3,
        10,
        "This export preserves the numerical models and evidence selected in XPS-master. A preferred AICc/BIC candidate, fitted component, survey match, or Stability Trust Score is not, by itself, a chemical-state confirmation. Review calibration, background, constraints, residual structure, line shape, companion lines, and instrument-specific sensitivity factors before reporting results.",
        formats["note"],
    )


def _write_global_series_sheet(
    workbook: xlsxwriter.Workbook,
    result: GlobalSeriesResult,
    items: Sequence[WorkspaceItemData],
    formats: Mapping[str, Any],
) -> None:
    """Write typed global-vs-independent evidence and component trends."""

    for series_item in result.spectrum_fits:
        if not 0 <= series_item.workspace_index < len(items):
            raise ValueError("A retained global-series result refers to a missing workspace spectrum.")
        spectrum = items[series_item.workspace_index].spectrum
        if series_item.spectrum_label != spectrum.label:
            raise ValueError("A retained global-series label no longer matches its workspace spectrum.")
        _aligned_fit_result(series_item.global_fit, spectrum.size)
        _aligned_fit_result(series_item.independent_fit, spectrum.size)

    sheet = workbook.add_worksheet("Global Series")
    sheet.hide_gridlines(2)
    sheet.set_tab_color("#8E5AA9")
    sheet.set_zoom(90)
    sheet.set_column("A:A", 13)
    sheet.set_column("B:B", 32)
    sheet.set_column("C:G", 18)
    sheet.set_column("H:Z", 16)
    sheet.merge_range("A1:J2", "Global Multi-Spectrum Fit", formats["title"])
    sheet.merge_range(
        "A3:J3",
        "Shared physical parameters, spectrum-specific areas and relative energy shifts, with independent-fit reference evidence",
        formats["subtitle"],
    )
    _write_key_value_block(
        sheet,
        4,
        0,
        (
            ("Spectra", len(result.spectrum_fits)),
            ("Components", len(result.component_trends)),
            ("Converged", result.success),
            ("Optimizer evaluations", result.optimizer_evaluations),
            ("Total samples", result.total_samples),
            ("Global parameters", result.global_parameter_count),
            ("Independent parameters", result.independent_parameter_count),
            ("Degrees of freedom", result.degrees_of_freedom),
        ),
        formats,
    )
    _write_key_value_block(
        sheet,
        4,
        3,
        (
            ("Global objective", result.global_objective),
            ("Reduced global objective", result.reduced_global_objective),
            ("Global AICc", result.global_aicc),
            ("Independent AICc", result.independent_aicc),
            ("Delta AICc (independent - global)", result.delta_aicc),
            ("Global BIC", result.global_bic),
            ("Independent BIC", result.independent_bic),
            ("Delta BIC (independent - global)", result.delta_bic),
        ),
        formats,
    )
    _write_key_value_block(
        sheet,
        4,
        6,
        (
            ("Weighting", result.settings.weighting),
            ("Count scale", result.settings.count_scale if result.settings.weighting == "poisson" else None),
            ("Share centers", result.settings.share_centers),
            ("Share FWHM", result.settings.share_fwhm),
            ("Share line shape", result.settings.share_eta and result.settings.share_asymmetry),
            ("Allow relative shifts", result.settings.allow_energy_shifts),
            ("Maximum shift (eV)", result.settings.maximum_energy_shift_eV),
            ("Maximum evaluations", result.settings.max_nfev),
        ),
        formats,
    )
    sheet.merge_range("A13:J13", "Model comparison result", formats["section"])
    sheet.merge_range("A14:J14", result.preference, formats["good"] if result.favours_global_model else formats["caution"])

    header_row = 16
    headers = [
        "Workspace #",
        "Spectrum label",
        "Relative shift (eV)",
        "Shift SE (eV)",
        "Global residual RMS",
        "Independent residual RMS",
        "Global objective",
        *(f"{trend.name} area" for trend in result.component_trends),
    ]
    for column, header in enumerate(headers):
        sheet.write_string(header_row, column, header, formats["header"])
    for series_position, item in enumerate(result.spectrum_fits):
        row = header_row + 1 + series_position
        sheet.write_number(row, 0, item.workspace_index + 1, formats["integer"])
        sheet.write_string(row, 1, item.spectrum_label, formats["text"])
        _write_scalar(sheet, row, 2, item.energy_shift_eV, formats)
        _write_scalar(sheet, row, 3, item.energy_shift_stderr_eV, formats)
        _write_scalar(sheet, row, 4, item.global_fit.residual_rms, formats)
        _write_scalar(sheet, row, 5, item.independent_fit.residual_rms, formats)
        _write_scalar(sheet, row, 6, item.global_fit.chi_square, formats)
        for component_index, trend in enumerate(result.component_trends):
            _write_scalar(sheet, row, 7 + component_index, trend.areas[series_position], formats)
    sheet.autofilter(header_row, 0, header_row + len(result.spectrum_fits), len(headers) - 1)
    sheet.freeze_panes(header_row + 1, 2)

    chart = workbook.add_chart({"type": "line"})
    for component_index, trend in enumerate(result.component_trends):
        colour = COMPONENT_COLOURS[component_index % len(COMPONENT_COLOURS)]
        chart.add_series(
            {
                "name": trend.name,
                "categories": ["Global Series", header_row + 1, 1, header_row + len(result.spectrum_fits), 1],
                "values": [
                    "Global Series",
                    header_row + 1,
                    7 + component_index,
                    header_row + len(result.spectrum_fits),
                    7 + component_index,
                ],
                "line": {"color": colour, "width": 2.0},
                "marker": {"type": "circle", "size": 5, "border": {"color": colour}, "fill": {"color": colour}},
            }
        )
    chart.set_title({"name": "Component area across the selected series", "name_font": {"name": "Aptos", "size": 12}})
    chart.set_x_axis({"name": "Spectrum", "name_font": {"name": "Aptos", "size": 10}, "num_font": {"name": "Aptos", "size": 9}})
    chart.set_y_axis({"name": "Fitted area", "name_font": {"name": "Aptos", "size": 10}, "num_font": {"name": "Aptos", "size": 9}, "major_gridlines": {"visible": True, "line": {"color": "#E4E4E4"}}})
    chart.set_legend({"position": "top", "font": {"name": "Aptos", "size": 9}})
    chart.set_style(10)
    chart.set_size({"width": 880, "height": 330})
    chart_row = header_row + len(result.spectrum_fits) + 3
    sheet.insert_chart(chart_row, 0, chart)

    component_row = chart_row + 18
    sheet.merge_range(component_row, 0, component_row, 9, "Shared component parameters and uncertainty", formats["section"])
    component_headers = (
        "Component",
        "Profile",
        "Reference center (eV)",
        "Center SE (eV)",
        "Mean FWHM (eV)",
        "Minimum area",
        "Maximum area",
        "Fit eta",
        "Fit asymmetry",
        "Fit instrument width",
    )
    for column, header in enumerate(component_headers):
        sheet.write_string(component_row + 1, column, header, formats["header"])
    for offset, trend in enumerate(result.component_trends, start=2):
        row = component_row + offset
        sheet.write_string(row, 0, trend.name, formats["text"])
        sheet.write_string(row, 1, trend.profile, formats["text"])
        _write_scalar(sheet, row, 2, trend.reference_center_eV, formats)
        _write_scalar(sheet, row, 3, trend.reference_center_stderr_eV, formats)
        _write_scalar(sheet, row, 4, float(np.mean(trend.fwhm_eV)), formats)
        _write_scalar(sheet, row, 5, min(trend.areas), formats)
        _write_scalar(sheet, row, 6, max(trend.areas), formats)
        sheet.write_boolean(row, 7, trend.vary_eta, formats["text"])
        sheet.write_boolean(row, 8, trend.vary_asymmetry, formats["text"])
        sheet.write_boolean(row, 9, trend.vary_instrument_fwhm, formats["text"])

    note_row = component_row + len(result.component_trends) + 4
    sheet.merge_range(note_row, 0, note_row, 9, "Scientific boundary", formats["section"])
    sheet.merge_range(note_row + 1, 0, note_row + 3, 9, result.numerical_note, formats["note"])

    raw_row = note_row + 5
    required_rows = raw_row + sum(
        items[item.workspace_index].spectrum.size + 3 for item in result.spectrum_fits
    )
    if required_rows >= EXCEL_MAX_ROWS:
        raise ExcelExportError("Global-series curve data exceed Excel's worksheet row limit.")
    for series_item in result.spectrum_fits:
        spectrum = items[series_item.workspace_index].spectrum
        component_names = tuple(series_item.global_fit.components)
        data_headers = (
            "Energy (eV)",
            "Measured intensity",
            "Background",
            "Global fit envelope",
            "Global residual (fit - measured)",
            "Global weighted residual",
            "Independent fit envelope",
            "Independent residual (fit - measured)",
            *(f"Global component: {name} (net)" for name in component_names),
        )
        last_column = len(data_headers) - 1
        sheet.merge_range(
            raw_row,
            0,
            raw_row,
            last_column,
            f"Global-series curve data: workspace {series_item.workspace_index + 1}, {series_item.spectrum_label}",
            formats["section"],
        )
        for column, header in enumerate(data_headers):
            sheet.write_string(raw_row + 1, column, header, formats["header"])
        component_curves = tuple(series_item.global_fit.components[name] for name in component_names)
        for point in range(spectrum.size):
            row = raw_row + 2 + point
            values = (
                spectrum.energy[point],
                spectrum.intensity[point],
                series_item.global_fit.background[point],
                series_item.global_fit.fitted_intensity[point],
                series_item.global_fit.residual[point],
                series_item.global_fit.weighted_residual[point],
                series_item.independent_fit.fitted_intensity[point],
                series_item.independent_fit.residual[point],
                *(curve[point] for curve in component_curves),
            )
            for column, value in enumerate(values):
                _write_scalar(sheet, row, column, value, formats)
        raw_row += spectrum.size + 3
    sheet.set_landscape()
    sheet.fit_to_pages(1, 0)
    sheet.set_margins(0.3, 0.3, 0.5, 0.5)


def _write_fit_parameter_table(
    worksheet: Any,
    start_row: int,
    item: WorkspaceItemData,
    result: FitResult | None,
    formats: Mapping[str, Any],
) -> int:
    worksheet.merge_range(start_row, 0, start_row, 10, "Fitted component parameters", formats["section"])
    headers = ("Component", "Profile", "Center (eV)", "SE", "FWHM (eV)", "SE", "Area", "SE", "Eta", "Asymmetry", "Instrument FWHM")
    for column, header in enumerate(headers):
        worksheet.write_string(start_row + 1, column, header, formats["header"])
    row = start_row + 2
    for peak in item.peaks:
        name = str(peak.get("name", f"Peak {row - start_row - 1}"))
        values = result.parameter_values if result is not None else {}
        errors = result.standard_errors if result is not None else {}
        worksheet.write_string(row, 0, name, formats["text"])
        worksheet.write_string(row, 1, str(peak.get("profile", "")), formats["text"])
        numeric = (
            values.get(f"{name}.center", peak.get("center")),
            errors.get(f"{name}.center"),
            values.get(f"{name}.fwhm", peak.get("fwhm")),
            errors.get(f"{name}.fwhm"),
            values.get(f"{name}.area", peak.get("area")),
            errors.get(f"{name}.area"),
            values.get(f"{name}.eta", peak.get("eta")),
            values.get(f"{name}.asymmetry", peak.get("asymmetry")),
            values.get(f"{name}.instrument_fwhm", peak.get("instrument_fwhm")),
        )
        for column, value in enumerate(numeric, start=2):
            number = _finite_number(value)
            if number is None:
                worksheet.write_blank(row, column, None, formats["number"])
            else:
                worksheet.write_number(row, column, number, formats["number"])
        row += 1
    if row == start_row + 2:
        worksheet.merge_range(row, 0, row, 10, "No peak components were retained for this spectrum.", formats["note"])
        row += 1
    return row


def _write_ties_table(worksheet: Any, start_row: int, item: WorkspaceItemData, formats: Mapping[str, Any]) -> int:
    worksheet.merge_range(start_row, 0, start_row, 5, "Parameter constraints", formats["section"])
    headers = ("Target", "Source", "Scale", "Offset", "Equation", "Review")
    for column, header in enumerate(headers):
        worksheet.write_string(start_row + 1, column, header, formats["header"])
    row = start_row + 2
    for tie in item.ties:
        target = str(tie.get("target", ""))
        source = str(tie.get("source", ""))
        scale = _finite_number(tie.get("scale", 1.0)) or 0.0
        offset = _finite_number(tie.get("offset", 0.0)) or 0.0
        worksheet.write_string(row, 0, target, formats["text"])
        worksheet.write_string(row, 1, source, formats["text"])
        worksheet.write_number(row, 2, scale, formats["number"])
        worksheet.write_number(row, 3, offset, formats["number"])
        worksheet.write_string(row, 4, f"{target} = {scale:g} × {source} + {offset:g}", formats["wrapped"])
        worksheet.write_string(row, 5, "Analyst-defined physical/mathematical constraint", formats["wrapped"])
        worksheet.set_row(row, 30)
        row += 1
    if row == start_row + 2:
        worksheet.merge_range(row, 0, row, 5, "No parameter ties were retained.", formats["note"])
        row += 1
    return row


def _write_history_table(
    worksheet: Any,
    start_row: int,
    title: str,
    history: Sequence[Mapping[str, object]],
    formats: Mapping[str, Any],
) -> int:
    worksheet.merge_range(start_row, 0, start_row, 5, title, formats["section"])
    worksheet.write_string(start_row + 1, 0, "Step", formats["header"])
    worksheet.write_string(start_row + 1, 1, "Operation", formats["header"])
    worksheet.merge_range(start_row + 1, 2, start_row + 1, 5, "Parameters / provenance", formats["header"])
    row = start_row + 2
    for index, entry in enumerate(history, start=1):
        operation = str(entry.get("operation", entry.get("reference_label", "Recorded step")))
        details = {key: value for key, value in entry.items() if key not in {"operation", "reference_label"}}
        worksheet.write_number(row, 0, index, formats["integer"])
        worksheet.write_string(row, 1, operation, formats["text"])
        worksheet.merge_range(row, 2, row, 5, _display_value(details), formats["wrapped"])
        row += 1
    if not history:
        worksheet.merge_range(row, 0, row, 5, "No history entries were recorded.", formats["note"])
        row += 1
    return row


def _write_survey_table(worksheet: Any, start_row: int, item: WorkspaceItemData, formats: Mapping[str, Any]) -> int:
    worksheet.merge_range(start_row, 0, start_row, 8, "Survey-screening evidence", formats["section"])
    headers = ("Element", "Name", "Confidence", "Score", "Consistency", "Matched lines", "Trace recovered", "Overlaps", "Scientific status")
    for column, header in enumerate(headers):
        worksheet.write_string(start_row + 1, column, header, formats["header"])
    row = start_row + 2
    analysis = item.survey_analysis
    if analysis is not None:
        recovered = set(analysis.trace_recovered_elements)
        for candidate in analysis.candidates:
            lines = ", ".join(dict.fromkeys(match.reference.label for match in candidate.matches))
            values = (
                candidate.symbol,
                candidate.element,
                candidate.confidence_label,
                candidate.score,
                candidate.consistency_score,
                lines,
                "Yes" if candidate.symbol in recovered else "No",
                ", ".join(candidate.overlaps_with),
                "Screening candidate — analyst confirmation required",
            )
            for column, value in enumerate(values):
                _write_scalar(worksheet, row, column, value, formats)
            row += 1
    if row == start_row + 2:
        worksheet.merge_range(row, 0, row, 8, "No survey candidates were retained for this spectrum.", formats["note"])
        row += 1
    return row


def _write_consensus_table(worksheet: Any, start_row: int, item: WorkspaceItemData, formats: Mapping[str, Any]) -> int:
    """Write aggregate stability evidence and exact model-count criteria."""

    worksheet.merge_range(start_row, 0, start_row, 12, "Stability-Consensus Trust Report", formats["section"])
    result = item.consensus_result
    if result is None:
        worksheet.merge_range(
            start_row + 1,
            0,
            start_row + 2,
            12,
            "No stability-consensus analysis was retained for this spectrum.",
            formats["note"],
        )
        return start_row + 3
    row = start_row + 1
    worksheet.write_string(row, 0, "Requested K", formats["label"])
    worksheet.write_number(row, 1, result.requested_peak_count, formats["integer"])
    worksheet.write_string(row, 2, "Valid model fits", formats["label"])
    worksheet.write_number(row, 3, result.successful_runs, formats["integer"])
    worksheet.write_string(row, 4, "Failed/rejected", formats["label"])
    worksheet.write_number(row, 5, result.failed_runs, formats["integer"])
    worksheet.write_string(row, 6, "Paired scenarios", formats["label"])
    worksheet.write_number(row, 7, result.paired_model_scenarios, formats["integer"])
    worksheet.write_string(row, 8, "Random seed", formats["label"])
    worksheet.write_number(row, 9, result.random_seed, formats["integer"])
    worksheet.write_string(row, 10, "Weighting / ROI", formats["label"])
    worksheet.write_string(
        row,
        11,
        f"{result.weighting} / {result.roi_jitter_fraction:.3g}",
        formats["text"],
    )
    worksheet.write_string(row, 12, f"Bootstrap ×{result.bootstrap_strength:g}", formats["text"])
    row += 2

    model_headers = (
        "Peak count",
        "Consensus support",
        "BIC support",
        "AICc support",
        "BIC criterion",
        "AICc criterion",
    )
    for column, header in enumerate(model_headers):
        worksheet.write_string(row, column, header, formats["header"])
    row += 1
    for count, probability in sorted(result.peak_count_probabilities.items()):
        worksheet.write_number(row, 0, count, formats["integer"])
        worksheet.write_number(row, 1, probability, formats["number"])
        worksheet.write_number(row, 2, result.bic_peak_count_probabilities[count], formats["number"])
        worksheet.write_number(row, 3, result.aicc_peak_count_probabilities[count], formats["number"])
        worksheet.write_number(row, 4, result.bic_by_peak_count[count], formats["number"])
        worksheet.write_number(row, 5, result.aicc_by_peak_count[count], formats["number"])
        row += 1
    row += 1

    headers = (
        "Component",
        "Samples",
        "Survival",
        "Model support",
        "Center mean",
        "Center σ",
        "FWHM mean",
        "Area mean",
        "Area CV",
        "Background sensitivity",
        "Calibration sensitivity",
        "Trust score",
        "Identifiability",
    )
    for column, header in enumerate(headers):
        worksheet.write_string(row, column, header, formats["header"])
    row += 1
    for component in result.components:
        values = (
            component.name,
            component.sample_count,
            component.survival_probability,
            component.model_support_probability,
            component.center_mean_eV,
            component.center_std_eV,
            component.fwhm_mean_eV,
            component.area_mean,
            component.area_cv,
            component.background_sensitivity,
            component.calibration_sensitivity,
            component.trust_score,
            component.identifiability,
        )
        for column, value in enumerate(values):
            _write_scalar(worksheet, row, column, value, formats)
        row += 1
    worksheet.merge_range(row, 0, row + 1, 12, result.numerical_note, formats["note"])
    return row + 2


def _write_model_comparison_table(
    worksheet: Any,
    start_row: int,
    item: WorkspaceItemData,
    formats: Mapping[str, Any],
) -> int:
    """Write ranked hypotheses, guardrails, warnings, and fitted parameters."""

    worksheet.merge_range(start_row, 0, start_row, 13, "Physics-Constrained Model Comparison", formats["section"])
    result = item.model_comparison
    if result is None:
        worksheet.merge_range(
            start_row + 1,
            0,
            start_row + 2,
            13,
            "No candidate-model comparison was retained for this spectrum.",
            formats["note"],
        )
        return start_row + 3
    row = start_row + 1
    headers = (
        "Rank",
        "Candidate",
        "K",
        "Background",
        "Profiles",
        "Eligible",
        "AICc",
        "BIC",
        "AICc weight",
        "BIC weight",
        "Combined support",
        "Residual rho1",
        "Max correlation",
        "Assessment / warnings",
    )
    for column, header in enumerate(headers):
        worksheet.write_string(row, column, header, formats["header"])
    row += 1
    for evaluation in result.candidates:
        profiles = ", ".join(
            sorted({component.profile.display_name for component in evaluation.candidate.components})
        )
        warning_text = "; ".join(evaluation.warnings) or "No automatic warning flags."
        values = (
            evaluation.rank,
            evaluation.candidate.label,
            evaluation.component_count,
            evaluation.candidate.background.method.value.title(),
            profiles,
            "Yes" if evaluation.eligible else "No",
            evaluation.aicc,
            evaluation.bic,
            evaluation.aicc_weight,
            evaluation.bic_weight,
            evaluation.model_support,
            evaluation.residual_autocorrelation,
            evaluation.max_parameter_correlation,
            f"{evaluation.identifiability}; {warning_text}",
        )
        for column, value in enumerate(values):
            _write_scalar(worksheet, row, column, value, formats)
        row += 1
    row += 1

    parameter_headers = (
        "Candidate",
        "Component",
        "Profile",
        "Center",
        "Center SE",
        "FWHM",
        "FWHM SE",
        "Area",
        "Area SE",
        "Eta",
        "Asymmetry",
        "Instrument FWHM",
        "Origin",
        "Exact background parameters",
    )
    for column, header in enumerate(parameter_headers):
        worksheet.write_string(row, column, header, formats["header"])
    row += 1
    for evaluation in result.candidates:
        values = evaluation.fit_result.parameter_values
        errors = evaluation.fit_result.standard_errors
        for component in evaluation.candidate.components:
            prefix = component.name
            entries = (
                evaluation.candidate.label,
                prefix,
                component.profile.display_name,
                values[f"{prefix}.center"],
                errors.get(f"{prefix}.center"),
                values[f"{prefix}.fwhm"],
                errors.get(f"{prefix}.fwhm"),
                values[f"{prefix}.area"],
                errors.get(f"{prefix}.area"),
                values[f"{prefix}.eta"],
                values[f"{prefix}.asymmetry"],
                values[f"{prefix}.instrument_fwhm"],
                evaluation.candidate.origin,
                _display_value(evaluation.candidate.background.parameters),
            )
            for column, value in enumerate(entries):
                _write_scalar(worksheet, row, column, value, formats)
            row += 1
    worksheet.merge_range(row, 0, row + 1, 13, result.numerical_note, formats["note"])
    return row + 2


def _write_spectrum_sheet(
    workbook: xlsxwriter.Workbook,
    item: WorkspaceItemData,
    sheet_name: str,
    index: int,
    active_index: int,
    formats: Mapping[str, Any],
) -> None:
    spectrum = item.spectrum
    result = _aligned_fit_result(item.fit_result, spectrum.size)
    background, background_note = _export_background(item, result)
    components = () if result is None else tuple(result.components.items())
    comparison_columns: list[tuple[Any, FitResult, int]] = []
    next_column = 8 + 2 * len(components)
    if item.model_comparison is not None:
        for evaluation in item.model_comparison.candidates:
            candidate_fit = _aligned_fit_result(evaluation.fit_result, spectrum.size)
            if candidate_fit is None:
                continue
            comparison_columns.append((evaluation, candidate_fit, next_column))
            next_column += 4 + len(candidate_fit.components)
    column_count = next_column
    data_start = 30
    data_end = data_start + spectrum.size
    if data_end >= EXCEL_MAX_ROWS:
        raise ValueError(f"Spectrum {spectrum.label!r} exceeds the Excel row limit.")
    if column_count > EXCEL_MAX_COLUMNS:
        raise ValueError(f"Spectrum {spectrum.label!r} has too many component columns for Excel.")

    sheet = workbook.add_worksheet(sheet_name)
    sheet.hide_gridlines(2)
    sheet.set_tab_color("#C3924C" if index == active_index else "#6E7779")
    sheet.set_zoom(85)
    sheet.set_default_row(15)
    sheet.set_row(0, 28)
    sheet.set_row(1, 20)
    sheet.set_row(data_start, 32)
    sheet.set_column(0, 1, 17)
    sheet.set_column(2, column_count - 1, 18)
    sheet.merge_range(0, 0, 0, 17, spectrum.label, formats["title"])
    active_marker = "Active workspace spectrum" if index == active_index else f"Workspace spectrum {index + 1}"
    sheet.merge_range(1, 0, 1, 17, f"{active_marker} • {spectrum.energy_scale.value.title()} energy • {spectrum.size:,} channels", formats["subtitle"])
    for summary_row in range(19, 24):
        sheet.set_row(summary_row, 28)

    _write_key_value_block(
        sheet,
        19,
        0,
        (
            ("Energy range (eV)", f"{spectrum.energy_range[0]:.6f} to {spectrum.energy_range[1]:.6f}"),
            ("Energy uncertainty (eV)", spectrum.energy_uncertainty_eV),
            ("Photon energy (eV)", spectrum.photon_energy_eV),
            ("Intensity unit", spectrum.intensity_unit),
            ("Original energy scale", spectrum.original_energy_scale.value.title()),
        ),
        formats,
    )
    _write_key_value_block(
        sheet,
        19,
        3,
        (
            ("Background model", item.background_method.value.title()),
            ("Background source", background_note),
            ("Fit status", "Not fitted" if result is None else ("Converged" if result.success else "Fit not converged")),
            ("Components", len(components)),
            ("Residual RMS", None if result is None else result.residual_rms),
        ),
        formats,
    )
    _write_key_value_block(
        sheet,
        19,
        6,
        (
            ("Weighting", None if result is None else result.weighting),
            ("Uncertainty model", None if result is None else result.uncertainty_model),
            ("Degrees of freedom", None if result is None else result.degrees_of_freedom),
            ("Reduced statistical chi-square", None if result is None else result.reduced_statistical_chi_square),
            ("Optimizer evaluations", None if result is None else result.optimizer_evaluations),
        ),
        formats,
    )
    sheet.merge_range(25, 0, 27, 8, "Measured intensity is the active spectrum shown in XPS-master. Net signal, fit envelope, residual, and component-plus-background columns are formula-driven from the visible exported cells. Individual component columns contain area-normalized net component intensities.", formats["note"])

    headers = [
        f"Active {spectrum.energy_scale.value} energy ({spectrum.energy_unit})",
        f"Original {spectrum.original_energy_scale.value} energy ({spectrum.energy_unit})",
        f"Measured intensity ({spectrum.intensity_unit})",
        f"Background ({spectrum.intensity_unit})",
        f"Net measured ({spectrum.intensity_unit})",
        f"Fit envelope ({spectrum.intensity_unit})",
        f"Residual: fit - measured ({spectrum.intensity_unit})",
        "Weighted residual",
    ]
    for name, _curve in components:
        headers.extend((f"{name} — net", f"{name} + background"))
    for evaluation, candidate_fit, _start_column in comparison_columns:
        label = evaluation.candidate.label
        headers.extend(
            (
                f"COMPARE · {label} · background",
                f"COMPARE · {label} · fit envelope",
                f"COMPARE · {label} · residual",
                f"COMPARE · {label} · weighted residual",
            )
        )
        headers.extend(f"COMPARE · {label} · {name} net" for name in candidate_fit.components)
    for column, header in enumerate(headers):
        sheet.write_string(data_start, column, header, formats["header"])

    component_net_columns = [8 + 2 * component_index for component_index in range(len(components))]
    for point in range(spectrum.size):
        row = data_start + 1 + point
        excel_row = row + 1
        sheet.write_number(row, 0, float(spectrum.energy[point]), formats["number"])
        sheet.write_number(row, 1, float(spectrum.original_energy[point]), formats["number"])
        sheet.write_number(row, 2, float(spectrum.intensity[point]), formats["scientific"])
        if background is None:
            sheet.write_blank(row, 3, None, formats["scientific"])
            sheet.write_blank(row, 4, None, formats["scientific"])
        else:
            sheet.write_number(row, 3, float(background[point]), formats["scientific"])
            sheet.write_formula(row, 4, f"=C{excel_row}-D{excel_row}", formats["scientific"], float(spectrum.intensity[point] - background[point]))

        for _evaluation, candidate_fit, start_column in comparison_columns:
            sheet.write_number(row, start_column, float(candidate_fit.background[point]), formats["scientific"])
            sheet.write_number(row, start_column + 1, float(candidate_fit.fitted_intensity[point]), formats["scientific"])
            sheet.write_number(row, start_column + 2, float(candidate_fit.residual[point]), formats["scientific"])
            sheet.write_number(row, start_column + 3, float(candidate_fit.weighted_residual[point]), formats["scientific"])
            for component_index, curve in enumerate(candidate_fit.components.values()):
                sheet.write_number(row, start_column + 4 + component_index, float(curve[point]), formats["scientific"])

        if result is None:
            for column in (5, 6, 7):
                sheet.write_blank(row, column, None, formats["scientific"])
            continue

        for component_index, (_name, curve) in enumerate(components):
            net_column = 8 + 2 * component_index
            total_column = net_column + 1
            net_value = float(curve[point])
            sheet.write_number(row, net_column, net_value, formats["scientific"])
            net_letter = xl_col_to_name(net_column)
            sheet.write_formula(
                row,
                total_column,
                f"=D{excel_row}+{net_letter}{excel_row}",
                formats["scientific"],
                float(result.background[point] + net_value),
            )
        if component_net_columns:
            references = ",".join(f"{xl_col_to_name(column)}{excel_row}" for column in component_net_columns)
            sheet.write_formula(
                row,
                5,
                f"=D{excel_row}+SUM({references})",
                formats["scientific"],
                float(result.fitted_intensity[point]),
            )
        else:
            sheet.write_number(row, 5, float(result.fitted_intensity[point]), formats["scientific"])
        sheet.write_formula(row, 6, f"=F{excel_row}-C{excel_row}", formats["scientific"], float(result.residual[point]))
        sheet.write_number(row, 7, float(result.weighted_residual[point]), formats["scientific"])

    sheet.autofilter(data_start, 0, data_end, column_count - 1)
    sheet.freeze_panes(data_start + 1, 2)
    sheet.repeat_rows(data_start, data_start)

    main_chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})
    main_chart.add_series(
        {"name": "Measured", "categories": [sheet_name, data_start + 1, 0, data_end, 0], "values": [sheet_name, data_start + 1, 2, data_end, 2], "line": {"color": "#343A40", "width": 1.25}, "marker": {"type": "none"}}
    )
    if background is not None:
        main_chart.add_series(
            {"name": "Background", "categories": [sheet_name, data_start + 1, 0, data_end, 0], "values": [sheet_name, data_start + 1, 3, data_end, 3], "line": {"color": "#C3924C", "width": 1.5}, "marker": {"type": "none"}}
        )
    for component_index, (name, _curve) in enumerate(components):
        total_column = 9 + 2 * component_index
        main_chart.add_series(
            {"name": name, "categories": [sheet_name, data_start + 1, 0, data_end, 0], "values": [sheet_name, data_start + 1, total_column, data_end, total_column], "line": {"color": COMPONENT_COLOURS[component_index % len(COMPONENT_COLOURS)], "width": 1.0}, "marker": {"type": "none"}}
        )
    if result is not None:
        main_chart.add_series(
            {"name": "Fit envelope", "categories": [sheet_name, data_start + 1, 0, data_end, 0], "values": [sheet_name, data_start + 1, 5, data_end, 5], "line": {"color": "#1F4E78", "width": 2.25}, "marker": {"type": "none"}}
        )
    for candidate_index, (evaluation, _candidate_fit, start_column) in enumerate(comparison_columns):
        main_chart.add_series(
            {
                "name": f"Compare · {evaluation.candidate.label}",
                "categories": [sheet_name, data_start + 1, 0, data_end, 0],
                "values": [sheet_name, data_start + 1, start_column + 1, data_end, start_column + 1],
                "line": {
                    "color": COMPONENT_COLOURS[candidate_index % len(COMPONENT_COLOURS)],
                    "width": 0.75,
                    "dash_type": "dash",
                    "transparency": 35,
                },
                "marker": {"type": "none"},
            }
        )
    main_chart.set_title(
        {"name": "Measured spectrum and deconvoluted components" if result is not None else "Measured spectrum and background"}
    )
    main_chart.set_x_axis(
        {
            "name": f"{spectrum.energy_scale.value.title()} energy ({spectrum.energy_unit})",
            "min": spectrum.energy_range[0],
            "max": spectrum.energy_range[1],
            "major_gridlines": {"visible": False},
        }
    )
    main_chart.set_y_axis({"name": f"Intensity ({spectrum.intensity_unit})", "major_gridlines": {"visible": True, "line": {"color": "#E6E6E6"}}})
    main_chart.set_legend({"position": "bottom"})
    main_chart.set_chartarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
    main_chart.set_plotarea({"border": {"color": "#B7B7B7"}, "fill": {"color": "#FFFFFF"}})
    main_chart.set_size({"width": 710, "height": 330})
    sheet.insert_chart(2, 0, main_chart)

    if result is not None or comparison_columns:
        residual_chart = workbook.add_chart({"type": "scatter", "subtype": "straight"})
        if result is not None:
            residual_chart.add_series(
                {"name": "Accepted fit residual", "categories": [sheet_name, data_start + 1, 0, data_end, 0], "values": [sheet_name, data_start + 1, 6, data_end, 6], "line": {"color": "#C94C4C", "width": 1.25}, "marker": {"type": "none"}}
            )
        for candidate_index, (evaluation, _candidate_fit, start_column) in enumerate(comparison_columns):
            residual_chart.add_series(
                {
                    "name": evaluation.candidate.label,
                    "categories": [sheet_name, data_start + 1, 0, data_end, 0],
                    "values": [sheet_name, data_start + 1, start_column + 2, data_end, start_column + 2],
                    "line": {
                        "color": COMPONENT_COLOURS[candidate_index % len(COMPONENT_COLOURS)],
                        "width": 0.8,
                        "transparency": 20,
                    },
                    "marker": {"type": "none"},
                }
            )
        residual_chart.set_title({"name": "Accepted and candidate-model residuals"})
        residual_chart.set_x_axis(
            {
                "name": f"{spectrum.energy_scale.value.title()} energy ({spectrum.energy_unit})",
                "min": spectrum.energy_range[0],
                "max": spectrum.energy_range[1],
                "major_gridlines": {"visible": False},
            }
        )
        residual_chart.set_y_axis({"name": f"Fit - measured ({spectrum.intensity_unit})", "crossing": 0, "major_gridlines": {"visible": True, "line": {"color": "#E6E6E6"}}})
        residual_chart.set_legend({"position": "bottom"})
        residual_chart.set_chartarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
        residual_chart.set_plotarea({"border": {"color": "#B7B7B7"}, "fill": {"color": "#FFFFFF"}})
        residual_chart.set_size({"width": 710, "height": 330})
        sheet.insert_chart(2, 9, residual_chart)
    else:
        sheet.merge_range(3, 9, 16, 17, "Residual chart unavailable\n\nNo retained fit result exists for this spectrum. The measured curve and calculated background are still exported.", formats["note"])

    row = data_end + 3
    row = _write_fit_parameter_table(sheet, row, item, result, formats) + 2
    row = _write_ties_table(sheet, row, item, formats) + 2
    row = _write_model_comparison_table(sheet, row, item, formats) + 2
    row = _write_consensus_table(sheet, row, item, formats) + 2

    sheet.merge_range(row, 0, row, 5, "Background parameters", formats["section"])
    row += 1
    for key, value in item.background_parameters.items():
        sheet.write_string(row, 0, str(key), formats["label"])
        sheet.merge_range(row, 1, row, 5, _display_value(value), formats["wrapped"])
        row += 1
    if not item.background_parameters:
        sheet.merge_range(row, 0, row, 5, "Default/no explicit background parameters.", formats["note"])
        row += 1
    row += 2

    if result is not None:
        sheet.merge_range(row, 0, row, 5, "Likelihood and identifiability diagnostics", formats["section"])
        entries = (
            ("Optimization objective", result.objective_name),
            ("Objective value", result.objective_value),
            ("Independent parameters", result.free_parameter_count),
            ("Jacobian rank", result.jacobian_rank),
            ("Scaled Jacobian condition", result.jacobian_condition),
            ("Maximum parameter correlation", result.max_parameter_correlation),
            ("Parameters at bounds", "; ".join(result.boundary_parameters)),
            ("Fit review warnings", "\n".join(result.fit_warnings)),
        )
        _write_key_value_block(sheet, row+1, 0, entries, formats)
        row += len(entries)+3
    row = _write_history_table(sheet, row, "Processing history", spectrum.processing_history, formats) + 2
    row = _write_history_table(sheet, row, "Energy calibration / correction history", spectrum.correction_history, formats) + 2
    row = _write_survey_table(sheet, row, item, formats) + 2

    sheet.merge_range(row, 0, row, 5, "Acquisition and source metadata", formats["section"])
    row += 1
    if spectrum.metadata:
        for key, value in sorted(spectrum.metadata.items(), key=lambda item: str(item[0]).casefold()):
            sheet.write_string(row, 0, str(key), formats["label"])
            sheet.merge_range(row, 1, row, 5, _display_value(value), formats["wrapped"])
            row += 1
    else:
        sheet.merge_range(row, 0, row, 5, "No acquisition metadata were present in the imported spectrum.", formats["note"])

    sheet.set_landscape()
    sheet.fit_to_pages(1, 0)
    sheet.set_margins(0.3, 0.3, 0.5, 0.5)
    sheet.set_header("&CXPS-master — scientific data export")
    sheet.set_footer("&LWorkbook sheet: &A&CPage &P of &N&RGenerated &D &T")


def export_workspace_to_excel(
    path: str | Path,
    workspace_items: Sequence[WorkspaceItemData],
    *,
    active_workspace_index: int = 0,
    project_name: str = "Untitled XPS project",
    global_series_result: GlobalSeriesResult | None = None,
) -> tuple[str, ...]:
    """Export a complete workspace to one styled, auditable `.xlsx` file.

    The write is atomic: a completed temporary workbook replaces the target
    only after every worksheet and chart has been finalized successfully.
    """

    items = tuple(workspace_items)
    if not items:
        raise ValueError("At least one workspace spectrum is required for Excel export.")
    if not 0 <= active_workspace_index < len(items):
        raise ValueError("The active workspace index is outside the exported workspace.")
    target = Path(path)
    if target.suffix.lower() != ".xlsx":
        target = target.with_suffix(".xlsx")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    existing_names = {"export summary", "global series"}
    sheet_names = tuple(
        _safe_sheet_name(item.spectrum.label, index, existing_names)
        for index, item in enumerate(items, start=1)
    )

    workbook: xlsxwriter.Workbook | None = None
    try:
        workbook = xlsxwriter.Workbook(
            str(temporary),
            {
                "constant_memory": False,
                "strings_to_formulas": False,
                "strings_to_urls": False,
                "nan_inf_to_errors": False,
            },
        )
        workbook.set_properties(
            {
                "title": f"XPS-master export — {project_name}",
                "subject": "XPS spectra, deconvoluted components, residuals, fit parameters, and provenance",
                "author": "Ali Afruz",
                "company": "chemitro.ir",
                "comments": "Generated from a retained XPS-master workspace. Scientific review is required.",
            }
        )
        formats = _formats(workbook)
        _write_summary_sheet(
            workbook,
            items,
            sheet_names,
            active_workspace_index,
            project_name,
            formats,
            global_series_result,
        )
        if global_series_result is not None:
            _write_global_series_sheet(workbook, global_series_result, items, formats)
        for index, (item, sheet_name) in enumerate(zip(items, sheet_names, strict=True)):
            _write_spectrum_sheet(workbook, item, sheet_name, index, active_workspace_index, formats)
        workbook.close()
        workbook = None
        os.replace(temporary, target)
    except Exception as error:
        if workbook is not None:
            try:
                workbook.close()
            except Exception:
                pass
        if isinstance(error, (OSError, XlsxWriterException)):
            raise ExcelExportError(f"Could not create Excel workbook: {error}") from error
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return sheet_names
