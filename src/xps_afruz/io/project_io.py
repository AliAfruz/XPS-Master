"""Portable JSON project files with complete workspace provenance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..backgrounds import BackgroundDiagnostics, BackgroundMethod, BackgroundSettings
from ..chemistry.survey import (
    DetectedPeak,
    ElementCandidate,
    ExpectedSecondaryLine,
    SurveyAnalysis,
    SurveyMatch,
)
from ..chemistry.survey_library import ReferenceLine
from ..core.energy import EnergyScale
from ..core.spectrum import Spectrum
from ..fitting.model import FitResult, ParameterTie, PeakComponent
from ..fitting.global_series import (
    GlobalComponentTrend,
    GlobalSeriesResult,
    GlobalSeriesSettings,
    GlobalSpectrumFit,
)
from ..fitting.model_comparison import (
    ModelCandidate,
    ModelCandidateEvaluation,
    ModelComparisonResult,
)
from ..fitting.stability_consensus import ComponentStability, StabilityConsensusResult


PROJECT_FORMAT_VERSION = 10
SUPPORTED_PROJECT_VERSIONS = frozenset(range(1, PROJECT_FORMAT_VERSION + 1))


@dataclass(frozen=True, slots=True)
class WorkspaceItemData:
    """One spectrum and every retained result attached to its workspace card."""

    spectrum: Spectrum
    background_method: BackgroundMethod
    background_parameters: Mapping[str, Any]
    peaks: tuple[dict[str, Any], ...] = ()
    ties: tuple[dict[str, Any], ...] = ()
    fit_result: FitResult | None = None
    survey_analysis: SurveyAnalysis | None = None
    consensus_result: StabilityConsensusResult | None = None
    model_comparison: ModelComparisonResult | None = None


@dataclass(frozen=True, slots=True)
class ProjectData:
    """Deserialized project content, including the complete v10 workspace."""

    spectrum: Spectrum
    background_method: BackgroundMethod
    background_parameters: Mapping[str, Any]
    peaks: tuple[dict[str, Any], ...]
    ties: tuple[dict[str, Any], ...]
    analysis_settings: Mapping[str, Any]
    workspace: tuple[WorkspaceItemData, ...]
    active_workspace_index: int
    global_series_result: GlobalSeriesResult | None
    ui_state: Mapping[str, Any]
    format_version: int


def _json_safe(value: Any) -> Any:
    """Convert supported numerical/configuration values to strict JSON data."""

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Project data contains an unsupported value: {type(value).__name__}")


def _spectrum_to_payload(spectrum: Spectrum) -> dict[str, Any]:
    return {
        "energy": spectrum.energy.tolist(),
        "intensity": spectrum.intensity.tolist(),
        "label": spectrum.label,
        "energy_unit": spectrum.energy_unit,
        "intensity_unit": spectrum.intensity_unit,
        "metadata": _json_safe(spectrum.metadata),
        "energy_scale": spectrum.energy_scale.value,
        "photon_energy_eV": spectrum.photon_energy_eV,
        "energy_uncertainty_eV": spectrum.energy_uncertainty_eV,
        "original_energy": spectrum.original_energy.tolist(),
        "original_energy_scale": spectrum.original_energy_scale.value,
        "correction_history": _json_safe(spectrum.correction_history),
        "processing_history": _json_safe(spectrum.processing_history),
    }


def _spectrum_from_payload(data: Mapping[str, Any]) -> Spectrum:
    return Spectrum(
        data["energy"],
        data["intensity"],
        label=str(data.get("label", "Loaded project")),
        energy_unit=str(data.get("energy_unit", "eV")),
        intensity_unit=str(data.get("intensity_unit", "counts")),
        metadata=data.get("metadata", {}),
        energy_scale=str(data.get("energy_scale", "binding")),
        photon_energy_eV=data.get("photon_energy_eV"),
        energy_uncertainty_eV=float(data.get("energy_uncertainty_eV", 0.0)),
        original_energy=data.get("original_energy"),
        original_energy_scale=data.get("original_energy_scale"),
        correction_history=tuple(data.get("correction_history", ())),
        processing_history=tuple(data.get("processing_history", ())),
    )


def _reference_to_payload(reference: ReferenceLine) -> dict[str, Any]:
    return {
        "symbol": reference.symbol,
        "element": reference.element,
        "line": reference.line,
        "binding_energy": reference.binding_energy,
        "relative_strength": reference.relative_strength,
        "feature_type": reference.feature_type,
        "kinetic_energy": reference.kinetic_energy,
        "screening_context": reference.screening_context,
    }


def _reference_from_payload(data: Mapping[str, Any]) -> ReferenceLine:
    return ReferenceLine(
        symbol=str(data["symbol"]),
        element=str(data["element"]),
        line=str(data["line"]),
        binding_energy=None if data.get("binding_energy") is None else float(data["binding_energy"]),
        relative_strength=float(data.get("relative_strength", 1.0)),
        feature_type=str(data.get("feature_type", "photoelectron")),
        kinetic_energy=None if data.get("kinetic_energy") is None else float(data["kinetic_energy"]),
        screening_context=str(data.get("screening_context", "")),
    )


def _peak_to_payload(peak: DetectedPeak) -> dict[str, Any]:
    return {
        "maximum_energy": peak.maximum_energy,
        "centroid_energy": peak.centroid_energy,
        "intensity": peak.intensity,
        "prominence": peak.prominence,
        "fwhm_estimate": peak.fwhm_estimate,
        "background_corrected_area": peak.background_corrected_area,
        "signal_to_noise": peak.signal_to_noise,
        "targeted": peak.targeted,
    }


def _peak_from_payload(data: Mapping[str, Any]) -> DetectedPeak:
    return DetectedPeak(
        maximum_energy=float(data["maximum_energy"]),
        centroid_energy=float(data["centroid_energy"]),
        intensity=float(data["intensity"]),
        prominence=float(data["prominence"]),
        fwhm_estimate=float(data["fwhm_estimate"]),
        background_corrected_area=float(data["background_corrected_area"]),
        signal_to_noise=float(data["signal_to_noise"]),
        targeted=bool(data.get("targeted", False)),
    )


def _match_to_payload(match: SurveyMatch) -> dict[str, Any]:
    return {
        "reference": _reference_to_payload(match.reference),
        "peak": _peak_to_payload(match.peak),
        "reference_energy": match.reference_energy,
        "energy_error": match.energy_error,
        "score": match.score,
        "overlapping_elements": list(match.overlapping_elements),
    }


def _match_from_payload(data: Mapping[str, Any]) -> SurveyMatch:
    return SurveyMatch(
        reference=_reference_from_payload(_mapping(data.get("reference"), "survey reference")),
        peak=_peak_from_payload(_mapping(data.get("peak"), "survey peak")),
        reference_energy=float(data["reference_energy"]),
        energy_error=float(data["energy_error"]),
        score=float(data["score"]),
        overlapping_elements=tuple(str(item) for item in data.get("overlapping_elements", ())),
    )


def _secondary_to_payload(line: ExpectedSecondaryLine) -> dict[str, Any]:
    return {
        "reference": _reference_to_payload(line.reference),
        "expected_energy": line.expected_energy,
        "observed": line.observed,
        "energy_error": line.energy_error,
    }


def _secondary_from_payload(data: Mapping[str, Any]) -> ExpectedSecondaryLine:
    return ExpectedSecondaryLine(
        reference=_reference_from_payload(_mapping(data.get("reference"), "secondary reference")),
        expected_energy=float(data["expected_energy"]),
        observed=bool(data["observed"]),
        energy_error=None if data.get("energy_error") is None else float(data["energy_error"]),
    )


def _candidate_to_payload(candidate: ElementCandidate) -> dict[str, Any]:
    return {
        "symbol": candidate.symbol,
        "element": candidate.element,
        "score": candidate.score,
        "consistency_score": candidate.consistency_score,
        "matches": [_match_to_payload(match) for match in candidate.matches],
        "expected_secondary_lines": [_secondary_to_payload(line) for line in candidate.expected_secondary_lines],
        "overlaps_with": list(candidate.overlaps_with),
    }


def _candidate_from_payload(data: Mapping[str, Any]) -> ElementCandidate:
    return ElementCandidate(
        symbol=str(data["symbol"]),
        element=str(data["element"]),
        score=float(data["score"]),
        consistency_score=float(data["consistency_score"]),
        matches=tuple(_match_from_payload(_mapping(item, "candidate match")) for item in data.get("matches", ())),
        expected_secondary_lines=tuple(
            _secondary_from_payload(_mapping(item, "secondary line"))
            for item in data.get("expected_secondary_lines", ())
        ),
        overlaps_with=tuple(str(item) for item in data.get("overlaps_with", ())),
    )


def _survey_to_payload(analysis: SurveyAnalysis | None) -> dict[str, Any] | None:
    if analysis is None:
        return None
    return {
        "spectrum_label": analysis.spectrum_label,
        "energy_range": list(analysis.energy_range),
        "energy_scale": analysis.energy_scale.value,
        "photon_energy_eV": analysis.photon_energy_eV,
        "tolerance": analysis.tolerance,
        "noise": analysis.noise,
        "detection_limit": analysis.detection_limit,
        "peaks": [_peak_to_payload(peak) for peak in analysis.peaks],
        "matches": [_match_to_payload(match) for match in analysis.matches],
        "candidates": [_candidate_to_payload(candidate) for candidate in analysis.candidates],
        "unassigned_peaks": [_peak_to_payload(peak) for peak in analysis.unassigned_peaks],
        "included_elements": list(analysis.included_elements),
        "excluded_elements": list(analysis.excluded_elements),
        "trace_searched_elements": list(analysis.trace_searched_elements),
        "trace_recovered_elements": list(analysis.trace_recovered_elements),
        "trace_minimum_snr": analysis.trace_minimum_snr,
    }


def _survey_from_payload(data: Mapping[str, Any] | None) -> SurveyAnalysis | None:
    if data is None:
        return None
    energy_range = data.get("energy_range", (0.0, 0.0))
    if not isinstance(energy_range, (list, tuple)) or len(energy_range) != 2:
        raise ValueError("Saved survey energy range is invalid.")
    return SurveyAnalysis(
        spectrum_label=str(data["spectrum_label"]),
        energy_range=(float(energy_range[0]), float(energy_range[1])),
        energy_scale=EnergyScale(str(data["energy_scale"])),
        photon_energy_eV=None if data.get("photon_energy_eV") is None else float(data["photon_energy_eV"]),
        tolerance=float(data["tolerance"]),
        noise=float(data["noise"]),
        detection_limit=float(data["detection_limit"]),
        peaks=tuple(_peak_from_payload(_mapping(item, "detected peak")) for item in data.get("peaks", ())),
        matches=tuple(_match_from_payload(_mapping(item, "survey match")) for item in data.get("matches", ())),
        candidates=tuple(_candidate_from_payload(_mapping(item, "element candidate")) for item in data.get("candidates", ())),
        unassigned_peaks=tuple(
            _peak_from_payload(_mapping(item, "unassigned peak")) for item in data.get("unassigned_peaks", ())
        ),
        included_elements=tuple(str(item) for item in data.get("included_elements", ())),
        excluded_elements=tuple(str(item) for item in data.get("excluded_elements", ())),
        trace_searched_elements=tuple(str(item) for item in data.get("trace_searched_elements", ())),
        trace_recovered_elements=tuple(str(item) for item in data.get("trace_recovered_elements", ())),
        trace_minimum_snr=float(data.get("trace_minimum_snr", 2.5)),
    )


def _fit_result_to_payload(result: FitResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return _json_safe(
        {
            "success": result.success,
            "message": result.message,
            "parameter_values": result.parameter_values,
            "standard_errors": result.standard_errors,
            "background": result.background,
            "components": result.components,
            "fitted_intensity": result.fitted_intensity,
            "residual": result.residual,
            "weighted_residual": result.weighted_residual,
            "chi_square": result.chi_square,
            "reduced_chi_square": result.reduced_chi_square,
            "degrees_of_freedom": result.degrees_of_freedom,
            "optimizer_evaluations": result.optimizer_evaluations,
            "weighting": result.weighting,
            "count_scale": result.count_scale,
            "residual_rms": result.residual_rms,
            "statistical_chi_square": result.statistical_chi_square,
            "reduced_statistical_chi_square": result.reduced_statistical_chi_square,
            "uncertainty_model": result.uncertainty_model,
            "objective_name": result.objective_name,
            "objective_value": result.objective_value,
            "free_parameter_count": result.free_parameter_count,
            "jacobian_rank": result.jacobian_rank,
            "jacobian_condition": result.jacobian_condition,
            "max_parameter_correlation": result.max_parameter_correlation,
            "boundary_parameters": result.boundary_parameters,
            "fit_warnings": result.fit_warnings,
        }
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _float_mapping(value: Any, label: str) -> dict[str, float]:
    data = _mapping(value, label)
    return {str(key): float("nan") if item is None else float(item) for key, item in data.items()}


def _fit_result_from_payload(data: Mapping[str, Any] | None) -> FitResult | None:
    if data is None:
        return None
    components_data = _mapping(data.get("components"), "fit components")
    return FitResult(
        success=bool(data["success"]),
        message=str(data["message"]),
        parameter_values=_float_mapping(data.get("parameter_values"), "fit parameter values"),
        standard_errors=_float_mapping(data.get("standard_errors"), "fit standard errors"),
        background=np.asarray(data["background"], dtype=float),
        components={str(name): np.asarray(values, dtype=float) for name, values in components_data.items()},
        fitted_intensity=np.asarray(data["fitted_intensity"], dtype=float),
        residual=np.asarray(data["residual"], dtype=float),
        weighted_residual=np.asarray(data["weighted_residual"], dtype=float),
        chi_square=float(data["chi_square"]),
        reduced_chi_square=float("nan") if data.get("reduced_chi_square") is None else float(data["reduced_chi_square"]),
        degrees_of_freedom=int(data["degrees_of_freedom"]),
        optimizer_evaluations=int(data["optimizer_evaluations"]),
        weighting=str(data["weighting"]),
        count_scale=_optional_float(data.get("count_scale")),
        residual_rms=float(data["residual_rms"]),
        statistical_chi_square=_optional_float(data.get("statistical_chi_square")),
        reduced_statistical_chi_square=_optional_float(data.get("reduced_statistical_chi_square")),
        uncertainty_model=str(data["uncertainty_model"]),
        objective_name=str(data.get("objective_name", "weighted_least_squares")),
        objective_value=_optional_float(data.get("objective_value")),
        free_parameter_count=None if data.get("free_parameter_count") is None else int(data["free_parameter_count"]),
        jacobian_rank=None if data.get("jacobian_rank") is None else int(data["jacobian_rank"]),
        jacobian_condition=_optional_float(data.get("jacobian_condition")),
        max_parameter_correlation=_optional_float(data.get("max_parameter_correlation")),
        boundary_parameters=tuple(str(name) for name in data.get("boundary_parameters", ())),
        fit_warnings=tuple(str(warning) for warning in data.get("fit_warnings", ())),
    )


def _bound_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Saved {label} bounds are invalid.")
    lower = -float("inf") if value[0] is None else float(value[0])
    upper = float("inf") if value[1] is None else float(value[1])
    return lower, upper


def _component_to_payload(component: PeakComponent) -> dict[str, Any]:
    return _json_safe(asdict(component))


def _component_from_payload(data: Mapping[str, Any]) -> PeakComponent:
    return PeakComponent(
        name=str(data["name"]),
        center=float(data["center"]),
        fwhm=float(data["fwhm"]),
        area=float(data["area"]),
        profile=str(data["profile"]),
        eta=float(data.get("eta", 0.5)),
        asymmetry=float(data.get("asymmetry", 0.0)),
        instrument_fwhm=float(data.get("instrument_fwhm", 0.0)),
        center_bounds=_bound_pair(data.get("center_bounds", (None, None)), "center"),
        fwhm_bounds=_bound_pair(data.get("fwhm_bounds", (0.05, 20.0)), "FWHM"),
        area_bounds=_bound_pair(data.get("area_bounds", (0.0, None)), "area"),
        eta_bounds=_bound_pair(data.get("eta_bounds", (0.0, 1.0)), "eta"),
        asymmetry_bounds=_bound_pair(data.get("asymmetry_bounds", (-0.9, 0.9)), "asymmetry"),
        instrument_fwhm_bounds=_bound_pair(
            data.get("instrument_fwhm_bounds", (0.0, 20.0)),
            "instrument FWHM",
        ),
        vary_center=bool(data.get("vary_center", True)),
        vary_fwhm=bool(data.get("vary_fwhm", True)),
        vary_area=bool(data.get("vary_area", True)),
        vary_eta=bool(data.get("vary_eta", False)),
        vary_asymmetry=bool(data.get("vary_asymmetry", False)),
        vary_instrument_fwhm=bool(data.get("vary_instrument_fwhm", False)),
    )


def _tie_to_payload(tie: ParameterTie) -> dict[str, Any]:
    return _json_safe(asdict(tie))


def _tie_from_payload(data: Mapping[str, Any]) -> ParameterTie:
    return ParameterTie(
        target=str(data["target"]),
        source=str(data["source"]),
        scale=float(data.get("scale", 1.0)),
        offset=float(data.get("offset", 0.0)),
    )


def _global_series_to_payload(result: GlobalSeriesResult | None) -> dict[str, Any] | None:
    """Serialize the project-level simultaneous fit and its reference fits."""

    if result is None:
        return None
    return _json_safe(
        {
            "settings": asdict(result.settings),
            "spectrum_fits": [
                {
                    "workspace_index": item.workspace_index,
                    "spectrum_label": item.spectrum_label,
                    "energy_shift_eV": item.energy_shift_eV,
                    "energy_shift_stderr_eV": item.energy_shift_stderr_eV,
                    "global_fit": _fit_result_to_payload(item.global_fit),
                    "independent_fit": _fit_result_to_payload(item.independent_fit),
                }
                for item in result.spectrum_fits
            ],
            "component_trends": [asdict(trend) for trend in result.component_trends],
            "ties": [_tie_to_payload(tie) for tie in result.ties],
            "success": result.success,
            "message": result.message,
            "optimizer_evaluations": result.optimizer_evaluations,
            "total_samples": result.total_samples,
            "global_parameter_count": result.global_parameter_count,
            "independent_parameter_count": result.independent_parameter_count,
            "degrees_of_freedom": result.degrees_of_freedom,
            "global_objective": result.global_objective,
            "reduced_global_objective": result.reduced_global_objective,
            "global_aicc": result.global_aicc,
            "global_bic": result.global_bic,
            "independent_objective": result.independent_objective,
            "independent_aicc": result.independent_aicc,
            "independent_bic": result.independent_bic,
            "delta_aicc": result.delta_aicc,
            "delta_bic": result.delta_bic,
            "preference": result.preference,
            "numerical_note": result.numerical_note,
        }
    )


def _global_float(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _global_series_from_payload(data: Mapping[str, Any] | None) -> GlobalSeriesResult | None:
    if data is None:
        return None
    settings_data = _mapping(data.get("settings"), "global-series settings")
    raw_fits = data.get("spectrum_fits", ())
    raw_trends = data.get("component_trends", ())
    raw_ties = data.get("ties", ())
    if not isinstance(raw_fits, list) or not isinstance(raw_trends, list) or not isinstance(raw_ties, list):
        raise ValueError("Saved global-series arrays are invalid.")
    spectrum_fits: list[GlobalSpectrumFit] = []
    for raw in raw_fits:
        item = _mapping(raw, "global spectrum fit")
        global_fit = _fit_result_from_payload(_mapping(item.get("global_fit"), "global fit result"))
        independent_fit = _fit_result_from_payload(_mapping(item.get("independent_fit"), "independent fit result"))
        if global_fit is None or independent_fit is None:
            raise ValueError("Saved global-series fit results are missing.")
        spectrum_fits.append(
            GlobalSpectrumFit(
                workspace_index=int(item["workspace_index"]),
                spectrum_label=str(item["spectrum_label"]),
                energy_shift_eV=float(item["energy_shift_eV"]),
                energy_shift_stderr_eV=_global_float(item.get("energy_shift_stderr_eV")),
                global_fit=global_fit,
                independent_fit=independent_fit,
            )
        )
    trends: list[GlobalComponentTrend] = []
    tuple_fields = (
        "centers_eV",
        "center_stderr_eV",
        "fwhm_eV",
        "fwhm_stderr_eV",
        "areas",
        "area_stderr",
        "eta",
        "asymmetry",
        "instrument_fwhm_eV",
    )
    for raw in raw_trends:
        trend = _mapping(raw, "global component trend")
        values = {
            field: tuple(_global_float(value) for value in trend.get(field, ()))
            for field in tuple_fields
        }
        trends.append(
            GlobalComponentTrend(
                name=str(trend["name"]),
                profile=str(trend["profile"]),
                reference_center_eV=float(trend["reference_center_eV"]),
                reference_center_stderr_eV=_global_float(trend.get("reference_center_stderr_eV")),
                **values,
                vary_eta=bool(trend.get("vary_eta", False)),
                vary_asymmetry=bool(trend.get("vary_asymmetry", False)),
                vary_instrument_fwhm=bool(trend.get("vary_instrument_fwhm", False)),
            )
        )
    return GlobalSeriesResult(
        settings=GlobalSeriesSettings(**dict(settings_data)),
        spectrum_fits=tuple(spectrum_fits),
        component_trends=tuple(trends),
        ties=tuple(_tie_from_payload(_mapping(tie, "global-series tie")) for tie in raw_ties),
        success=bool(data["success"]),
        message=str(data["message"]),
        optimizer_evaluations=int(data["optimizer_evaluations"]),
        total_samples=int(data["total_samples"]),
        global_parameter_count=int(data["global_parameter_count"]),
        independent_parameter_count=int(data["independent_parameter_count"]),
        degrees_of_freedom=int(data["degrees_of_freedom"]),
        global_objective=float(data["global_objective"]),
        reduced_global_objective=_global_float(data.get("reduced_global_objective")),
        global_aicc=_global_float(data.get("global_aicc")),
        global_bic=_global_float(data.get("global_bic")),
        independent_objective=float(data["independent_objective"]),
        independent_aicc=_global_float(data.get("independent_aicc")),
        independent_bic=_global_float(data.get("independent_bic")),
        delta_aicc=_global_float(data.get("delta_aicc")),
        delta_bic=_global_float(data.get("delta_bic")),
        preference=str(data["preference"]),
        numerical_note=str(data["numerical_note"]),
    )


def _model_candidate_to_payload(candidate: ModelCandidate) -> dict[str, Any]:
    return {
        "identifier": candidate.identifier,
        "label": candidate.label,
        "components": [_component_to_payload(component) for component in candidate.components],
        "ties": [_tie_to_payload(tie) for tie in candidate.ties],
        "background": candidate.background.as_dict(),
        "origin": candidate.origin,
    }


def _model_candidate_from_payload(data: Mapping[str, Any]) -> ModelCandidate:
    raw_components = data.get("components", ())
    raw_ties = data.get("ties", ())
    if not isinstance(raw_components, list) or not isinstance(raw_ties, list):
        raise ValueError("Saved model-comparison candidate components or ties are invalid.")
    background = _mapping(data.get("background"), "candidate background")
    return ModelCandidate(
        identifier=str(data["identifier"]),
        label=str(data["label"]),
        components=tuple(
            _component_from_payload(_mapping(component, "candidate component"))
            for component in raw_components
        ),
        ties=tuple(_tie_from_payload(_mapping(tie, "candidate tie")) for tie in raw_ties),
        background=BackgroundSettings(
            str(background.get("method", BackgroundMethod.LINEAR.value)),
            dict(_mapping(background.get("parameters", {}), "candidate background parameters")),
        ),
        origin=str(data["origin"]),
    )


def _model_comparison_to_payload(result: ModelComparisonResult | None) -> dict[str, Any] | None:
    """Serialize every retained alternative curve and diagnostic."""

    if result is None:
        return None
    return _json_safe(
        {
            "spectrum_label": result.spectrum_label,
            "recommended_candidate_id": result.recommended_candidate_id,
            "weighting": result.weighting,
            "count_scale": result.count_scale,
            "numerical_note": result.numerical_note,
            "candidates": [
                {
                    "candidate": _model_candidate_to_payload(evaluation.candidate),
                    "fit_result": _fit_result_to_payload(evaluation.fit_result),
                    "background_diagnostics": asdict(evaluation.background_diagnostics),
                    "parameter_count": evaluation.parameter_count,
                    "aicc": evaluation.aicc,
                    "bic": evaluation.bic,
                    "aicc_weight": evaluation.aicc_weight,
                    "bic_weight": evaluation.bic_weight,
                    "model_support": evaluation.model_support,
                    "residual_autocorrelation": evaluation.residual_autocorrelation,
                    "durbin_watson": evaluation.durbin_watson,
                    "boundary_hit_fraction": evaluation.boundary_hit_fraction,
                    "collapse_risk": evaluation.collapse_risk,
                    "max_parameter_correlation": evaluation.max_parameter_correlation,
                    "identifiability": evaluation.identifiability,
                    "eligible": evaluation.eligible,
                    "rank": evaluation.rank,
                    "warnings": evaluation.warnings,
                }
                for evaluation in result.candidates
            ],
        }
    )


def _model_comparison_from_payload(data: Mapping[str, Any] | None) -> ModelComparisonResult | None:
    if data is None:
        return None
    raw_candidates = data.get("candidates", ())
    if not isinstance(raw_candidates, list):
        raise ValueError("Saved model-comparison candidates are invalid.")
    evaluations: list[ModelCandidateEvaluation] = []
    for raw in raw_candidates:
        entry = _mapping(raw, "model-comparison evaluation")
        fit = _fit_result_from_payload(_mapping(entry.get("fit_result"), "candidate fit result"))
        if fit is None:
            raise ValueError("Saved model-comparison candidate has no fit result.")
        diagnostics = _mapping(entry.get("background_diagnostics"), "candidate background diagnostics")
        evaluations.append(
            ModelCandidateEvaluation(
                candidate=_model_candidate_from_payload(_mapping(entry.get("candidate"), "model candidate")),
                fit_result=fit,
                background_diagnostics=BackgroundDiagnostics(
                    left_anchor=float(diagnostics["left_anchor"]),
                    right_anchor=float(diagnostics["right_anchor"]),
                    fraction_above_signal=float(diagnostics["fraction_above_signal"]),
                    minimum_net_intensity=float(diagnostics["minimum_net_intensity"]),
                    robust_net_noise=float(diagnostics["robust_net_noise"]),
                ),
                parameter_count=int(entry["parameter_count"]),
                aicc=float(entry["aicc"]),
                bic=float(entry["bic"]),
                aicc_weight=float(entry["aicc_weight"]),
                bic_weight=float(entry["bic_weight"]),
                model_support=float(entry["model_support"]),
                residual_autocorrelation=float(entry["residual_autocorrelation"]),
                durbin_watson=float(entry["durbin_watson"]),
                boundary_hit_fraction=float(entry["boundary_hit_fraction"]),
                collapse_risk=float(entry["collapse_risk"]),
                max_parameter_correlation=float(entry["max_parameter_correlation"]),
                identifiability=str(entry["identifiability"]),
                eligible=bool(entry["eligible"]),
                rank=int(entry["rank"]),
                warnings=tuple(str(warning) for warning in entry.get("warnings", ())),
            )
        )
    return ModelComparisonResult(
        spectrum_label=str(data["spectrum_label"]),
        candidates=tuple(evaluations),
        recommended_candidate_id=str(data["recommended_candidate_id"]),
        weighting=str(data["weighting"]),
        count_scale=_optional_float(data.get("count_scale")),
        numerical_note=str(data["numerical_note"]),
    )


def _consensus_result_to_payload(result: StabilityConsensusResult | None) -> dict[str, Any] | None:
    """Serialize aggregate Trust Report evidence without transient replica curves."""

    if result is None:
        return None
    return _json_safe(
        {
            "spectrum_label": result.spectrum_label,
            "requested_peak_count": result.requested_peak_count,
            "peak_count_probabilities": result.peak_count_probabilities,
            "bic_peak_count_probabilities": result.bic_peak_count_probabilities,
            "aicc_peak_count_probabilities": result.aicc_peak_count_probabilities,
            "bic_by_peak_count": result.bic_by_peak_count,
            "aicc_by_peak_count": result.aicc_by_peak_count,
            "components": [asdict(component) for component in result.components],
            "requested_runs": result.requested_runs,
            "successful_runs": result.successful_runs,
            "failed_runs": result.failed_runs,
            "paired_model_scenarios": result.paired_model_scenarios,
            "candidate_trials_per_run": result.candidate_trials_per_run,
            "random_seed": result.random_seed,
            "background_models": result.background_models,
            "profiles": result.profiles,
            "weighting": result.weighting,
            "count_scale": result.count_scale,
            "calibration_uncertainty_eV": result.calibration_uncertainty_eV,
            "roi_jitter_fraction": result.roi_jitter_fraction,
            "bootstrap_strength": result.bootstrap_strength,
            "calibration_sigma_scale": result.calibration_sigma_scale,
            "max_nfev": result.max_nfev,
            "max_background_above_fraction": result.max_background_above_fraction,
            "numerical_note": result.numerical_note,
        }
    )


def _consensus_float(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _consensus_result_from_payload(data: Mapping[str, Any] | None) -> StabilityConsensusResult | None:
    """Restore one validated aggregate Trust Report."""

    if data is None:
        return None
    raw_components = data.get("components", ())
    if not isinstance(raw_components, list):
        raise ValueError("Saved stability-consensus components are invalid.")
    components: list[ComponentStability] = []
    for raw in raw_components:
        component = _mapping(raw, "stability-consensus component")
        components.append(
            ComponentStability(
                name=str(component["name"]),
                sample_count=int(component["sample_count"]),
                survival_probability=_consensus_float(component["survival_probability"]),
                model_support_probability=_consensus_float(component["model_support_probability"]),
                center_mean_eV=_consensus_float(component["center_mean_eV"]),
                center_std_eV=_consensus_float(component["center_std_eV"]),
                fwhm_mean_eV=_consensus_float(component["fwhm_mean_eV"]),
                fwhm_std_eV=_consensus_float(component["fwhm_std_eV"]),
                area_mean=_consensus_float(component["area_mean"]),
                area_std=_consensus_float(component["area_std"]),
                area_cv=_consensus_float(component["area_cv"]),
                background_sensitivity=_consensus_float(component["background_sensitivity"]),
                background_sensitivity_label=str(component["background_sensitivity_label"]),
                calibration_sensitivity=_consensus_float(component["calibration_sensitivity"]),
                calibration_sensitivity_label=str(component["calibration_sensitivity_label"]),
                calibration_contribution_eV=_consensus_float(component["calibration_contribution_eV"]),
                boundary_hit_frequency=_consensus_float(component["boundary_hit_frequency"]),
                collapse_frequency=_consensus_float(component["collapse_frequency"]),
                max_parameter_correlation=_consensus_float(component["max_parameter_correlation"]),
                trust_score=_consensus_float(component["trust_score"]),
                identifiability=str(component["identifiability"]),
            )
        )

    def integer_mapping(key: str, label: str) -> dict[int, float]:
        return {int(count): value for count, value in _float_mapping(data.get(key), label).items()}

    combined_probabilities = integer_mapping("peak_count_probabilities", "peak-count probabilities")
    bic_probabilities = (
        integer_mapping("bic_peak_count_probabilities", "peak-count BIC probabilities")
        if data.get("bic_peak_count_probabilities") is not None
        else dict(combined_probabilities)
    )
    aicc_probabilities = (
        integer_mapping("aicc_peak_count_probabilities", "peak-count AICc probabilities")
        if data.get("aicc_peak_count_probabilities") is not None
        else dict(combined_probabilities)
    )
    return StabilityConsensusResult(
        spectrum_label=str(data["spectrum_label"]),
        requested_peak_count=int(data["requested_peak_count"]),
        peak_count_probabilities=combined_probabilities,
        bic_peak_count_probabilities=bic_probabilities,
        aicc_peak_count_probabilities=aicc_probabilities,
        bic_by_peak_count=integer_mapping("bic_by_peak_count", "peak-count BIC"),
        aicc_by_peak_count=integer_mapping("aicc_by_peak_count", "peak-count AICc"),
        components=tuple(components),
        requested_runs=int(data["requested_runs"]),
        successful_runs=int(data["successful_runs"]),
        failed_runs=int(data["failed_runs"]),
        paired_model_scenarios=int(data.get("paired_model_scenarios", data["requested_runs"])),
        candidate_trials_per_run=int(data["candidate_trials_per_run"]),
        random_seed=int(data["random_seed"]),
        background_models=tuple(str(value) for value in data.get("background_models", ())),
        profiles=tuple(str(value) for value in data.get("profiles", ())),
        weighting=str(data["weighting"]),
        count_scale=_optional_float(data.get("count_scale")),
        calibration_uncertainty_eV=float(data["calibration_uncertainty_eV"]),
        roi_jitter_fraction=float(data["roi_jitter_fraction"]),
        bootstrap_strength=float(data.get("bootstrap_strength", 1.0)),
        calibration_sigma_scale=float(data.get("calibration_sigma_scale", 1.0)),
        max_nfev=int(data.get("max_nfev", 900)),
        max_background_above_fraction=float(data.get("max_background_above_fraction", 0.12)),
        numerical_note=str(data["numerical_note"]),
    )


def _workspace_item_to_payload(item: WorkspaceItemData) -> dict[str, Any]:
    return {
        "spectrum": _spectrum_to_payload(item.spectrum),
        "fit_setup": {
            "background_method": item.background_method.value,
            "background_parameters": _json_safe(item.background_parameters),
            "peaks": _json_safe(item.peaks),
            "ties": _json_safe(item.ties),
        },
        "fit_result": _fit_result_to_payload(item.fit_result),
        "survey_analysis": _survey_to_payload(item.survey_analysis),
        "stability_consensus": _consensus_result_to_payload(item.consensus_result),
        "model_comparison": _model_comparison_to_payload(item.model_comparison),
    }


def _workspace_item_from_payload(data: Mapping[str, Any]) -> WorkspaceItemData:
    spectrum_data = _mapping(data.get("spectrum"), "workspace spectrum")
    fit_setup = _mapping(data.get("fit_setup"), "workspace fit setup")
    peaks = _dict_sequence(fit_setup.get("peaks", ()), "workspace peaks")
    ties = _dict_sequence(fit_setup.get("ties", ()), "workspace ties")
    parameters = _mapping(fit_setup.get("background_parameters", {}), "workspace background parameters")
    spectrum = _spectrum_from_payload(spectrum_data)
    fit_result = _fit_result_from_payload(
        None if data.get("fit_result") is None else _mapping(data.get("fit_result"), "workspace fit result")
    )
    if fit_result is not None:
        curves = (
            fit_result.background,
            fit_result.fitted_intensity,
            fit_result.residual,
            fit_result.weighted_residual,
            *fit_result.components.values(),
        )
        if any(curve.size != spectrum.size or not np.all(np.isfinite(curve)) for curve in curves):
            raise ValueError("Saved fit curves do not align with their workspace spectrum.")
    comparison = _model_comparison_from_payload(
        None
        if data.get("model_comparison") is None
        else _mapping(data.get("model_comparison"), "workspace model comparison")
    )
    if comparison is not None:
        for evaluation in comparison.candidates:
            curves = (
                evaluation.fit_result.background,
                evaluation.fit_result.fitted_intensity,
                evaluation.fit_result.residual,
                evaluation.fit_result.weighted_residual,
                *evaluation.fit_result.components.values(),
            )
            if any(curve.size != spectrum.size or not np.all(np.isfinite(curve)) for curve in curves):
                raise ValueError("Saved comparison curves do not align with their workspace spectrum.")
    return WorkspaceItemData(
        spectrum=spectrum,
        background_method=BackgroundMethod(str(fit_setup.get("background_method", BackgroundMethod.LINEAR.value))),
        background_parameters=dict(parameters),
        peaks=peaks,
        ties=ties,
        fit_result=fit_result,
        survey_analysis=_survey_from_payload(
            None if data.get("survey_analysis") is None else _mapping(data.get("survey_analysis"), "workspace survey analysis")
        ),
        consensus_result=_consensus_result_from_payload(
            None
            if data.get("stability_consensus") is None
            else _mapping(data.get("stability_consensus"), "workspace stability consensus")
        ),
        model_comparison=comparison,
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Saved {label} is invalid.")
    return value


def _dict_sequence(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"Saved {label} are invalid.")
    return tuple(dict(item) for item in value)


def save_project(
    path: str | Path,
    *,
    spectrum: Spectrum,
    background_method: BackgroundMethod | str,
    background_parameters: Mapping[str, Any] | None = None,
    peaks: Sequence[Mapping[str, Any]],
    ties: Sequence[Mapping[str, Any]] = (),
    analysis_settings: Mapping[str, Any] | None = None,
    workspace_items: Sequence[WorkspaceItemData] = (),
    active_workspace_index: int | None = None,
    global_series_result: GlobalSeriesResult | None = None,
    ui_state: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save a self-contained v10 `.xpsa` workspace project."""

    target = Path(path)
    selected = BackgroundMethod(background_method)
    items = tuple(workspace_items)
    if items:
        active_index = 0 if active_workspace_index is None else int(active_workspace_index)
        if not 0 <= active_index < len(items):
            raise ValueError("Active workspace index is outside the saved workspace.")
    else:
        active_index = 0
        items = (
            WorkspaceItemData(
                spectrum=spectrum,
                background_method=selected,
                background_parameters=dict(background_parameters or {}),
                peaks=tuple(dict(peak) for peak in peaks),
                ties=tuple(dict(tie) for tie in ties),
            ),
        )
    payload = {
        "format": "xps-afruz-project",
        "version": PROJECT_FORMAT_VERSION,
        # Retain the active-spectrum fields so the v1-v5 migration boundary
        # stays explicit and third-party readers can inspect the active setup.
        "spectrum": _spectrum_to_payload(spectrum),
        "fit": {
            "background_method": selected.value,
            "background_parameters": _json_safe(background_parameters or {}),
            "peaks": _json_safe(peaks),
            "ties": _json_safe(ties),
        },
        "analysis": _json_safe(analysis_settings or {}),
        "workspace": {
            "active_index": active_index,
            "items": [_workspace_item_to_payload(item) for item in items],
        },
        "global_series": _global_series_to_payload(global_series_result),
        "ui": _json_safe(ui_state or {}),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_project(path: str | Path) -> ProjectData:
    """Load and validate a v1-v10 XPS-master project file."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Project file is not valid JSON: {source}") from error
    if not isinstance(payload, dict) or payload.get("format") != "xps-afruz-project":
        raise ValueError("This file is not an XPS-master project.")
    version = payload.get("version")
    if version not in SUPPORTED_PROJECT_VERSIONS:
        raise ValueError(f"Unsupported project format version: {version!r}")

    spectrum_data = _mapping(payload.get("spectrum"), "project spectrum")
    fit_data = _mapping(payload.get("fit"), "project fit")
    try:
        spectrum = _spectrum_from_payload(spectrum_data)
        method = BackgroundMethod(str(fit_data.get("background_method", BackgroundMethod.LINEAR.value)))
        peaks = _dict_sequence(fit_data.get("peaks", ()), "project peaks")
        ties = _dict_sequence(fit_data.get("ties", ()), "project ties")
        background_parameters = dict(_mapping(fit_data.get("background_parameters", {}), "background parameters"))
        analysis_settings = dict(_mapping(payload.get("analysis", {}), "analysis settings"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Project spectrum, fit, or analysis settings are invalid.") from error

    workspace_payload = payload.get("workspace")
    if isinstance(workspace_payload, Mapping):
        raw_items = workspace_payload.get("items", ())
        if not isinstance(raw_items, (list, tuple)) or not all(isinstance(item, Mapping) for item in raw_items):
            raise ValueError("Project workspace items are invalid.")
        try:
            workspace = tuple(_workspace_item_from_payload(item) for item in raw_items)
            active_index = int(workspace_payload.get("active_index", 0))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Project workspace content is invalid.") from error
    else:
        workspace = ()
        active_index = 0
    if not workspace:
        workspace = (
            WorkspaceItemData(
                spectrum=spectrum,
                background_method=method,
                background_parameters=background_parameters,
                peaks=peaks,
                ties=ties,
            ),
        )
        active_index = 0
    if not 0 <= active_index < len(workspace):
        raise ValueError("Project active workspace index is invalid.")
    try:
        global_series = _global_series_from_payload(
            None
            if payload.get("global_series") is None
            else _mapping(payload.get("global_series"), "project global series")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Project global-series result is invalid.") from error
    if global_series is not None:
        for item in global_series.spectrum_fits:
            if not 0 <= item.workspace_index < len(workspace):
                raise ValueError("Saved global-series workspace index is invalid.")
            spectrum_item = workspace[item.workspace_index].spectrum
            if item.spectrum_label != spectrum_item.label:
                raise ValueError("Saved global-series label does not match its workspace spectrum.")
            curves = (
                item.global_fit.background,
                item.global_fit.fitted_intensity,
                item.global_fit.residual,
                item.global_fit.weighted_residual,
                item.independent_fit.background,
                item.independent_fit.fitted_intensity,
                item.independent_fit.residual,
                item.independent_fit.weighted_residual,
                *item.global_fit.components.values(),
                *item.independent_fit.components.values(),
            )
            if any(curve.size != spectrum_item.size or not np.all(np.isfinite(curve)) for curve in curves):
                raise ValueError("Saved global-series curves do not align with their workspace spectrum.")
    ui_state = dict(_mapping(payload.get("ui", {}), "UI state"))
    return ProjectData(
        spectrum=spectrum,
        background_method=method,
        background_parameters=background_parameters,
        peaks=peaks,
        ties=ties,
        analysis_settings=analysis_settings,
        workspace=workspace,
        active_workspace_index=active_index,
        global_series_result=global_series,
        ui_state=ui_state,
        format_version=int(version),
    )
