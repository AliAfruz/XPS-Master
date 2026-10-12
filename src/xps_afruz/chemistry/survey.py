"""Peak finding, consistency checks, and cautious matching for XPS surveys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from ..core.energy import EnergyScale
from ..core.spectrum import Spectrum
from .survey_library import REFERENCE_LIBRARY, ReferenceLine


@dataclass(frozen=True, slots=True)
class DetectedPeak:
    """A survey maximum, centroid, approximate area, and detection metrics."""

    maximum_energy: float
    centroid_energy: float
    intensity: float
    prominence: float
    fwhm_estimate: float
    background_corrected_area: float
    signal_to_noise: float
    targeted: bool = False

    @property
    def binding_energy(self) -> float:
        """Compatibility name for the active-axis maximum position."""

        return self.maximum_energy


@dataclass(frozen=True, slots=True)
class SurveyMatch:
    """One measured feature matched to one approximate reference line."""

    reference: ReferenceLine
    peak: DetectedPeak
    reference_energy: float
    energy_error: float
    score: float
    overlapping_elements: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExpectedSecondaryLine:
    """A companion feature expected for a proposed elemental candidate."""

    reference: ReferenceLine
    expected_energy: float
    observed: bool
    energy_error: float | None = None


@dataclass(frozen=True, slots=True)
class ElementCandidate:
    """A possible element ranked by line proximity, SNR, and consistency."""

    symbol: str
    element: str
    score: float
    consistency_score: float
    matches: tuple[SurveyMatch, ...]
    expected_secondary_lines: tuple[ExpectedSecondaryLine, ...]
    overlaps_with: tuple[str, ...]

    @property
    def confidence_label(self) -> str:
        """A qualitative screen label; it is not a chemical confirmation."""

        if self.matches and all(match.peak.targeted for match in self.matches):
            return "targeted trace"
        if self.score >= 75.0 and self.consistency_score >= 0.50:
            return "strong screen"
        if self.score >= 45.0:
            return "possible"
        return "tentative"


@dataclass(frozen=True, slots=True)
class SurveyAnalysis:
    """Traceable output from one survey-screening pass."""

    spectrum_label: str
    energy_range: tuple[float, float]
    energy_scale: EnergyScale
    photon_energy_eV: float | None
    tolerance: float
    noise: float
    detection_limit: float
    peaks: tuple[DetectedPeak, ...]
    matches: tuple[SurveyMatch, ...]
    candidates: tuple[ElementCandidate, ...]
    unassigned_peaks: tuple[DetectedPeak, ...]
    included_elements: tuple[str, ...]
    excluded_elements: tuple[str, ...]
    trace_searched_elements: tuple[str, ...]
    trace_recovered_elements: tuple[str, ...]
    trace_minimum_snr: float


def _energy_step(spectrum: Spectrum) -> float:
    step = float(np.median(np.diff(spectrum.energy)))
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Survey matching needs a strictly increasing energy axis.")
    return step


def _peak_measurements(
    x: np.ndarray,
    raw_y: np.ndarray,
    smooth_y: np.ndarray,
    indices: np.ndarray,
    prominences: np.ndarray,
    noise: float,
) -> tuple[DetectedPeak, ...]:
    """Estimate centroids and areas locally without treating them as fit areas."""

    _, _, left_ips, right_ips = peak_widths(smooth_y, indices, rel_height=0.5)
    output: list[DetectedPeak] = []
    for position, index in enumerate(indices):
        left_index = max(0, int(np.floor(left_ips[position])))
        right_index = min(x.size - 1, int(np.ceil(right_ips[position])))
        if right_index <= left_index:
            right_index = min(x.size - 1, left_index + 1)
        local_x = x[left_index : right_index + 1]
        local_y = raw_y[left_index : right_index + 1]
        baseline = np.interp(local_x, (local_x[0], local_x[-1]), (local_y[0], local_y[-1]))
        net = np.clip(local_y - baseline, 0.0, None)
        area = float(np.trapezoid(net, local_x))
        centroid = float(np.average(local_x, weights=net)) if float(net.sum()) > 0.0 else float(x[index])
        fwhm = float(abs(np.interp(right_ips[position], np.arange(x.size), x) - np.interp(left_ips[position], np.arange(x.size), x)))
        prominence = float(prominences[position])
        output.append(
            DetectedPeak(
                maximum_energy=float(x[index]),
                centroid_energy=centroid,
                intensity=float(raw_y[index]),
                prominence=prominence,
                fwhm_estimate=fwhm,
                background_corrected_area=area,
                signal_to_noise=float(prominence / max(noise, np.finfo(float).eps)),
            )
        )
    return tuple(output)


def detect_survey_peaks(
    spectrum: Spectrum,
    *,
    smoothing_eV: float = 0.70,
    minimum_distance_eV: float = 2.5,
    prominence_sigma: float = 5.0,
    maximum_peaks: int = 80,
) -> tuple[DetectedPeak, ...]:
    """Return reproducibly detected maxima from a wide-range XPS spectrum."""

    if smoothing_eV <= 0.0 or minimum_distance_eV <= 0.0:
        raise ValueError("Smoothing and minimum peak distance must be positive.")
    if prominence_sigma <= 0.0 or maximum_peaks < 1:
        raise ValueError("Prominence sigma and maximum peaks must be positive.")

    x, y = spectrum.energy, spectrum.intensity
    step = _energy_step(spectrum)
    sigma = max(float(smoothing_eV) / step, 0.8)
    smoothed = gaussian_filter1d(y, sigma=sigma, mode="nearest")
    noise = max(spectrum.robust_noise(), np.finfo(float).eps)
    dynamic_floor = 0.004 * max(float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5)), 0.0)
    minimum_prominence = max(float(prominence_sigma) * noise, dynamic_floor)
    distance = max(1, int(round(float(minimum_distance_eV) / step)))
    indices, properties = find_peaks(smoothed, prominence=minimum_prominence, distance=distance)
    peaks = _peak_measurements(x, y, smoothed, indices, properties["prominences"], noise)
    ranked = sorted(peaks, key=lambda peak: peak.prominence, reverse=True)[:maximum_peaks]
    return tuple(sorted(ranked, key=lambda peak: peak.maximum_energy))


def _targeted_peak_measurement(
    spectrum: Spectrum,
    reference_energy: float,
    *,
    tolerance: float,
    smoothing_eV: float,
    minimum_snr: float,
    global_noise: float,
) -> DetectedPeak | None:
    """Measure a weak peak inside one pre-declared reference window.

    This is intentionally a targeted measurement rather than unconstrained
    peak discovery.  A robust edge-to-edge local baseline is used, and the
    signal must still exceed the requested local SNR.  The returned area is a
    survey estimate for screening/RSF quantification, not a high-resolution
    fitted component area.
    """

    x, y = spectrum.energy, spectrum.intensity
    step = _energy_step(spectrum)
    half_window = max(3.0 * float(tolerance), 4.0, 5.0 * step)
    mask = (x >= reference_energy - half_window) & (x <= reference_energy + half_window)
    if int(mask.sum()) < 7:
        return None
    local_x = x[mask]
    local_y = y[mask]
    edge_count = max(2, min(local_x.size // 4, int(round(max(1.5, tolerance) / step))))
    left_level = float(np.median(local_y[:edge_count]))
    right_level = float(np.median(local_y[-edge_count:]))
    baseline = np.interp(local_x, (local_x[0], local_x[-1]), (left_level, right_level))
    net = local_y - baseline
    sigma = max(min(float(smoothing_eV), max(float(tolerance) / 2.0, step)) / step, 0.6)
    smoothed = gaussian_filter1d(net, sigma=sigma, mode="nearest")
    residual = net - smoothed
    residual_median = float(np.median(residual))
    local_noise = float(1.4826 * np.median(np.abs(residual - residual_median)))
    noise = max(float(global_noise), local_noise, np.finfo(float).eps)

    core_indices = np.flatnonzero(np.abs(local_x - reference_energy) <= tolerance)
    if core_indices.size == 0:
        return None
    index = int(core_indices[np.argmax(smoothed[core_indices])])
    prominence = float(smoothed[index])
    signal_to_noise = prominence / noise
    if prominence <= 0.0 or signal_to_noise < minimum_snr:
        return None

    integration_threshold = max(noise, 0.05 * prominence)
    left = index
    while left > 0 and smoothed[left - 1] > integration_threshold:
        left -= 1
    right = index
    while right < local_x.size - 1 and smoothed[right + 1] > integration_threshold:
        right += 1
    if right <= left:
        left = max(0, index - 1)
        right = min(local_x.size - 1, index + 1)
    area_x = local_x[left : right + 1]
    area_net = np.clip(net[left : right + 1], 0.0, None)
    area = float(np.trapezoid(area_net, area_x))
    if not np.isfinite(area) or area <= 0.0:
        return None
    centroid = float(np.average(area_x, weights=area_net)) if float(area_net.sum()) > 0.0 else float(local_x[index])

    half_height = 0.5 * prominence
    above_half = np.flatnonzero(smoothed >= half_height)
    component_half = above_half[(above_half >= left) & (above_half <= right)]
    if component_half.size >= 2:
        fwhm = float(local_x[component_half[-1]] - local_x[component_half[0]])
    else:
        fwhm = float(2.0 * step)
    return DetectedPeak(
        maximum_energy=float(local_x[index]),
        centroid_energy=centroid,
        intensity=float(local_y[index]),
        prominence=prominence,
        fwhm_estimate=max(fwhm, step),
        background_corrected_area=area,
        signal_to_noise=float(signal_to_noise),
        targeted=True,
    )


def _targeted_matches(
    spectrum: Spectrum,
    references: tuple[ReferenceLine, ...],
    existing_matches: tuple[SurveyMatch, ...],
    target_symbols: tuple[str, ...],
    *,
    tolerance: float,
    smoothing_eV: float,
    minimum_snr: float,
    global_noise: float,
) -> tuple[SurveyMatch, ...]:
    """Recover locally significant lines for explicitly targeted elements."""

    already_matched = {match.reference.symbol.lower() for match in existing_matches}
    target_set = {symbol.lower() for symbol in target_symbols}
    recovered: list[SurveyMatch] = []
    for reference in references:
        symbol_key = reference.symbol.lower()
        if symbol_key not in target_set or symbol_key in already_matched:
            continue
        expected = reference.expected_energy(spectrum.energy_scale, spectrum.photon_energy_eV)
        if expected is None or expected < spectrum.energy_range[0] or expected > spectrum.energy_range[1]:
            continue
        peak = _targeted_peak_measurement(
            spectrum,
            float(expected),
            tolerance=tolerance,
            smoothing_eV=smoothing_eV,
            minimum_snr=minimum_snr,
            global_noise=global_noise,
        )
        if peak is None:
            continue
        error = peak.maximum_energy - float(expected)
        energy_score = max(0.0, 1.0 - abs(error) / tolerance)
        snr_score = min(np.log1p(peak.signal_to_noise) / np.log(11.0), 1.0)
        # Targeted evidence is deliberately capped below an automatic
        # "possible" assignment; the analyst must still review the feature.
        score = min(44.0, 100.0 * (0.22 + 0.18 * snr_score) * (0.35 + 0.65 * energy_score) * reference.relative_strength)
        recovered.append(SurveyMatch(reference, peak, float(expected), float(error), float(score)))
    return tuple(sorted(recovered, key=lambda match: match.score, reverse=True))


def _matches_for_peaks(
    peaks: Iterable[DetectedPeak],
    references: Iterable[ReferenceLine],
    *,
    scale: EnergyScale,
    photon_energy_eV: float | None,
    tolerance: float,
) -> tuple[SurveyMatch, ...]:
    peak_list = tuple(peaks)
    maximum_prominence = max((peak.prominence for peak in peak_list), default=1.0)
    raw_matches: list[SurveyMatch] = []
    for reference in references:
        reference_energy = reference.expected_energy(scale, photon_energy_eV)
        if reference_energy is None:
            continue
        for peak in peak_list:
            error = peak.maximum_energy - reference_energy
            if abs(error) > tolerance:
                continue
            prominence_score = np.sqrt(max(peak.prominence, 0.0) / maximum_prominence)
            snr_score = min(np.log1p(peak.signal_to_noise) / np.log(31.0), 1.0)
            energy_score = max(0.0, 1.0 - abs(error) / tolerance)
            score = 100.0 * (0.15 + 0.55 * prominence_score + 0.30 * snr_score) * (0.15 + 0.85 * energy_score) * reference.relative_strength
            raw_matches.append(SurveyMatch(reference, peak, float(reference_energy), error, float(min(score, 100.0))))

    overlaps: dict[float, set[str]] = {}
    for match in raw_matches:
        overlaps.setdefault(match.peak.maximum_energy, set()).add(match.reference.symbol)
    matches = tuple(
        SurveyMatch(
            reference=match.reference,
            peak=match.peak,
            reference_energy=match.reference_energy,
            energy_error=match.energy_error,
            score=match.score,
            overlapping_elements=tuple(sorted(overlaps[match.peak.maximum_energy].difference({match.reference.symbol}))),
        )
        for match in raw_matches
    )
    return tuple(sorted(matches, key=lambda match: match.score, reverse=True))


def _candidate_for_element(
    symbol: str,
    element_matches: list[SurveyMatch],
    all_references: tuple[ReferenceLine, ...],
    all_peaks: tuple[DetectedPeak, ...],
    *,
    scale: EnergyScale,
    photon_energy_eV: float | None,
    tolerance: float,
    energy_range: tuple[float, float],
) -> ElementCandidate:
    element = element_matches[0].reference.element
    references = [reference for reference in all_references if reference.symbol == symbol]
    primary_match = element_matches[0]
    matches_by_reference = {id(match.reference): match for match in element_matches}
    secondary: list[ExpectedSecondaryLine] = []
    for reference in references:
        if id(reference) == id(primary_match.reference):
            continue
        # Multiple reference positions for the same core line are alternatives
        # across screening environments, not independent companion evidence.
        if reference.line == primary_match.reference.line:
            continue
        expected = reference.expected_energy(scale, photon_energy_eV)
        if expected is None or expected < energy_range[0] or expected > energy_range[1]:
            continue
        matched = matches_by_reference.get(id(reference))
        if matched is not None:
            secondary.append(ExpectedSecondaryLine(reference, float(expected), True, matched.energy_error))
            continue
        nearest = min(all_peaks, key=lambda peak: abs(peak.maximum_energy - expected), default=None)
        error = None if nearest is None else nearest.maximum_energy - expected
        secondary.append(ExpectedSecondaryLine(reference, float(expected), nearest is not None and abs(error) <= tolerance, error))
    observed_secondary = sum(line.observed for line in secondary)
    available_secondary = len(secondary)
    consistency = 0.0 if not available_secondary else observed_secondary / available_secondary
    base_score = element_matches[0].score
    score = min(100.0, base_score * (0.80 + 0.20 * consistency))
    overlaps = tuple(sorted({other for match in element_matches for other in match.overlapping_elements}))
    return ElementCandidate(
        symbol=symbol,
        element=element,
        score=float(score),
        consistency_score=float(consistency),
        matches=tuple(element_matches),
        expected_secondary_lines=tuple(secondary),
        overlaps_with=overlaps,
    )


def analyze_survey(
    spectrum: Spectrum,
    *,
    tolerance: float = 2.0,
    references: Iterable[ReferenceLine] = REFERENCE_LIBRARY,
    include_elements: Iterable[str] | None = None,
    exclude_elements: Iterable[str] = (),
    target_elements: Iterable[str] = (),
    trace_minimum_snr: float = 2.5,
    smoothing_eV: float = 0.70,
    minimum_distance_eV: float = 2.5,
    prominence_sigma: float = 5.0,
    maximum_peaks: int = 80,
) -> SurveyAnalysis:
    """Screen a survey for *possible* elements with consistency checking.

    Each candidate retains the observed primary line, available companion
    features, overlaps, SNR, and an explicit detection-limit proxy. Scores
    rank screening evidence only; they are not probabilities, concentrations,
    or chemical-state confirmations.
    """

    if tolerance <= 0.0:
        raise ValueError("Reference-match tolerance must be positive.")
    if trace_minimum_snr <= 0.0:
        raise ValueError("Trace-search minimum SNR must be positive.")
    all_references = tuple(references)
    canonical_symbols = {reference.symbol.lower(): reference.symbol for reference in all_references}
    if include_elements is None:
        included = tuple(dict.fromkeys(reference.symbol for reference in all_references))
    else:
        requested = {str(symbol).strip().lower() for symbol in include_elements if str(symbol).strip()}
        unknown = sorted(requested.difference(canonical_symbols))
        if unknown:
            raise ValueError(f"Unknown survey element symbol(s): {', '.join(unknown)}")
        included = tuple(symbol for symbol in dict.fromkeys(reference.symbol for reference in all_references) if symbol.lower() in requested)
    included_set = {symbol.lower() for symbol in included}
    requested_targets = {str(symbol).strip().lower() for symbol in target_elements if str(symbol).strip()}
    unknown_targets = sorted(requested_targets.difference(canonical_symbols))
    if unknown_targets:
        raise ValueError(f"Unknown targeted survey element symbol(s): {', '.join(unknown_targets)}")
    excluded = tuple(sorted({str(symbol).strip() for symbol in exclude_elements if str(symbol).strip()}))
    excluded_set = {symbol.lower() for symbol in excluded}
    reference_lines = tuple(
        reference
        for reference in all_references
        if reference.symbol.lower() in included_set and reference.symbol.lower() not in excluded_set
    )
    trace_searched = tuple(
        symbol
        for symbol in included
        if symbol.lower() in requested_targets and symbol.lower() not in excluded_set
    )
    peaks = detect_survey_peaks(
        spectrum,
        smoothing_eV=smoothing_eV,
        minimum_distance_eV=minimum_distance_eV,
        prominence_sigma=prominence_sigma,
        maximum_peaks=maximum_peaks,
    )
    automatic_matches = _matches_for_peaks(
        peaks,
        reference_lines,
        scale=spectrum.energy_scale,
        photon_energy_eV=spectrum.photon_energy_eV,
        tolerance=float(tolerance),
    )
    noise = max(spectrum.robust_noise(), np.finfo(float).eps)
    trace_matches = _targeted_matches(
        spectrum,
        reference_lines,
        automatic_matches,
        trace_searched,
        tolerance=float(tolerance),
        smoothing_eV=float(smoothing_eV),
        minimum_snr=float(trace_minimum_snr),
        global_noise=noise,
    )
    matches = tuple(sorted((*automatic_matches, *trace_matches), key=lambda match: match.score, reverse=True))
    grouped: dict[str, list[SurveyMatch]] = {}
    for match in matches:
        grouped.setdefault(match.reference.symbol, []).append(match)
    candidates = tuple(
        sorted(
            (
                _candidate_for_element(
                    symbol,
                    element_matches,
                    reference_lines,
                    peaks,
                    scale=spectrum.energy_scale,
                    photon_energy_eV=spectrum.photon_energy_eV,
                    tolerance=float(tolerance),
                    energy_range=spectrum.energy_range,
                )
                for symbol, element_matches in grouped.items()
            ),
            key=lambda candidate: candidate.score,
            reverse=True,
        )
    )
    matched_energies = {match.peak.maximum_energy for match in matches}
    trace_recovered = tuple(dict.fromkeys(match.reference.symbol for match in trace_matches))
    return SurveyAnalysis(
        spectrum_label=spectrum.label,
        energy_range=spectrum.energy_range,
        energy_scale=spectrum.energy_scale,
        photon_energy_eV=spectrum.photon_energy_eV,
        tolerance=float(tolerance),
        noise=noise,
        detection_limit=3.0 * noise,
        peaks=peaks,
        matches=matches,
        candidates=candidates,
        unassigned_peaks=tuple(peak for peak in peaks if peak.maximum_energy not in matched_energies),
        included_elements=included,
        excluded_elements=excluded,
        trace_searched_elements=trace_searched,
        trace_recovered_elements=trace_recovered,
        trace_minimum_snr=float(trace_minimum_snr),
    )
