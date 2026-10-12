"""Residual-aware multi-start assistance for reproducible XPS peak fitting.

This module proposes numerical starting models.  It does not assign chemical
states or replace analyst review of line shapes, background, constraints, and
the selected number of components.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from ..core.spectrum import Spectrum
from .model import FitModel, FitResult, PeakComponent
from .optimizer import fit_spectrum
from .statistics import fit_information_criteria
from .peak_shapes import PeakProfile

ProgressCallback = Callable[[int, int, str], None]
CancellationCallback = Callable[[], bool]


class SmartDeconvolutionCancelled(RuntimeError):
    """Raised when a caller stops the cooperative multi-start search."""


@dataclass(frozen=True, slots=True)
class SmartDeconvolutionSettings:
    """Explicit, serializable choices for the numerical starting-model search."""

    peak_count: int
    profile: PeakProfile | str = PeakProfile.PSEUDO_VOIGT
    candidate_trials: int = 48
    max_nfev: int = 2_000
    weighting: str = "uniform"
    count_scale: float = 1.0
    random_seed: int = 41_027
    min_separation_eV: float | None = None
    center_window_eV: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", PeakProfile(self.profile))
        if not 1 <= self.peak_count <= 20:
            raise ValueError("Smart deconvolution supports 1 to 20 components.")
        if not 4 <= self.candidate_trials <= 512:
            raise ValueError("Candidate trials must be between 4 and 512.")
        if self.max_nfev < 50:
            raise ValueError("Maximum evaluations per candidate must be at least 50.")
        if self.weighting not in ("uniform", "poisson"):
            raise ValueError("Smart-deconvolution weighting must be uniform or poisson.")
        if not np.isfinite(self.count_scale) or self.count_scale <= 0.0:
            raise ValueError("Poisson count scale must be positive and finite.")
        for label, value in (("Minimum separation", self.min_separation_eV), ("Center window", self.center_window_eV)):
            if value is not None and value <= 0.0:
                raise ValueError(f"{label} must be positive when supplied.")


@dataclass(frozen=True, slots=True)
class SmartDeconvolutionResult:
    """Best candidate and transparent numerical evidence from the search."""

    fit_result: FitResult
    components: tuple[PeakComponent, ...]
    score: float
    bic: float
    residual_autocorrelation: float
    overlap_penalty: float
    candidate_trials: int
    completed_trials: int


def smart_deconvolve(
    spectrum: Spectrum,
    background: NDArray[np.float64],
    settings: SmartDeconvolutionSettings,
    *,
    progress_callback: ProgressCallback | None = None,
    cancellation_callback: CancellationCallback | None = None,
) -> SmartDeconvolutionResult:
    """Fit a user-selected number of components from diverse, ranked seeds.

    Each trial is an ordinary bounded least-squares fit.  Trials are generated
    from smoothed background-corrected features, broad alternative combinations,
    and small deterministic jitters.  Selection uses BIC plus penalties for a
    strongly structured residual and numerically collapsed components.
    """

    values = np.asarray(background, dtype=float).reshape(-1)
    if spectrum.size != values.size or not np.all(np.isfinite(values)):
        raise ValueError("The smart-deconvolution background must be finite and aligned with the spectrum.")
    energy = spectrum.energy
    signal = spectrum.intensity
    step = max(abs(float(np.median(np.diff(energy)))), 1e-9)
    energy_span = float(energy[-1] - energy[0])
    if energy_span <= step * settings.peak_count:
        raise ValueError("The selected region is too narrow for the requested component count.")

    if progress_callback is not None:
        progress_callback(0, settings.candidate_trials, "Detecting background-corrected feature candidates")
    candidate_indices, smoothed = _feature_candidates(signal - values, step, settings)
    seed_sets = _seed_sets(candidate_indices, smoothed, energy, settings)
    if not seed_sets:
        raise ValueError("Could not create distinct numerical starting positions for the requested component count.")

    best: tuple[float, float, float, float, FitResult, tuple[PeakComponent, ...]] | None = None
    completed = 0
    for trial_index, indices in enumerate(seed_sets, start=1):
        if cancellation_callback is not None and cancellation_callback():
            raise SmartDeconvolutionCancelled("Smart deconvolution was cancelled.")
        components = _components_from_indices(energy, signal, values, smoothed, indices, settings)
        model = FitModel(list(components), values, name=f"Smart {settings.peak_count}-component search")
        try:
            result = fit_spectrum(
                spectrum,
                model,
                weighting=settings.weighting,  # type: ignore[arg-type]
                count_scale=settings.count_scale,
                max_nfev=settings.max_nfev,
            )
        except ValueError:
            completed += 1
            continue
        if not result.success:
            completed += 1
            if progress_callback is not None:
                progress_callback(trial_index, len(seed_sets), f"Candidate {trial_index} did not converge; excluded")
            continue
        score, bic, autocorrelation, overlap_penalty = _score_candidate(result, len(model.free_parameters()), signal)
        if np.isfinite(score) and (best is None or score < best[0]):
            best = (score, bic, autocorrelation, overlap_penalty, result, components)
        completed += 1
        if progress_callback is not None:
            progress_callback(
                trial_index,
                len(seed_sets),
                f"Fitted candidate {trial_index}/{len(seed_sets)} • best model score is updated continuously",
            )

    if best is None:
        raise ValueError("No valid fit was produced by the smart-deconvolution search.")
    score, bic, autocorrelation, overlap_penalty, result, components = best
    return SmartDeconvolutionResult(
        fit_result=result,
        components=components,
        score=score,
        bic=bic,
        residual_autocorrelation=autocorrelation,
        overlap_penalty=overlap_penalty,
        candidate_trials=len(seed_sets),
        completed_trials=completed,
    )


def _feature_candidates(
    net_signal: NDArray[np.float64], step: float, settings: SmartDeconvolutionSettings
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return ranked local maxima, supplemented with broad coverage points."""

    sample_count = net_signal.size
    smooth_window = int(np.clip(round(0.20 / step), 5, 31))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed = np.convolve(net_signal, np.ones(smooth_window) / smooth_window, mode="same")
    min_separation = settings.min_separation_eV or max(step * 3.0, 0.12)
    distance = max(1, int(round(min_separation / step)))
    noise = float(np.median(np.abs(np.diff(smoothed)))) * 1.4826 if sample_count > 2 else 0.0
    prominence = max(float(np.ptp(smoothed)) * 0.008, noise * 2.5, np.finfo(float).eps)
    detected, _ = find_peaks(smoothed, distance=distance, prominence=prominence)
    order = detected[np.argsort(smoothed[detected])[::-1]] if detected.size else np.empty(0, dtype=int)
    required = max(settings.peak_count * 5, 12)
    broad = np.linspace(max(1, sample_count // 25), sample_count - max(2, sample_count // 25), required, dtype=int)
    merged: list[int] = []
    for index in np.concatenate((order, broad, np.array([int(np.argmax(smoothed))], dtype=int))):
        value = int(index)
        if value not in merged:
            merged.append(value)
    ranked = sorted(merged, key=lambda index: float(smoothed[index]), reverse=True)
    return np.asarray(ranked, dtype=np.int64), smoothed


def _seed_sets(
    candidate_indices: NDArray[np.int64],
    smoothed: NDArray[np.float64],
    energy: NDArray[np.float64],
    settings: SmartDeconvolutionSettings,
) -> list[tuple[int, ...]]:
    """Create deterministic diverse starting combinations instead of blind brute force."""

    if candidate_indices.size < settings.peak_count:
        return []
    minimum = settings.min_separation_eV or max(abs(float(np.median(np.diff(energy)))) * 3.0, 0.12)
    pool_size = min(candidate_indices.size, max(settings.peak_count * 2 + 4, 12), 18)
    pool = candidate_indices[:pool_size]
    combinations_ranked: list[tuple[float, tuple[int, ...]]] = []
    if comb(pool_size, settings.peak_count) <= 4_000:
        for choice in combinations((int(index) for index in pool), settings.peak_count):
            positions = np.sort(energy[list(choice)])
            if positions.size > 1 and np.any(np.diff(positions) < minimum):
                continue
            priority = float(np.sum(smoothed[list(choice)]))
            combinations_ranked.append((priority, tuple(sorted(choice))))
    combinations_ranked.sort(key=lambda item: item[0], reverse=True)
    seeds: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for _, choice in combinations_ranked:
        if choice not in seen:
            seeds.append(choice)
            seen.add(choice)
        if len(seeds) >= settings.candidate_trials:
            return seeds

    rng = np.random.default_rng(settings.random_seed)
    ranked_pool = candidate_indices[: max(settings.peak_count * 3, settings.peak_count)]
    weights = np.linspace(1.0, 0.15, ranked_pool.size)
    weights = weights / weights.sum()
    attempts = 0
    while len(seeds) < settings.candidate_trials and attempts < settings.candidate_trials * 30:
        attempts += 1
        choice = tuple(sorted(int(index) for index in rng.choice(ranked_pool, size=settings.peak_count, replace=False, p=weights)))
        positions = np.sort(energy[list(choice)])
        if choice in seen or (positions.size > 1 and np.any(np.diff(positions) < minimum)):
            continue
        seeds.append(choice)
        seen.add(choice)
    return seeds


def _components_from_indices(
    energy: NDArray[np.float64],
    signal: NDArray[np.float64],
    background: NDArray[np.float64],
    smoothed: NDArray[np.float64],
    indices: tuple[int, ...],
    settings: SmartDeconvolutionSettings,
) -> tuple[PeakComponent, ...]:
    """Turn candidate locations into finite, bounded component definitions."""

    step = max(abs(float(np.median(np.diff(energy)))), 1e-9)
    span = float(energy[-1] - energy[0])
    net = np.maximum(signal - background, 0.0)
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    total_area = max(float(integrate(net, energy)), np.finfo(float).eps)
    components: list[PeakComponent] = []
    for ordinal, index in enumerate(sorted(indices, key=lambda item: float(energy[item])), start=1):
        center = float(energy[index])
        width = _local_fwhm(energy, smoothed, int(index), fallback=max(span / (settings.peak_count * 5.0), step * 4.0))
        center_window = settings.center_window_eV or max(width * 1.8, span / (settings.peak_count * 14.0), step * 4.0)
        height = max(float(net[index]), float(np.ptp(net)) * 0.01, np.finfo(float).eps)
        area = max(height * width * np.sqrt(np.pi / (4.0 * np.log(2.0))), total_area * 1e-5)
        components.append(
            PeakComponent(
                name=f"Smart Peak {ordinal}",
                center=center,
                fwhm=width,
                area=area,
                profile=settings.profile,
                eta=0.35,
                center_bounds=(max(float(energy[0]), center - center_window), min(float(energy[-1]), center + center_window)),
                fwhm_bounds=(max(step * 1.5, 0.01), min(span * 0.85, max(width * 5.0, step * 6.0))),
                area_bounds=(0.0, total_area * 8.0),
            )
        )
    return tuple(components)


def _local_fwhm(energy: NDArray[np.float64], signal: NDArray[np.float64], index: int, *, fallback: float) -> float:
    """Estimate width from local half-height crossings on a smoothed net trace."""

    level = max(float(signal[index]), 0.0) * 0.5
    if level <= 0.0:
        return fallback
    left = index
    right = index
    while left > 0 and signal[left] > level:
        left -= 1
    while right < signal.size - 1 and signal[right] > level:
        right += 1
    width = float(energy[right] - energy[left])
    step = max(abs(float(np.median(np.diff(energy)))), 1e-9)
    return float(np.clip(width, step * 2.0, max(fallback * 3.0, step * 4.0)))


def _score_candidate(result: FitResult, parameter_count: int, observed: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Prefer parsimonious fits with non-structured residuals and distinct components."""

    residual = observed - result.fitted_intensity
    count = residual.size
    bic = fit_information_criteria(result, count, parameter_count).bic
    if count > 3 and np.std(residual[:-1]) > 0.0 and np.std(residual[1:]) > 0.0:
        autocorrelation = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    else:
        autocorrelation = 0.0
    component_values = sorted(
        (
            float(value),
            float(result.parameter_values[name.removesuffix(".center") + ".fwhm"]),
        )
        for name, value in result.parameter_values.items()
        if name.endswith(".center")
    )
    overlap_penalty = 0.0
    for (left, left_width), (right, right_width) in zip(component_values, component_values[1:], strict=False):
        separation_ratio = abs(right - left) / max((left_width + right_width) / 2.0, 1e-12)
        overlap_penalty += max(0.0, 0.28 - separation_ratio) * count * 0.35
    structured_residual_penalty = max(0.0, autocorrelation) ** 2 * count * 0.25
    return bic + structured_residual_penalty + overlap_penalty, bic, autocorrelation, overlap_penalty
