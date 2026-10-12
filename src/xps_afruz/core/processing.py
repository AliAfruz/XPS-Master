"""Non-destructive, parameter-recorded spectrum processing operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter

from .spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class NoiseMetrics:
    """Robust noise and simple peak-to-noise facts for a spectrum."""

    noise: float
    signal: float
    signal_to_noise: float
    method: str = "first-difference MAD; signal above median"


@dataclass(frozen=True, slots=True)
class SpikeDetection:
    """Indices and robust threshold used for a reproducible despike operation."""

    indices: tuple[int, ...]
    threshold: float
    local_noise: float


def _derived_spectrum(
    spectrum: Spectrum,
    intensity: np.ndarray,
    *,
    operation: str,
    parameters: dict[str, object],
    label_suffix: str,
    intensity_unit: str | None = None,
) -> Spectrum:
    """Construct a transformed spectrum without altering imported coordinates."""

    return Spectrum(
        spectrum.energy,
        intensity,
        label=f"{spectrum.label} | {label_suffix}",
        energy_unit=spectrum.energy_unit,
        intensity_unit=intensity_unit or spectrum.intensity_unit,
        metadata=spectrum.metadata,
        energy_scale=spectrum.energy_scale,
        photon_energy_eV=spectrum.photon_energy_eV,
        energy_uncertainty_eV=spectrum.energy_uncertainty_eV,
        original_energy=spectrum.original_energy,
        original_energy_scale=spectrum.original_energy_scale,
        correction_history=spectrum.correction_history,
        processing_history=spectrum.processing_history + ({"operation": operation, **parameters},),
    )


def normalize_spectrum(spectrum: Spectrum, *, mode: Literal["maximum", "area", "mean"] = "maximum") -> Spectrum:
    """Return a normalized copy and record the exact normalization divisor."""

    selected = mode.lower()
    if selected == "maximum":
        divisor = float(np.max(np.abs(spectrum.intensity)))
    elif selected == "area":
        divisor = float(np.trapezoid(np.abs(spectrum.intensity), spectrum.energy))
    elif selected == "mean":
        divisor = float(np.mean(np.abs(spectrum.intensity)))
    else:
        raise ValueError("Normalization mode must be maximum, area, or mean.")
    if not np.isfinite(divisor) or divisor <= np.finfo(float).eps:
        raise ValueError("Cannot normalize a spectrum with zero signal.")
    return _derived_spectrum(
        spectrum,
        spectrum.intensity / divisor,
        operation="normalization",
        parameters={"mode": selected, "divisor": divisor},
        label_suffix=f"normalized {selected}",
        intensity_unit="normalized intensity",
    )


def gaussian_smooth(spectrum: Spectrum, *, sigma_eV: float) -> Spectrum:
    """Apply a Gaussian smoother with its physical width recorded in eV."""

    if sigma_eV <= 0.0 or not np.isfinite(sigma_eV):
        raise ValueError("Gaussian sigma must be a positive finite value in eV.")
    step = float(np.median(np.diff(spectrum.energy)))
    sigma_points = sigma_eV / step
    return _derived_spectrum(
        spectrum,
        gaussian_filter1d(spectrum.intensity, sigma=max(sigma_points, 0.1), mode="nearest"),
        operation="gaussian_smoothing",
        parameters={"sigma_eV": float(sigma_eV), "sigma_points": sigma_points},
        label_suffix=f"Gaussian σ={sigma_eV:g} eV",
    )


def savitzky_golay_smooth(
    spectrum: Spectrum,
    *,
    window_eV: float,
    polynomial_order: int = 3,
) -> Spectrum:
    """Apply a Savitzky–Golay filter using an explicit energy window."""

    if window_eV <= 0.0 or polynomial_order < 1:
        raise ValueError("Savitzky–Golay window and polynomial order must be positive.")
    step = float(np.median(np.diff(spectrum.energy)))
    window_points = max(3, int(round(window_eV / step)))
    if window_points % 2 == 0:
        window_points += 1
    if window_points > spectrum.size:
        window_points = spectrum.size if spectrum.size % 2 else spectrum.size - 1
    if window_points <= polynomial_order:
        raise ValueError("Savitzky–Golay window must exceed the polynomial order.")
    return _derived_spectrum(
        spectrum,
        savgol_filter(spectrum.intensity, window_length=window_points, polyorder=polynomial_order, mode="interp"),
        operation="savitzky_golay",
        parameters={"window_eV": float(window_eV), "window_points": window_points, "polynomial_order": polynomial_order},
        label_suffix=f"Savitzky–Golay {window_eV:g} eV",
    )


def derivative_spectrum(spectrum: Spectrum, *, order: Literal[1, 2] = 1) -> Spectrum:
    """Return a first or second numerical derivative with provenance."""

    if order not in {1, 2}:
        raise ValueError("Derivative order must be 1 or 2.")
    values = np.gradient(spectrum.intensity, spectrum.energy)
    if order == 2:
        values = np.gradient(values, spectrum.energy)
    return _derived_spectrum(
        spectrum,
        values,
        operation="derivative",
        parameters={"order": order},
        label_suffix=f"derivative {order}",
        intensity_unit=f"{spectrum.intensity_unit} / eV^{order}",
    )


def detect_spikes(spectrum: Spectrum, *, threshold_sigma: float = 6.0, median_window_points: int = 5) -> SpikeDetection:
    """Detect impulse-like outliers against a local median baseline."""

    if threshold_sigma <= 0.0 or median_window_points < 3:
        raise ValueError("Spike threshold must be positive and median window must contain at least three points.")
    window = median_window_points if median_window_points % 2 else median_window_points + 1
    local_median = median_filter(spectrum.intensity, size=window, mode="nearest")
    residual = spectrum.intensity - local_median
    local_noise = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    if local_noise <= np.finfo(float).eps:
        # Perfectly smooth synthetic/processed data can have zero MAD. In
        # that limiting case, any non-zero impulse residual is still an
        # unambiguous spike; retain the decision rather than silently skip it.
        indices = tuple(int(index) for index in np.flatnonzero(np.abs(residual) > np.finfo(float).eps))
        return SpikeDetection(indices, threshold_sigma, local_noise)
    indices = tuple(int(index) for index in np.flatnonzero(np.abs(residual) > threshold_sigma * local_noise))
    return SpikeDetection(indices, threshold_sigma, local_noise)


def despike_spectrum(
    spectrum: Spectrum,
    *,
    threshold_sigma: float = 6.0,
    median_window_points: int = 5,
) -> tuple[Spectrum, SpikeDetection]:
    """Replace only flagged impulse outliers and return the detection record."""

    detection = detect_spikes(spectrum, threshold_sigma=threshold_sigma, median_window_points=median_window_points)
    window = median_window_points if median_window_points % 2 else median_window_points + 1
    filtered = median_filter(spectrum.intensity, size=window, mode="nearest")
    corrected = spectrum.intensity.copy()
    corrected[list(detection.indices)] = filtered[list(detection.indices)]
    return (
        _derived_spectrum(
            spectrum,
            corrected,
            operation="despike",
            parameters={
                "threshold_sigma": threshold_sigma,
                "median_window_points": window,
                "replaced_indices": list(detection.indices),
                "local_noise": detection.local_noise,
            },
            label_suffix=f"despiked ({len(detection.indices)})",
        ),
        detection,
    )


def signal_to_noise(spectrum: Spectrum) -> NoiseMetrics:
    """Compute a robust noise estimate and signal above the median baseline."""

    noise = max(spectrum.robust_noise(), np.finfo(float).eps)
    signal = float(np.max(spectrum.intensity) - np.median(spectrum.intensity))
    return NoiseMetrics(noise=noise, signal=signal, signal_to_noise=signal / noise)


def _common_grid(spectra: Iterable[Spectrum], *, step_eV: float | None = None) -> tuple[tuple[Spectrum, ...], np.ndarray]:
    sources = tuple(spectra)
    if len(sources) < 2:
        raise ValueError("At least two spectra are required for a combined operation.")
    if len({spectrum.energy_scale for spectrum in sources}) != 1:
        raise ValueError("Combined spectra must use the same energy representation.")
    lower = max(spectrum.energy_range[0] for spectrum in sources)
    upper = min(spectrum.energy_range[1] for spectrum in sources)
    if upper <= lower:
        raise ValueError("Spectra have no overlapping energy range.")
    step = float(step_eV) if step_eV is not None else max(float(np.median(np.diff(spectrum.energy))) for spectrum in sources)
    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("Common-grid step must be positive and finite.")
    grid = np.arange(lower, upper + step * 0.25, step)
    if grid.size < 4:
        raise ValueError("The overlapping region is too small for a derived spectrum.")
    return sources, grid


def combine_spectra(
    spectra: Iterable[Spectrum],
    *,
    operation: Literal["average", "add"] = "average",
    step_eV: float | None = None,
) -> Spectrum:
    """Average or add replicate spectra after interpolation on a common grid."""

    sources, grid = _common_grid(spectra, step_eV=step_eV)
    if operation not in {"average", "add"}:
        raise ValueError("Combined operation must be average or add.")
    values = np.vstack([np.interp(grid, source.energy, source.intensity) for source in sources])
    combined = values.mean(axis=0) if operation == "average" else values.sum(axis=0)
    return Spectrum(
        grid,
        combined,
        label=f"{operation.title()} of {len(sources)} spectra",
        energy_unit=sources[0].energy_unit,
        intensity_unit=sources[0].intensity_unit,
        metadata={"derived_from": [source.label for source in sources]},
        energy_scale=sources[0].energy_scale,
        photon_energy_eV=sources[0].photon_energy_eV,
        original_energy=grid,
        original_energy_scale=sources[0].energy_scale,
        processing_history=(
            {"operation": f"spectrum_{operation}", "sources": [source.label for source in sources], "step_eV": float(np.median(np.diff(grid)))},
        ),
    )


def subtract_spectra(minuend: Spectrum, subtrahend: Spectrum, *, step_eV: float | None = None) -> Spectrum:
    """Calculate a reproducible difference spectrum on an overlapping grid."""

    sources, grid = _common_grid((minuend, subtrahend), step_eV=step_eV)
    result = np.interp(grid, sources[0].energy, sources[0].intensity) - np.interp(grid, sources[1].energy, sources[1].intensity)
    return Spectrum(
        grid,
        result,
        label=f"Difference: {minuend.label} − {subtrahend.label}",
        energy_unit=minuend.energy_unit,
        intensity_unit=minuend.intensity_unit,
        metadata={"minuend": minuend.label, "subtrahend": subtrahend.label},
        energy_scale=minuend.energy_scale,
        photon_energy_eV=minuend.photon_energy_eV,
        original_energy=grid,
        original_energy_scale=minuend.energy_scale,
        processing_history=(
            {
                "operation": "spectrum_subtraction",
                "minuend": minuend.label,
                "subtrahend": subtrahend.label,
                "step_eV": float(np.median(np.diff(grid))),
            },
        ),
    )
