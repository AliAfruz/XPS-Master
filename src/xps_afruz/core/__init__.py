"""Core scientific data structures and explicit energy operations."""

from .energy import (
    EnergyCalibration,
    EnergyScale,
    apply_energy_calibration,
    calibration_comparison,
    charge_correct,
    convert_energy_scale,
    original_axis_in_active_scale,
    photon_energy_for_source,
)
from .spectrum import Spectrum
from .processing import (
    NoiseMetrics,
    SpikeDetection,
    combine_spectra,
    derivative_spectrum,
    despike_spectrum,
    detect_spikes,
    gaussian_smooth,
    normalize_spectrum,
    savitzky_golay_smooth,
    signal_to_noise,
    subtract_spectra,
)

__all__ = [
    "EnergyCalibration",
    "EnergyScale",
    "Spectrum",
    "NoiseMetrics",
    "SpikeDetection",
    "apply_energy_calibration",
    "calibration_comparison",
    "charge_correct",
    "convert_energy_scale",
    "combine_spectra",
    "derivative_spectrum",
    "despike_spectrum",
    "detect_spikes",
    "gaussian_smooth",
    "normalize_spectrum",
    "original_axis_in_active_scale",
    "photon_energy_for_source",
    "savitzky_golay_smooth",
    "signal_to_noise",
    "subtract_spectra",
]
