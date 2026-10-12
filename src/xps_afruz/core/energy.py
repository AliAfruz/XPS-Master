"""Energy-scale conversion and calibration with explicit provenance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .spectrum import Spectrum


class EnergyScale(str, Enum):
    """The physical meaning of the active energy coordinate."""

    BINDING = "binding"
    KINETIC = "kinetic"


@dataclass(frozen=True, slots=True)
class EnergyCalibration:
    """A reproducible energy-axis correction record."""

    shift_eV: float
    reference_label: str
    observed_energy_eV: float | None = None
    reference_energy_eV: float | None = None
    uncertainty_eV: float = 0.0
    method: str = "manual_shift"
    scope: str = "active_spectrum"
    affected_spectra: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.shift_eV):
            raise ValueError("Energy shift must be finite.")
        if self.uncertainty_eV < 0.0 or not np.isfinite(self.uncertainty_eV):
            raise ValueError("Energy uncertainty must be finite and non-negative.")
        if not self.reference_label.strip():
            raise ValueError("A calibration reference label is required.")
        if not self.scope.strip():
            raise ValueError("A calibration scope is required.")
        if self.affected_spectra is not None and self.affected_spectra < 1:
            raise ValueError("Affected-spectrum count must be positive when supplied.")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready provenance record."""

        return {
            "operation": "energy_calibration",
            "method": self.method,
            "reference_label": self.reference_label,
            "observed_energy_eV": self.observed_energy_eV,
            "reference_energy_eV": self.reference_energy_eV,
            "shift_eV": self.shift_eV,
            "uncertainty_eV": self.uncertainty_eV,
            "scope": self.scope,
            "affected_spectra": self.affected_spectra,
        }


def photon_energy_for_source(source: str) -> float:
    """Return a nominal laboratory X-ray photon energy for named sources."""

    normalized = source.strip().lower().replace(" ", "")
    sources = {
        "al", "alka", "al-kα", "al-ka", "aluminiumka", "aluminium-kα", "aluminium-ka"
    }
    if normalized in sources:
        return 1486.68
    magnesium = {"mg", "mgka", "mg-kα", "mg-ka", "magnesiumka", "magnesium-kα", "magnesium-ka"}
    if normalized in magnesium:
        return 1253.60
    raise ValueError(f"Unknown photon source: {source!r}. Supply an explicit photon energy instead.")


def convert_energy_scale(
    spectrum: "Spectrum",
    target_scale: EnergyScale | str,
    *,
    photon_energy_eV: float | None = None,
    uncertainty_eV: float = 0.0,
) -> "Spectrum":
    """Convert binding ↔ kinetic energy using an explicit photon energy.

    The imported, uncorrected axis stays stored in ``Spectrum.original_energy``
    and each conversion is appended to the spectrum provenance.
    """

    from .spectrum import Spectrum

    target = EnergyScale(target_scale)
    photon_energy = spectrum.photon_energy_eV if photon_energy_eV is None else float(photon_energy_eV)
    if photon_energy is None or not np.isfinite(photon_energy) or photon_energy <= 0.0:
        raise ValueError("A positive photon energy is required to convert energy scales.")
    if uncertainty_eV < 0.0 or not np.isfinite(uncertainty_eV):
        raise ValueError("Conversion uncertainty must be finite and non-negative.")
    if target is spectrum.energy_scale:
        return spectrum
    converted = photon_energy - spectrum.energy
    history = spectrum.correction_history + (
        {
            "operation": "energy_conversion",
            "from_scale": spectrum.energy_scale.value,
            "to_scale": target.value,
            "photon_energy_eV": photon_energy,
            "uncertainty_eV": float(uncertainty_eV),
        },
    )
    return Spectrum(
        converted,
        spectrum.intensity,
        label=spectrum.label,
        energy_unit=spectrum.energy_unit,
        intensity_unit=spectrum.intensity_unit,
        metadata=spectrum.metadata,
        energy_scale=target,
        photon_energy_eV=photon_energy,
        energy_uncertainty_eV=float(np.hypot(spectrum.energy_uncertainty_eV, uncertainty_eV)),
        original_energy=spectrum.original_energy,
        original_energy_scale=spectrum.original_energy_scale,
        correction_history=history,
        processing_history=spectrum.processing_history,
    )


def apply_energy_calibration(spectrum: "Spectrum", calibration: EnergyCalibration) -> "Spectrum":
    """Return a shifted active axis, preserving its original imported scale."""

    from .spectrum import Spectrum

    history = spectrum.correction_history + (calibration.as_dict(),)
    return Spectrum(
        spectrum.energy + calibration.shift_eV,
        spectrum.intensity,
        label=spectrum.label,
        energy_unit=spectrum.energy_unit,
        intensity_unit=spectrum.intensity_unit,
        metadata=spectrum.metadata,
        energy_scale=spectrum.energy_scale,
        photon_energy_eV=spectrum.photon_energy_eV,
        energy_uncertainty_eV=float(np.hypot(spectrum.energy_uncertainty_eV, calibration.uncertainty_eV)),
        original_energy=spectrum.original_energy,
        original_energy_scale=spectrum.original_energy_scale,
        correction_history=history,
        processing_history=spectrum.processing_history,
    )


def charge_correct(
    spectrum: "Spectrum",
    *,
    observed_energy_eV: float,
    reference_energy_eV: float,
    reference_label: str = "Charge reference",
    uncertainty_eV: float = 0.0,
) -> "Spectrum":
    """Correct a binding-energy axis from a measured reference-peak position."""

    if spectrum.energy_scale is not EnergyScale.BINDING:
        raise ValueError("Charge correction is defined on a binding-energy spectrum. Convert it first.")
    observed = float(observed_energy_eV)
    reference = float(reference_energy_eV)
    if not np.isfinite(observed) or not np.isfinite(reference):
        raise ValueError("Observed and reference energies must be finite.")
    return apply_energy_calibration(
        spectrum,
        EnergyCalibration(
            shift_eV=reference - observed,
            reference_label=reference_label,
            observed_energy_eV=observed,
            reference_energy_eV=reference,
            uncertainty_eV=uncertainty_eV,
            method="charge_correction",
        ),
    )


def calibration_comparison(spectrum: "Spectrum") -> dict[str, float | int | str]:
    """Provide concise before/after calibration facts for a UI or report."""

    original_active = original_axis_in_active_scale(spectrum)
    shift = float(np.median(spectrum.energy - original_active)) if original_active is not None else float("nan")
    return {
        "active_scale": spectrum.energy_scale.value,
        "original_scale": spectrum.original_energy_scale.value,
        "median_shift_eV": shift,
        "uncertainty_eV": spectrum.energy_uncertainty_eV,
        "operations": len(spectrum.correction_history),
    }


def original_axis_in_active_scale(spectrum: "Spectrum") -> np.ndarray | None:
    """Express the retained import axis on the current scale for comparison."""

    if spectrum.original_energy_scale is spectrum.energy_scale:
        return spectrum.original_energy.copy()
    photon_energy = spectrum.photon_energy_eV
    if photon_energy is None:
        return None
    return photon_energy - spectrum.original_energy
