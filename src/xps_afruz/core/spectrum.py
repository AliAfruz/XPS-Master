"""Immutable-friendly representation of one XPS spectrum."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .energy import EnergyScale


@dataclass(frozen=True, slots=True)
class Spectrum:
    """One-dimensional XPS signal indexed by binding or kinetic energy.

    The energy axis is stored in ascending order internally, even when an
    instrument exports a conventional descending binding-energy axis. Counts
    are reordered with it. This gives every numerical routine one predictable
    coordinate convention.
    """

    energy: ArrayLike
    intensity: ArrayLike
    label: str = "Untitled spectrum"
    energy_unit: str = "eV"
    intensity_unit: str = "counts"
    metadata: Mapping[str, object] = field(default_factory=dict)
    energy_scale: EnergyScale | str = EnergyScale.BINDING
    photon_energy_eV: float | None = None
    energy_uncertainty_eV: float = 0.0
    original_energy: ArrayLike | None = None
    original_energy_scale: EnergyScale | str | None = None
    correction_history: tuple[Mapping[str, object], ...] = ()
    processing_history: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        energy = np.asarray(self.energy, dtype=float).reshape(-1)
        intensity = np.asarray(self.intensity, dtype=float).reshape(-1)
        original_energy = energy.copy() if self.original_energy is None else np.asarray(self.original_energy, dtype=float).reshape(-1)
        if energy.size < 4:
            raise ValueError("A spectrum needs at least four samples.")
        if energy.size != intensity.size:
            raise ValueError("Energy and intensity must have the same length.")
        if original_energy.size != energy.size:
            raise ValueError("Original energy and active energy must have the same length.")
        if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(intensity)) or not np.all(np.isfinite(original_energy)):
            raise ValueError("Energy and intensity values must be finite.")

        order = np.argsort(energy)
        energy = energy[order]
        intensity = intensity[order]
        original_energy = original_energy[order]
        if np.any(np.diff(energy) <= 0):
            raise ValueError("Energy values must be unique.")
        energy_scale = EnergyScale(self.energy_scale)
        original_scale = energy_scale if self.original_energy_scale is None else EnergyScale(self.original_energy_scale)
        photon_energy = None if self.photon_energy_eV is None else float(self.photon_energy_eV)
        if photon_energy is not None and (not np.isfinite(photon_energy) or photon_energy <= 0.0):
            raise ValueError("Photon energy must be positive and finite when supplied.")
        uncertainty = float(self.energy_uncertainty_eV)
        if uncertainty < 0.0 or not np.isfinite(uncertainty):
            raise ValueError("Energy uncertainty must be finite and non-negative.")
        history = tuple(dict(entry) for entry in self.correction_history)
        processing_history = tuple(dict(entry) for entry in self.processing_history)

        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "original_energy", original_energy)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "energy_scale", energy_scale)
        object.__setattr__(self, "original_energy_scale", original_scale)
        object.__setattr__(self, "photon_energy_eV", photon_energy)
        object.__setattr__(self, "energy_uncertainty_eV", uncertainty)
        object.__setattr__(self, "correction_history", history)
        object.__setattr__(self, "processing_history", processing_history)

    @property
    def size(self) -> int:
        """Number of spectral channels."""

        return int(self.energy.size)

    @property
    def energy_range(self) -> tuple[float, float]:
        """Minimum and maximum energy, in eV by default."""

        return float(self.energy[0]), float(self.energy[-1])

    def region(self, minimum: float, maximum: float, *, label: str | None = None) -> "Spectrum":
        """Return an inclusive energy region.

        Parameters are accepted in either order to keep UI selection code
        simple. At least four samples are required for a fit-ready region.
        """

        low, high = sorted((float(minimum), float(maximum)))
        mask = (self.energy >= low) & (self.energy <= high)
        if int(mask.sum()) < 4:
            raise ValueError("The selected region must contain at least four samples.")
        return Spectrum(
            self.energy[mask],
            self.intensity[mask],
            label=label or f"{self.label} [{low:g}, {high:g} {self.energy_unit}]",
            energy_unit=self.energy_unit,
            intensity_unit=self.intensity_unit,
            metadata=self.metadata,
            energy_scale=self.energy_scale,
            photon_energy_eV=self.photon_energy_eV,
            energy_uncertainty_eV=self.energy_uncertainty_eV,
            original_energy=self.original_energy[mask],
            original_energy_scale=self.original_energy_scale,
            correction_history=self.correction_history,
            processing_history=self.processing_history
            + ({"operation": "region_of_interest", "minimum_eV": low, "maximum_eV": high},),
        )

    def with_intensity(
        self,
        intensity: ArrayLike,
        *,
        label: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "Spectrum":
        """Create a spectrum sharing this energy axis with replacement data."""

        return Spectrum(
            self.energy,
            intensity,
            label=label or self.label,
            energy_unit=self.energy_unit,
            intensity_unit=self.intensity_unit,
            metadata=self.metadata if metadata is None else metadata,
            energy_scale=self.energy_scale,
            photon_energy_eV=self.photon_energy_eV,
            energy_uncertainty_eV=self.energy_uncertainty_eV,
            original_energy=self.original_energy,
            original_energy_scale=self.original_energy_scale,
            correction_history=self.correction_history,
            processing_history=self.processing_history,
        )

    def robust_noise(self) -> float:
        """Estimate point-to-point noise via a robust first-difference MAD."""

        differences = np.diff(self.intensity)
        median = np.median(differences)
        mad = np.median(np.abs(differences - median))
        return float(1.4826 * mad / np.sqrt(2.0))

    def as_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return energy and intensity arrays for numerical libraries."""

        return self.energy.copy(), self.intensity.copy()
