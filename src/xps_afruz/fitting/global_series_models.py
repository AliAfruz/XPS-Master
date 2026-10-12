"""Validated retained-result models for global multi-spectrum fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FitResult, ParameterTie


class GlobalSeriesCancelled(RuntimeError):
    """Raised when a global-series calculation is cooperatively cancelled."""


@dataclass(frozen=True, slots=True)
class GlobalSeriesSettings:
    """Scientific and numerical controls for one simultaneous fit."""

    share_centers: bool = True
    share_fwhm: bool = True
    share_eta: bool = True
    share_asymmetry: bool = True
    share_instrument_fwhm: bool = True
    allow_energy_shifts: bool = True
    maximum_energy_shift_eV: float = 0.25
    weighting: str = "uniform"
    count_scale: float = 1.0
    max_nfev: int = 10_000

    def __post_init__(self) -> None:
        if self.weighting not in ("uniform", "poisson"):
            raise ValueError("Global-series weighting must be uniform or poisson.")
        if not np.isfinite(self.count_scale) or self.count_scale <= 0.0:
            raise ValueError("Global-series count scale must be positive and finite.")
        if not np.isfinite(self.maximum_energy_shift_eV) or self.maximum_energy_shift_eV < 0.0:
            raise ValueError("Maximum energy shift must be finite and non-negative.")
        if self.allow_energy_shifts and not self.share_centers:
            raise ValueError("Spectrum energy shifts require shared peak centers.")
        if self.max_nfev < 100:
            raise ValueError("Global-series maximum evaluations must be at least 100.")


@dataclass(frozen=True, slots=True)
class GlobalComponentTrend:
    """Per-spectrum values for one matched component in a global result."""

    name: str
    profile: str
    reference_center_eV: float
    reference_center_stderr_eV: float
    centers_eV: tuple[float, ...]
    center_stderr_eV: tuple[float, ...]
    fwhm_eV: tuple[float, ...]
    fwhm_stderr_eV: tuple[float, ...]
    areas: tuple[float, ...]
    area_stderr: tuple[float, ...]
    eta: tuple[float, ...]
    asymmetry: tuple[float, ...]
    instrument_fwhm_eV: tuple[float, ...]
    vary_eta: bool = False
    vary_asymmetry: bool = False
    vary_instrument_fwhm: bool = False

    def __post_init__(self) -> None:
        lengths = {
            len(self.centers_eV),
            len(self.center_stderr_eV),
            len(self.fwhm_eV),
            len(self.fwhm_stderr_eV),
            len(self.areas),
            len(self.area_stderr),
            len(self.eta),
            len(self.asymmetry),
            len(self.instrument_fwhm_eV),
        }
        if not self.name.strip() or not self.profile.strip() or len(lengths) != 1 or not next(iter(lengths)):
            raise ValueError("A global component trend needs a name, profile, and aligned values.")


@dataclass(frozen=True, slots=True)
class GlobalSpectrumFit:
    """Global and independent solutions retained for one workspace spectrum."""

    workspace_index: int
    spectrum_label: str
    energy_shift_eV: float
    energy_shift_stderr_eV: float
    global_fit: FitResult
    independent_fit: FitResult

    def __post_init__(self) -> None:
        if self.workspace_index < 0 or not self.spectrum_label.strip():
            raise ValueError("A global spectrum fit needs a valid workspace index and label.")
        if not np.isfinite(self.energy_shift_eV):
            raise ValueError("A retained global energy shift must be finite.")


@dataclass(frozen=True, slots=True)
class GlobalSeriesResult:
    """Complete retained evidence from one simultaneous series analysis."""

    settings: GlobalSeriesSettings
    spectrum_fits: tuple[GlobalSpectrumFit, ...]
    component_trends: tuple[GlobalComponentTrend, ...]
    ties: tuple[ParameterTie, ...]
    success: bool
    message: str
    optimizer_evaluations: int
    total_samples: int
    global_parameter_count: int
    independent_parameter_count: int
    degrees_of_freedom: int
    global_objective: float
    reduced_global_objective: float
    global_aicc: float
    global_bic: float
    independent_objective: float
    independent_aicc: float
    independent_bic: float
    delta_aicc: float
    delta_bic: float
    preference: str
    numerical_note: str

    def __post_init__(self) -> None:
        if len(self.spectrum_fits) < 2:
            raise ValueError("A global-series result needs at least two spectra.")
        if not self.component_trends:
            raise ValueError("A global-series result needs at least one matched component trend.")
        if self.total_samples < 8 or self.global_parameter_count < 0 or self.independent_parameter_count < 0:
            raise ValueError("Retained global-series dimensions are invalid.")
        if self.degrees_of_freedom != max(0, self.total_samples - self.global_parameter_count):
            raise ValueError("Retained global-series degrees of freedom are inconsistent.")
        if not self.message.strip() or not self.preference.strip() or not self.numerical_note.strip():
            raise ValueError("A global-series result requires messages and a scientific-boundary note.")
        expected = len(self.spectrum_fits)
        if any(len(trend.areas) != expected for trend in self.component_trends):
            raise ValueError("Global component trends do not align with the retained spectra.")
        if len(set(self.workspace_indices)) != expected:
            raise ValueError("Global-series workspace indices must be unique.")
        names = [trend.name for trend in self.component_trends]
        if len(set(names)) != len(names):
            raise ValueError("Global component trend names must be unique.")

    @property
    def workspace_indices(self) -> tuple[int, ...]:
        return tuple(item.workspace_index for item in self.spectrum_fits)

    @property
    def favours_global_model(self) -> bool:
        return self.delta_aicc > 0.0 and self.delta_bic > 0.0
