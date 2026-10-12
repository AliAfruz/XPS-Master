"""Typed contracts retained by the stability-consensus engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .peak_shapes import PeakProfile

BACKGROUND_MODES = ("current", "linear", "shirley", "tougaard_u2", "tougaard_u3")


class StabilityConsensusCancelled(RuntimeError):
    """Raised when an analyst cooperatively stops a consensus ensemble."""


@dataclass(frozen=True, slots=True)
class StabilityConsensusSettings:
    """Explicit settings for one reproducible perturbation ensemble."""

    peak_count: int
    profile: PeakProfile | str = PeakProfile.PSEUDO_VOIGT
    ensemble_runs: int = 30
    candidate_trials_per_run: int = 4
    max_nfev: int = 900
    weighting: str = "uniform"
    count_scale: float = 1.0
    compare_neighbor_counts: bool = True
    roi_jitter_fraction: float = 0.025
    bootstrap_strength: float = 1.0
    calibration_sigma_scale: float = 1.0
    random_seed: int = 73_109
    background_modes: tuple[str, ...] = BACKGROUND_MODES
    profiles: tuple[PeakProfile | str, ...] = ()
    max_background_above_fraction: float = 0.12
    minimum_success_fraction: float = 0.40

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", PeakProfile(self.profile))
        object.__setattr__(self, "background_modes", tuple(str(mode) for mode in self.background_modes))
        object.__setattr__(self, "profiles", tuple(PeakProfile(profile) for profile in self.profiles))
        if not 1 <= self.peak_count <= 20:
            raise ValueError("Consensus analysis supports 1 to 20 requested components.")
        if not 6 <= self.ensemble_runs <= 512:
            raise ValueError("Consensus ensemble runs must be between 6 and 512.")
        if not 4 <= self.candidate_trials_per_run <= 64:
            raise ValueError("Candidate trials per consensus run must be between 4 and 64.")
        if self.max_nfev < 50:
            raise ValueError("Maximum evaluations per consensus candidate must be at least 50.")
        if self.weighting not in ("uniform", "poisson"):
            raise ValueError("Consensus weighting must be uniform or poisson.")
        if not np.isfinite(self.count_scale) or self.count_scale <= 0.0:
            raise ValueError("Consensus Poisson count scale must be positive and finite.")
        if not 0.0 <= self.roi_jitter_fraction <= 0.10:
            raise ValueError("ROI jitter fraction must be within [0, 0.10].")
        if not 0.0 <= self.bootstrap_strength <= 3.0:
            raise ValueError("Bootstrap strength must be within [0, 3].")
        if self.calibration_sigma_scale < 0.0 or not np.isfinite(self.calibration_sigma_scale):
            raise ValueError("Calibration uncertainty scale must be finite and non-negative.")
        if not self.background_modes or any(mode not in BACKGROUND_MODES for mode in self.background_modes):
            raise ValueError(f"Consensus background modes must be chosen from {BACKGROUND_MODES}.")
        if not 0.0 <= self.max_background_above_fraction <= 1.0:
            raise ValueError("Maximum background-above-signal fraction must be within [0, 1].")
        if not 0.0 < self.minimum_success_fraction <= 1.0:
            raise ValueError("Minimum successful-run fraction must be within (0, 1].")

    @property
    def peak_counts(self) -> tuple[int, ...]:
        """Requested count, plus valid neighbours when comparison is enabled."""

        if not self.compare_neighbor_counts:
            return (self.peak_count,)
        return tuple(count for count in (self.peak_count - 1, self.peak_count, self.peak_count + 1) if count >= 1)

    @property
    def profiles_to_test(self) -> tuple[PeakProfile, ...]:
        """Return a conservative, unique line-shape sensitivity set."""

        requested: Sequence[PeakProfile]
        if self.profiles:
            requested = self.profiles
        else:
            requested = (self.profile, PeakProfile.GAUSSIAN, PeakProfile.PSEUDO_VOIGT, PeakProfile.VOIGT)
        unique: list[PeakProfile] = []
        for profile in requested:
            if profile not in unique:
                unique.append(profile)
        return tuple(unique)


@dataclass(frozen=True, slots=True)
class ComponentStability:
    """Aggregated evidence for one reference component cluster."""

    name: str
    sample_count: int
    survival_probability: float
    model_support_probability: float
    center_mean_eV: float
    center_std_eV: float
    fwhm_mean_eV: float
    fwhm_std_eV: float
    area_mean: float
    area_std: float
    area_cv: float
    background_sensitivity: float
    background_sensitivity_label: str
    calibration_sensitivity: float
    calibration_sensitivity_label: str
    calibration_contribution_eV: float
    boundary_hit_frequency: float
    collapse_frequency: float
    max_parameter_correlation: float
    trust_score: float
    identifiability: str


@dataclass(frozen=True, slots=True)
class StabilityConsensusResult:
    """Complete retained output of one stability-consensus analysis."""

    spectrum_label: str
    requested_peak_count: int
    peak_count_probabilities: Mapping[int, float]
    bic_peak_count_probabilities: Mapping[int, float]
    aicc_peak_count_probabilities: Mapping[int, float]
    bic_by_peak_count: Mapping[int, float]
    aicc_by_peak_count: Mapping[int, float]
    components: tuple[ComponentStability, ...]
    requested_runs: int
    successful_runs: int
    failed_runs: int
    paired_model_scenarios: int
    candidate_trials_per_run: int
    random_seed: int
    background_models: tuple[str, ...]
    profiles: tuple[str, ...]
    weighting: str
    count_scale: float | None
    calibration_uncertainty_eV: float
    roi_jitter_fraction: float
    bootstrap_strength: float
    calibration_sigma_scale: float
    max_nfev: int
    max_background_above_fraction: float
    numerical_note: str

    @property
    def preferred_peak_count(self) -> int:
        """Peak count with the largest combined BIC/AICc support weight."""

        return max(self.peak_count_probabilities, key=self.peak_count_probabilities.__getitem__)


@dataclass(frozen=True, slots=True)
class _Observation:
    center: float
    fwhm: float
    area: float
    boundary_hit: bool
    collapsed: bool


@dataclass(frozen=True, slots=True)
class _RunRecord:
    peak_count: int
    background: str
    profile: str
    calibration_shift_eV: float
    observations: tuple[_Observation, ...]
    bic: float
    aicc: float
    sample_count: int
    scenario_index: int = 0
