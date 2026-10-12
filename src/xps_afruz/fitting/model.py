"""Fit-model declarations and reusable XPS physical constraints."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from .peak_shapes import PeakProfile


@dataclass(frozen=True, slots=True)
class PeakComponent:
    """One fitted component whose intensity parameter is its integrated area.

    Eta is the Lorentzian contribution for mixed profiles. Asymmetry is an
    empirical side-width parameter except for Doniach-Sunjic, where it is the
    non-negative metallic-state tail alpha. Instrument FWHM is an optional
    Gaussian convolution width for Doniach-Sunjic.
    """

    name: str
    center: float
    fwhm: float
    area: float
    profile: PeakProfile | str = PeakProfile.PSEUDO_VOIGT
    eta: float = 0.5
    asymmetry: float = 0.0
    instrument_fwhm: float = 0.0
    center_bounds: tuple[float, float] = (-np.inf, np.inf)
    fwhm_bounds: tuple[float, float] = (0.05, 20.0)
    area_bounds: tuple[float, float] = (0.0, np.inf)
    eta_bounds: tuple[float, float] = (0.0, 1.0)
    asymmetry_bounds: tuple[float, float] = (-0.90, 0.90)
    instrument_fwhm_bounds: tuple[float, float] = (0.0, 20.0)
    vary_center: bool = True
    vary_fwhm: bool = True
    vary_area: bool = True
    vary_eta: bool = False
    vary_asymmetry: bool = False
    vary_instrument_fwhm: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Peak components need a non-empty name.")
        object.__setattr__(self, "profile", PeakProfile(self.profile))
        if self.profile is PeakProfile.DONIACH_SUNJIC and self.asymmetry_bounds[0] < 0.0:
            object.__setattr__(self, "asymmetry_bounds", (0.0, self.asymmetry_bounds[1]))
        if self.fwhm <= 0.0:
            raise ValueError("Peak FWHM must be positive.")
        if self.area < 0.0:
            raise ValueError("Peak area cannot be negative.")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("Peak eta must be within [0, 1].")
        if self.profile is PeakProfile.DONIACH_SUNJIC and not 0.0 <= self.asymmetry < 0.95:
            raise ValueError("Doniach-Sunjic asymmetry alpha must be within [0, 0.95).")
        if self.profile is not PeakProfile.DONIACH_SUNJIC and not -0.95 < self.asymmetry < 0.95:
            raise ValueError("Peak asymmetry must be within (-0.95, 0.95).")
        if self.instrument_fwhm < 0.0:
            raise ValueError("Instrument FWHM cannot be negative.")
        for value, bounds, parameter in (
            (self.center, self.center_bounds, "center"),
            (self.fwhm, self.fwhm_bounds, "fwhm"),
            (self.area, self.area_bounds, "area"),
            (self.eta, self.eta_bounds, "eta"),
            (self.asymmetry, self.asymmetry_bounds, "asymmetry"),
            (self.instrument_fwhm, self.instrument_fwhm_bounds, "instrument_fwhm"),
        ):
            _validate_bounds(parameter, value, bounds)


@dataclass(frozen=True, slots=True)
class ParameterTie:
    """A one-source linear constraint: ``target = scale * source + offset``.

    This covers common XPS rules including spin-orbit energy splitting, fixed
    doublet area ratios, and matched component widths while remaining fully
    serializable in a project file.
    """

    target: str
    source: str
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.target or not self.source:
            raise ValueError("Parameter ties need both target and source names.")
        if self.target == self.source:
            raise ValueError("A parameter cannot be tied to itself.")
        if not np.isfinite(self.scale) or not np.isfinite(self.offset):
            raise ValueError("Tie scale and offset must be finite.")


@dataclass(frozen=True, slots=True)
class FitParameter:
    """Internal normalized declaration of one scalar model parameter."""

    name: str
    value: float
    lower: float
    upper: float
    vary: bool

    def __post_init__(self) -> None:
        _validate_bounds(self.name, self.value, (self.lower, self.upper))


@dataclass(slots=True)
class FitModel:
    """A background plus area-normalized peak components and explicit ties."""

    components: list[PeakComponent]
    background: NDArray[np.float64]
    ties: list[ParameterTie] = field(default_factory=list)
    name: str = "XPS fit"

    def __post_init__(self) -> None:
        self.background = np.asarray(self.background, dtype=float).reshape(-1)
        if self.background.size < 4 or not np.all(np.isfinite(self.background)):
            raise ValueError("Fit background must be a finite array with at least four points.")
        names = [component.name for component in self.components]
        if not names or len(names) != len(set(names)):
            raise ValueError("A model needs uniquely named peak components.")
        parameters = self.parameters()
        known = set(parameters)
        targets: set[str] = set()
        for tie in self.ties:
            if tie.target not in known or tie.source not in known:
                raise ValueError(f"Tie {tie.target!r} <- {tie.source!r} refers to an unknown parameter.")
            if tie.target in targets:
                raise ValueError(f"Parameter {tie.target!r} is controlled by more than one tie.")
            targets.add(tie.target)
        _validate_tie_cycles(self.ties)

    def parameters(self) -> dict[str, FitParameter]:
        """Return every component parameter in stable insertion order."""

        parameters: dict[str, FitParameter] = {}
        for component in self.components:
            prefix = component.name
            entries = (
                ("center", component.center, component.center_bounds, component.vary_center),
                ("fwhm", component.fwhm, component.fwhm_bounds, component.vary_fwhm),
                ("area", component.area, component.area_bounds, component.vary_area),
                ("eta", component.eta, component.eta_bounds, component.vary_eta),
                ("asymmetry", component.asymmetry, component.asymmetry_bounds, component.vary_asymmetry),
                (
                    "instrument_fwhm",
                    component.instrument_fwhm,
                    component.instrument_fwhm_bounds,
                    component.vary_instrument_fwhm,
                ),
            )
            for suffix, value, bounds, vary in entries:
                name = f"{prefix}.{suffix}"
                parameters[name] = FitParameter(name, float(value), float(bounds[0]), float(bounds[1]), bool(vary))
        return parameters

    def resolved_values(self, values: Mapping[str, float]) -> dict[str, float]:
        """Apply the model's ties to supplied free-parameter values."""

        parameters = self.parameters()
        unknown = set(values).difference(parameters)
        if unknown:
            raise ValueError(f"Unknown parameter values supplied: {sorted(unknown)}")
        result = {name: float(values.get(name, parameter.value)) for name, parameter in parameters.items()}
        ties = {tie.target: tie for tie in self.ties}
        resolving: set[str] = set()

        def resolve(name: str) -> float:
            tie = ties.get(name)
            if tie is None:
                return result[name]
            if name in resolving:
                raise ValueError(f"Circular parameter tie involving {name!r}.")
            resolving.add(name)
            value = tie.scale * resolve(tie.source) + tie.offset
            resolving.remove(name)
            result[name] = value
            return value

        for name in ties:
            resolve(name)
        for name, parameter in parameters.items():
            value = result[name]
            tolerance = max(1e-10, abs(value)*1e-12)
            if not np.isfinite(value) or value < parameter.lower-tolerance or value > parameter.upper+tolerance:
                raise ValueError(f"Resolved parameter {name!r} violates its declared bounds.")
        return result

    def free_parameters(self) -> list[FitParameter]:
        """Independent parameters with all dependent tie bounds propagated."""
        parameters = self.parameters()
        tied = {tie.target for tie in self.ties}
        free = {name: parameter for name, parameter in parameters.items() if parameter.vary and name not in tied}
        bounds = {name: [parameter.lower, parameter.upper] for name, parameter in free.items()}
        for name, (root, scale, offset) in self.parameter_dependencies().items():
            target = parameters[name]
            if root is None or scale == 0.0:
                if not target.lower <= offset <= target.upper:
                    raise ValueError(f"Fixed/tied parameter {name!r} violates its declared bounds.")
                continue
            lower, upper = sorted(((target.lower-offset)/scale, (target.upper-offset)/scale))
            bounds[root][0] = max(bounds[root][0], lower)
            bounds[root][1] = min(bounds[root][1], upper)
        result = []
        for name, parameter in free.items():
            lower, upper = bounds[name]
            if lower >= upper:
                raise ValueError(f"Parameter ties leave no feasible interval for {name!r}.")
            result.append(replace(parameter, lower=lower, upper=upper, value=float(np.clip(parameter.value, lower, upper))))
        return result

    def parameter_dependencies(self) -> dict[str, tuple[str | None, float, float]]:
        """Compile every parameter to ``scale * free_root + offset``."""
        parameters = self.parameters()
        ties = {tie.target: tie for tie in self.ties}
        dependencies: dict[str, tuple[str | None, float, float]] = {}

        def resolve(name: str) -> tuple[str | None, float, float]:
            if name in dependencies:
                return dependencies[name]
            tie = ties.get(name)
            if tie is not None:
                root, scale, offset = resolve(tie.source)
                value = (root, tie.scale*scale, tie.scale*offset+tie.offset)
            elif parameters[name].vary:
                value = (name, 1.0, 0.0)
            else:
                value = (None, 0.0, parameters[name].value)
            dependencies[name] = value
            return value

        for name in parameters:
            resolve(name)
        return dependencies


@dataclass(frozen=True, slots=True)
class FitResult:
    """Complete numerical output of a reproducible fit.

    ``chi_square`` retains the optimization objective for API compatibility:
    RSS in uniform mode, Poisson deviance in Poisson mode. ``objective_name``
    labels it explicitly. ``statistical_chi_square`` separately retains the
    expected-count Pearson diagnostic; its interpretation is asymptotic.
    """

    success: bool
    message: str
    parameter_values: Mapping[str, float]
    standard_errors: Mapping[str, float]
    background: NDArray[np.float64]
    components: Mapping[str, NDArray[np.float64]]
    fitted_intensity: NDArray[np.float64]
    residual: NDArray[np.float64]
    weighted_residual: NDArray[np.float64]
    chi_square: float
    reduced_chi_square: float
    degrees_of_freedom: int
    optimizer_evaluations: int
    weighting: str
    count_scale: float | None
    residual_rms: float
    statistical_chi_square: float | None
    reduced_statistical_chi_square: float | None
    uncertainty_model: str
    objective_name: str = "weighted_least_squares"
    objective_value: float | None = None
    free_parameter_count: int | None = None
    jacobian_rank: int | None = None
    jacobian_condition: float | None = None
    max_parameter_correlation: float | None = None
    boundary_parameters: tuple[str, ...] = ()
    fit_warnings: tuple[str, ...] = ()

    @property
    def component_areas(self) -> dict[str, float]:
        """Integrated area of each fitted component."""

        return {name: float(self.parameter_values[f"{name}.area"]) for name in self.components}

    @property
    def has_calibrated_chi_square(self) -> bool:
        """Whether the fit used an explicit variance model for χ² reporting."""

        return self.statistical_chi_square is not None


def shift_fit_result_energy(result: FitResult, shift_eV: float) -> FitResult:
    """Translate fitted center coordinates without changing any intensity fit.

    A uniform calibration shift changes only the energy coordinate. Component
    curves, residuals, areas, widths, uncertainties, and fit statistics remain
    valid because their sample-by-sample intensities are unchanged.
    """

    shift = float(shift_eV)
    if not np.isfinite(shift):
        raise ValueError("Energy shift must be finite.")
    values = {
        name: float(value) + shift if name.endswith(".center") else float(value)
        for name, value in result.parameter_values.items()
    }
    return replace(result, parameter_values=values)


def shift_parameter_ties_energy(ties: Iterable[ParameterTie], shift_eV: float) -> list[ParameterTie]:
    """Translate linear constraints so they describe the same shifted model."""

    shift = float(shift_eV)
    if not np.isfinite(shift):
        raise ValueError("Energy shift must be finite.")
    translated: list[ParameterTie] = []
    for tie in ties:
        target_shift = shift if tie.target.endswith(".center") else 0.0
        source_shift = shift if tie.source.endswith(".center") else 0.0
        translated.append(
            replace(tie, offset=float(tie.offset) + target_shift - float(tie.scale) * source_shift)
        )
    return translated


def make_spin_orbit_doublet(
    label: str,
    *,
    center: float,
    fwhm: float,
    area: float,
    splitting: float,
    upper_to_lower_area_ratio: float,
    profile: PeakProfile | str = PeakProfile.PSEUDO_VOIGT,
    eta: float = 0.5,
    asymmetry: float = 0.0,
    instrument_fwhm: float = 0.0,
    center_bounds: tuple[float, float] = (-np.inf, np.inf),
    fwhm_bounds: tuple[float, float] = (0.05, 20.0),
    area_bounds: tuple[float, float] = (0.0, np.inf),
) -> tuple[list[PeakComponent], list[ParameterTie]]:
    """Create a physically linked lower-/higher-binding-energy doublet.

    ``center`` and ``area`` describe the lower-binding-energy component. The
    higher component is constrained to ``center + splitting`` and its area is
    constrained using ``upper_to_lower_area_ratio``.
    """

    if splitting <= 0.0 or upper_to_lower_area_ratio <= 0.0:
        raise ValueError("Doublet splitting and area ratio must be positive.")
    lower_name = f"{label} lower"
    upper_name = f"{label} upper"
    lower = PeakComponent(
        lower_name,
        center=center,
        fwhm=fwhm,
        area=area,
        profile=profile,
        eta=eta,
        asymmetry=asymmetry,
        instrument_fwhm=instrument_fwhm,
        center_bounds=center_bounds,
        fwhm_bounds=fwhm_bounds,
        area_bounds=area_bounds,
    )
    upper = PeakComponent(
        upper_name,
        center=center + splitting,
        fwhm=fwhm,
        area=area * upper_to_lower_area_ratio,
        profile=profile,
        eta=eta,
        asymmetry=asymmetry,
        instrument_fwhm=instrument_fwhm,
        center_bounds=(center_bounds[0] + splitting, center_bounds[1] + splitting),
        fwhm_bounds=fwhm_bounds,
        area_bounds=(area_bounds[0] * upper_to_lower_area_ratio, area_bounds[1] * upper_to_lower_area_ratio),
    )
    ties = [
        ParameterTie(f"{upper_name}.center", f"{lower_name}.center", offset=splitting),
        ParameterTie(f"{upper_name}.fwhm", f"{lower_name}.fwhm"),
        ParameterTie(f"{upper_name}.area", f"{lower_name}.area", scale=upper_to_lower_area_ratio),
        ParameterTie(f"{upper_name}.eta", f"{lower_name}.eta"),
        ParameterTie(f"{upper_name}.asymmetry", f"{lower_name}.asymmetry"),
        ParameterTie(f"{upper_name}.instrument_fwhm", f"{lower_name}.instrument_fwhm"),
    ]
    return [lower, upper], ties


def _validate_bounds(name: str, value: float, bounds: tuple[float, float]) -> None:
    lower, upper = (float(bounds[0]), float(bounds[1]))
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if lower >= upper:
        raise ValueError(f"{name} bounds must have lower < upper.")
    if not lower <= value <= upper:
        raise ValueError(f"{name}={value:g} is outside bounds [{lower:g}, {upper:g}].")


def _validate_tie_cycles(ties: Iterable[ParameterTie]) -> None:
    graph = {tie.target: tie.source for tie in ties}
    for start in graph:
        seen: set[str] = set()
        current = start
        while current in graph:
            if current in seen:
                raise ValueError("Parameter ties contain a circular dependency.")
            seen.add(current)
            current = graph[current]
