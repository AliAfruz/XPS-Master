"""Physics-constrained comparison of alternative XPS fit models.

The engine compares explicit numerical hypotheses.  It does not assign a
chemical state, decide that a component is physically real, or replace analyst
review of the ROI, background family, line shape, and parameter constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..backgrounds import (
    BackgroundDiagnostics,
    BackgroundMethod,
    BackgroundSettings,
    EdgeStatistic,
    TougaardVariant,
    diagnose_background,
    estimate_background,
)
from ..core.spectrum import Spectrum
from .model import (
    FitModel,
    FitResult,
    ParameterTie,
    PeakComponent,
    shift_fit_result_energy,
    shift_parameter_ties_energy,
)
from .optimizer import fit_spectrum
from .statistics import fit_information_criteria
from .peak_shapes import PeakProfile, evaluate_peak

ProgressCallback = Callable[[int, int, str], None]
CancellationCallback = Callable[[], bool]


class ModelComparisonCancelled(RuntimeError):
    """Raised when an analyst cooperatively stops a comparison batch."""


@dataclass(frozen=True, slots=True)
class ModelComparisonSettings:
    """Explicit controls used to construct and evaluate candidate models."""

    compare_backgrounds: bool = True
    compare_profiles: bool = True
    compare_neighbor_counts: bool = True
    weighting: str = "uniform"
    count_scale: float = 1.0
    max_nfev: int = 2_500
    max_background_above_fraction: float = 0.12

    def __post_init__(self) -> None:
        if self.weighting not in ("uniform", "poisson"):
            raise ValueError("Model-comparison weighting must be uniform or poisson.")
        if not np.isfinite(self.count_scale) or self.count_scale <= 0.0:
            raise ValueError("Model-comparison count scale must be positive and finite.")
        if self.max_nfev < 100:
            raise ValueError("Model-comparison maximum evaluations must be at least 100.")
        if not 0.0 <= self.max_background_above_fraction <= 1.0:
            raise ValueError("Maximum background-above-signal fraction must be within [0, 1].")


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """One complete, reproducible fit hypothesis."""

    identifier: str
    label: str
    components: tuple[PeakComponent, ...]
    ties: tuple[ParameterTie, ...]
    background: BackgroundSettings
    origin: str

    def __post_init__(self) -> None:
        if not self.identifier.strip() or not self.label.strip() or not self.origin.strip():
            raise ValueError("Model candidates require an identifier, label, and origin note.")
        if not self.components:
            raise ValueError("A model candidate needs at least one peak component.")
        FitModel(list(self.components), np.zeros(4), list(self.ties), name=self.label)


@dataclass(frozen=True, slots=True)
class ModelCandidateEvaluation:
    """Fit output and diagnostic evidence for one candidate."""

    candidate: ModelCandidate
    fit_result: FitResult
    background_diagnostics: BackgroundDiagnostics
    parameter_count: int
    aicc: float
    bic: float
    aicc_weight: float
    bic_weight: float
    model_support: float
    residual_autocorrelation: float
    durbin_watson: float
    boundary_hit_fraction: float
    collapse_risk: float
    max_parameter_correlation: float
    identifiability: str
    eligible: bool
    rank: int
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.parameter_count < 0 or self.rank < 0:
            raise ValueError("Candidate parameter count and rank cannot be negative.")
        for label, value in (
            ("AICc weight", self.aicc_weight),
            ("BIC weight", self.bic_weight),
            ("model support", self.model_support),
            ("boundary-hit fraction", self.boundary_hit_fraction),
            ("collapse risk", self.collapse_risk),
            ("maximum parameter correlation", self.max_parameter_correlation),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be within [0, 1].")

    @property
    def component_count(self) -> int:
        return len(self.candidate.components)


@dataclass(frozen=True, slots=True)
class ModelComparisonResult:
    """Retained alternative fits and their model-selection evidence."""

    spectrum_label: str
    candidates: tuple[ModelCandidateEvaluation, ...]
    recommended_candidate_id: str
    weighting: str
    count_scale: float | None
    numerical_note: str

    def __post_init__(self) -> None:
        if len(self.candidates) < 2:
            raise ValueError("A retained model comparison needs at least two candidates.")
        identifiers = [evaluation.candidate.identifier for evaluation in self.candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Retained model-comparison candidate identifiers must be unique.")
        if self.recommended_candidate_id not in identifiers:
            raise ValueError("The recommended candidate identifier is not retained.")
        if self.weighting not in ("uniform", "poisson"):
            raise ValueError("Retained model-comparison weighting is invalid.")
        if self.weighting == "poisson" and (
            self.count_scale is None or not np.isfinite(self.count_scale) or self.count_scale <= 0.0
        ):
            raise ValueError("Poisson model comparisons require a positive retained count scale.")
        if not self.numerical_note.strip():
            raise ValueError("A retained model comparison requires its scientific-boundary note.")

    @property
    def recommended(self) -> ModelCandidateEvaluation:
        for evaluation in self.candidates:
            if evaluation.candidate.identifier == self.recommended_candidate_id:
                return evaluation
        raise ValueError("The recommended candidate is missing from this comparison result.")

    def candidate(self, identifier: str) -> ModelCandidateEvaluation:
        for evaluation in self.candidates:
            if evaluation.candidate.identifier == identifier:
                return evaluation
        raise KeyError(identifier)


def default_background_settings(method: BackgroundMethod | str) -> BackgroundSettings:
    """Return one explicit, robust starting parameterization."""

    selected = BackgroundMethod(method)
    if selected is BackgroundMethod.LINEAR:
        parameters: dict[str, object] = {
            "edge_fraction": 0.03,
            "edge_statistic": EdgeStatistic.LOWER_QUANTILE.value,
        }
    elif selected is BackgroundMethod.SHIRLEY:
        parameters = {
            "edge_fraction": 0.03,
            "edge_statistic": EdgeStatistic.LOWER_QUANTILE.value,
            "max_iterations": 200,
            "tolerance": 1e-7,
        }
    elif selected is BackgroundMethod.TOUGAARD:
        parameters = {
            "variant": TougaardVariant.U2.value,
            "b": 2866.0,
            "c": 1643.0,
            "d": 0.0,
            "scale": None,
            "edge_fraction": 0.03,
            "edge_statistic": EdgeStatistic.LOWER_QUANTILE.value,
        }
    else:
        parameters = {}
    return BackgroundSettings(selected, parameters)


def build_standard_candidates(
    spectrum: Spectrum,
    components: Sequence[PeakComponent],
    ties: Sequence[ParameterTie],
    active_background: BackgroundSettings,
    settings: ModelComparisonSettings,
) -> tuple[ModelCandidate, ...]:
    """Build a bounded family around the analyst's current physical model.

    The current constrained model is always retained. Optional alternatives
    vary one modelling choice at a time so differences remain interpretable.
    """

    base_components = tuple(components)
    base_ties = tuple(ties)
    if not base_components:
        raise ValueError("Add at least one peak before building model candidates.")
    candidates: list[tuple[str, tuple[PeakComponent, ...], tuple[ParameterTie, ...], BackgroundSettings, str]] = [
        (
            "Analyst model",
            base_components,
            base_ties,
            active_background,
            "Exact current components, constraints, profiles, and background.",
        )
    ]

    if settings.compare_backgrounds:
        for method in (BackgroundMethod.LINEAR, BackgroundMethod.SHIRLEY, BackgroundMethod.TOUGAARD):
            background = active_background if method is active_background.method else default_background_settings(method)
            if background.method is active_background.method and dict(background.parameters) == dict(active_background.parameters):
                continue
            candidates.append(
                (
                    f"{method.value.title()} background",
                    base_components,
                    base_ties,
                    background,
                    "Current constrained peak model with one alternative background family.",
                )
            )

    if settings.compare_profiles:
        current_profiles = {component.profile for component in base_components}
        for profile in (PeakProfile.GAUSSIAN, PeakProfile.PSEUDO_VOIGT):
            if current_profiles == {profile}:
                continue
            profile_components = tuple(_component_with_profile(component, profile) for component in base_components)
            candidates.append(
                (
                    profile.display_name,
                    profile_components,
                    base_ties,
                    active_background,
                    "Current centers, constraints, and background with one mathematical line-shape family.",
                )
            )

    if settings.compare_neighbor_counts:
        if len(base_components) > 1:
            weakest = min(base_components, key=lambda component: component.area)
            reduced = tuple(component for component in base_components if component.name != weakest.name)
            reduced_names = {component.name for component in reduced}
            reduced_ties = tuple(
                tie
                for tie in base_ties
                if any(tie.target.startswith(f"{name}.") for name in reduced_names)
                and any(tie.source.startswith(f"{name}.") for name in reduced_names)
            )
            candidates.append(
                (
                    f"Parsimonious K={len(reduced)}",
                    reduced,
                    reduced_ties,
                    active_background,
                    f"Removed the smallest starting-area component ({weakest.name}) while retaining valid constraints.",
                )
            )
        expanded = base_components + (_residual_seed_component(spectrum, base_components, active_background),)
        candidates.append(
            (
                f"Expanded K={len(expanded)}",
                expanded,
                base_ties,
                active_background,
                "Added one bounded component at the largest positive unexplained feature.",
            )
        )

    unique: list[ModelCandidate] = []
    signatures: set[str] = set()
    for label, candidate_components, candidate_ties, background, origin in candidates:
        signature = _candidate_signature(candidate_components, candidate_ties, background)
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append(
            ModelCandidate(
                identifier=f"candidate-{len(unique) + 1:02d}",
                label=label,
                components=candidate_components,
                ties=candidate_ties,
                background=background,
                origin=origin,
            )
        )
    return tuple(unique)


def compare_models(
    spectrum: Spectrum,
    candidates: Sequence[ModelCandidate],
    settings: ModelComparisonSettings,
    *,
    progress_callback: ProgressCallback | None = None,
    cancellation_callback: CancellationCallback | None = None,
) -> ModelComparisonResult:
    """Fit, diagnose, and rank explicit candidates without mutating any fit."""

    hypotheses = tuple(candidates)
    if len(hypotheses) < 2:
        raise ValueError("Model comparison requires at least two distinct candidates.")
    identifiers = [candidate.identifier for candidate in hypotheses]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Model candidate identifiers must be unique.")

    preliminary: list[ModelCandidateEvaluation] = []
    failures: list[str] = []
    for index, candidate in enumerate(hypotheses, start=1):
        if cancellation_callback is not None and cancellation_callback():
            raise ModelComparisonCancelled("Model comparison was cancelled.")
        if progress_callback is not None:
            progress_callback(index - 1, len(hypotheses), f"Fitting {candidate.label}")
        try:
            background = estimate_background(
                spectrum,
                candidate.background.method,
                **dict(candidate.background.parameters),
            )
            diagnostics = diagnose_background(spectrum, background)
            model = FitModel(list(candidate.components), background, list(candidate.ties), name=candidate.label)
            fit = fit_spectrum(
                spectrum,
                model,
                weighting=settings.weighting,  # type: ignore[arg-type]
                count_scale=settings.count_scale,
                max_nfev=settings.max_nfev,
            )
            preliminary.append(_evaluate_candidate(spectrum, candidate, model, fit, diagnostics, settings))
        except (TypeError, ValueError) as error:
            failures.append(f"{candidate.label}: {error}")
        if progress_callback is not None:
            progress_callback(index, len(hypotheses), f"Completed {candidate.label}")

    if len(preliminary) < 2:
        detail = "; ".join(failures[:3])
        raise ValueError(f"Fewer than two candidate fits completed successfully. {detail}".strip())

    eligible = [evaluation for evaluation in preliminary if evaluation.eligible]
    support_pool = eligible
    aicc_weights = _criterion_weights([evaluation.aicc for evaluation in support_pool]) if support_pool else np.empty(0)
    bic_weights = _criterion_weights([evaluation.bic for evaluation in support_pool]) if support_pool else np.empty(0)
    support_by_id = {
        evaluation.candidate.identifier: (aicc_weight, bic_weight)
        for evaluation, aicc_weight, bic_weight in zip(support_pool, aicc_weights, bic_weights, strict=True)
    }
    weighted = [
        replace(
            evaluation,
            aicc_weight=support_by_id.get(evaluation.candidate.identifier, (0.0, 0.0))[0],
            bic_weight=support_by_id.get(evaluation.candidate.identifier, (0.0, 0.0))[1],
            model_support=sum(support_by_id.get(evaluation.candidate.identifier, (0.0, 0.0))) / 2.0,
        )
        for evaluation in preliminary
    ]
    ordered = sorted(
        weighted,
        key=lambda evaluation: (
            not evaluation.eligible,
            -evaluation.model_support,
            evaluation.aicc,
            evaluation.bic,
        ),
    )
    ranked = tuple(replace(evaluation, rank=index) for index, evaluation in enumerate(ordered, start=1))
    recommended = ranked[0]
    return ModelComparisonResult(
        spectrum_label=spectrum.label,
        candidates=ranked,
        recommended_candidate_id=recommended.candidate.identifier,
        weighting=settings.weighting,
        count_scale=settings.count_scale if settings.weighting == "poisson" else None,
        numerical_note=(
            ("No candidate passed the eligibility checks; all support weights are zero. " if not eligible else "") +
            "AICc/BIC support compares only the tested numerical candidates. The recommended row is not a chemical-state "
            "assignment and must be reviewed against XPS physics, acquisition conditions, background validity, and constraints."
        ),
    )


def optimized_components(evaluation: ModelCandidateEvaluation) -> tuple[PeakComponent, ...]:
    """Return the candidate definitions populated with their fitted values."""

    values = evaluation.fit_result.parameter_values
    return tuple(
        replace(
            component,
            center=float(values[f"{component.name}.center"]),
            fwhm=float(values[f"{component.name}.fwhm"]),
            area=float(values[f"{component.name}.area"]),
            eta=float(values[f"{component.name}.eta"]),
            asymmetry=float(values[f"{component.name}.asymmetry"]),
            instrument_fwhm=float(values[f"{component.name}.instrument_fwhm"]),
        )
        for component in evaluation.candidate.components
    )


def shift_model_comparison_energy(result: ModelComparisonResult, shift_eV: float) -> ModelComparisonResult:
    """Translate all retained candidate centers during a project-wide shift."""

    shift = float(shift_eV)
    if not np.isfinite(shift):
        raise ValueError("Energy shift must be finite.")
    shifted: list[ModelCandidateEvaluation] = []
    for evaluation in result.candidates:
        components = tuple(
            replace(
                component,
                center=component.center + shift,
                center_bounds=(component.center_bounds[0] + shift, component.center_bounds[1] + shift),
            )
            for component in evaluation.candidate.components
        )
        candidate = replace(
            evaluation.candidate,
            components=components,
            ties=tuple(shift_parameter_ties_energy(evaluation.candidate.ties, shift)),
        )
        shifted.append(
            replace(
                evaluation,
                candidate=candidate,
                fit_result=shift_fit_result_energy(evaluation.fit_result, shift),
            )
        )
    return replace(result, candidates=tuple(shifted))


def _component_with_profile(component: PeakComponent, profile: PeakProfile) -> PeakComponent:
    if profile is PeakProfile.GAUSSIAN:
        return replace(
            component,
            profile=profile,
            eta=0.0,
            asymmetry=0.0,
            instrument_fwhm=0.0,
            vary_eta=False,
            vary_asymmetry=False,
            vary_instrument_fwhm=False,
        )
    return replace(
        component,
        profile=profile,
        eta=float(np.clip(component.eta, 0.0, 1.0)),
        asymmetry=0.0,
        instrument_fwhm=0.0,
        vary_asymmetry=False,
        vary_instrument_fwhm=False,
    )


def _residual_seed_component(
    spectrum: Spectrum,
    components: Sequence[PeakComponent],
    background_settings: BackgroundSettings,
) -> PeakComponent:
    background = estimate_background(spectrum, background_settings.method, **dict(background_settings.parameters))
    explained = background.copy()
    for component in components:
        explained += evaluate_peak(
            spectrum.energy,
            profile=component.profile,
            center=component.center,
            fwhm=component.fwhm,
            area=component.area,
            eta=component.eta,
            asymmetry=component.asymmetry,
            instrument_fwhm=component.instrument_fwhm,
        )
    residual = spectrum.intensity - explained
    step = max(abs(float(np.median(np.diff(spectrum.energy)))), 1e-9)
    window = int(np.clip(round(0.25 / step), 5, 31))
    if window % 2 == 0:
        window += 1
    smoothed = np.convolve(residual, np.ones(window) / window, mode="same")
    allowed = np.ones(spectrum.size, dtype=bool)
    for component in components:
        allowed &= np.abs(spectrum.energy - component.center) > max(component.fwhm * 0.55, step * 3.0)
    if not np.any(allowed):
        allowed[:] = True
    index = int(np.argmax(np.where(allowed, smoothed, -np.inf)))
    center = float(spectrum.energy[index])
    span = float(spectrum.energy[-1] - spectrum.energy[0])
    width = max(step * 4.0, span / max(20.0, len(components) * 8.0))
    height = max(float(smoothed[index]), float(np.ptp(spectrum.intensity)) * 0.01, np.finfo(float).eps)
    area = height * width * np.sqrt(np.pi / (4.0 * np.log(2.0)))
    names = {component.name for component in components}
    ordinal = 1
    while f"Comparison extra {ordinal}" in names:
        ordinal += 1
    return PeakComponent(
        name=f"Comparison extra {ordinal}",
        center=center,
        fwhm=width,
        area=area,
        profile=PeakProfile.PSEUDO_VOIGT,
        eta=0.35,
        center_bounds=(max(float(spectrum.energy[0]), center - width * 2.0), min(float(spectrum.energy[-1]), center + width * 2.0)),
        fwhm_bounds=(max(step * 1.5, 0.01), max(width * 5.0, step * 4.0)),
        area_bounds=(0.0, max(area * 20.0, float(np.ptp(spectrum.intensity)) * span * 2.0)),
    )


def _candidate_signature(
    components: Sequence[PeakComponent],
    ties: Sequence[ParameterTie],
    background: BackgroundSettings,
) -> str:
    component_signature = tuple(
        (
            component.name,
            round(component.center, 10),
            round(component.fwhm, 10),
            round(component.area, 10),
            component.profile.value,
        )
        for component in components
    )
    tie_signature = tuple((tie.target, tie.source, tie.scale, tie.offset) for tie in ties)
    background_signature = (background.method.value, repr(sorted(background.parameters.items(), key=lambda item: item[0])))
    return repr((component_signature, tie_signature, background_signature))


def _evaluate_candidate(
    spectrum: Spectrum,
    candidate: ModelCandidate,
    model: FitModel,
    fit: FitResult,
    diagnostics: BackgroundDiagnostics,
    settings: ModelComparisonSettings,
) -> ModelCandidateEvaluation:
    residual = spectrum.intensity - fit.fitted_intensity
    sample_count = residual.size
    parameter_count = len(model.free_parameters())
    criteria = fit_information_criteria(fit, sample_count, parameter_count)
    aicc, bic = criteria.aicc, criteria.bic
    autocorrelation = _lag_one_correlation(residual)
    denominator = max(float(np.dot(residual, residual)), np.finfo(float).tiny)
    durbin_watson = float(np.dot(np.diff(residual), np.diff(residual)) / denominator)
    boundary_fraction = _boundary_hit_fraction(model, fit.parameter_values)
    collapse = _collapse_risk(fit.parameter_values)
    correlation = _maximum_parameter_correlation(spectrum, model, fit, settings)

    warnings: list[str] = []
    warnings.extend(fit.fit_warnings)
    if not fit.success:
        warnings.append("Optimizer did not report convergence.")
    if diagnostics.fraction_above_signal > settings.max_background_above_fraction:
        warnings.append("Background exceeds the measured signal too often for the selected acceptance limit.")
    if boundary_fraction > 0.0:
        warnings.append(f"{boundary_fraction:.0%} of free parameters are at or near a bound.")
    if collapse > 0.0:
        warnings.append("One or more components are numerically collapsed or strongly overlapping.")
    if abs(autocorrelation) > 0.35:
        warnings.append("Residuals retain strong lag-one structure.")
    if correlation > 0.95:
        warnings.append("Free parameters are strongly correlated and may not be uniquely identifiable.")
    weak_area = _weak_component_fraction(fit)
    if weak_area > 0.0:
        warnings.append(f"{weak_area:.0%} of components carry less than 0.5% of total fitted area.")

    full_rank = fit.jacobian_rank is None or fit.jacobian_rank == parameter_count
    eligible = bool(fit.success and full_rank and np.isfinite(aicc) and np.isfinite(bic)
                    and diagnostics.fraction_above_signal <= settings.max_background_above_fraction)
    if not full_rank:
        identifiability = "WEAK / NON-UNIQUE"
    elif correlation < 0.85 and boundary_fraction == 0.0 and collapse == 0.0 and abs(autocorrelation) < 0.30:
        identifiability = "STRONG NUMERICAL SUPPORT"
    elif correlation < 0.97 and boundary_fraction <= 0.25 and collapse < 0.5:
        identifiability = "MODERATE — REVIEW CORRELATIONS"
    else:
        identifiability = "WEAK / NON-UNIQUE"
    return ModelCandidateEvaluation(
        candidate=candidate,
        fit_result=fit,
        background_diagnostics=diagnostics,
        parameter_count=parameter_count,
        aicc=aicc,
        bic=bic,
        aicc_weight=0.0,
        bic_weight=0.0,
        model_support=0.0,
        residual_autocorrelation=autocorrelation,
        durbin_watson=durbin_watson,
        boundary_hit_fraction=boundary_fraction,
        collapse_risk=collapse,
        max_parameter_correlation=correlation,
        identifiability=identifiability,
        eligible=eligible,
        rank=0,
        warnings=tuple(warnings),
    )


def _criterion_weights(criteria: Sequence[float]) -> NDArray[np.float64]:
    values = np.asarray(criteria, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.size, 1.0 / values.size)
    minimum = float(np.min(values[finite]))
    raw = np.zeros(values.size, dtype=float)
    raw[finite] = np.exp(np.clip(-0.5 * (values[finite] - minimum), -745.0, 0.0))
    total = float(raw.sum())
    return raw / total if total > 0.0 else np.full(values.size, 1.0 / values.size)


def _lag_one_correlation(residual: NDArray[np.float64]) -> float:
    if residual.size < 4 or np.std(residual[:-1]) <= 0.0 or np.std(residual[1:]) <= 0.0:
        return 0.0
    return float(np.corrcoef(residual[:-1], residual[1:])[0, 1])


def _boundary_hit_fraction(model: FitModel, values: Mapping[str, float]) -> float:
    parameters = model.free_parameters()
    if not parameters:
        return 0.0
    hits = 0
    for parameter in parameters:
        value = float(values[parameter.name])
        span = parameter.upper - parameter.lower
        tolerance = max(abs(value) * 1e-5, 1e-7)
        if np.isfinite(span):
            tolerance = max(tolerance, span * 1e-4)
        if value - parameter.lower <= tolerance or parameter.upper - value <= tolerance:
            hits += 1
    return hits / len(parameters)


def _collapse_risk(values: Mapping[str, float]) -> float:
    centers = sorted(
        (
            float(value),
            float(values[name.removesuffix(".center") + ".fwhm"]),
        )
        for name, value in values.items()
        if name.endswith(".center")
    )
    if len(centers) < 2:
        return 0.0
    collapsed = 0
    for (left, left_width), (right, right_width) in zip(centers, centers[1:], strict=False):
        ratio = abs(right - left) / max((left_width + right_width) / 2.0, 1e-12)
        collapsed += ratio < 0.28
    return collapsed / (len(centers) - 1)


def _weak_component_fraction(fit: FitResult) -> float:
    areas = np.asarray(list(fit.component_areas.values()), dtype=float)
    total = float(areas.sum())
    if areas.size == 0 or total <= 0.0:
        return 0.0
    return float(np.mean(areas / total < 0.005))


def _maximum_parameter_correlation(
    spectrum: Spectrum,
    model: FitModel,
    fit: FitResult,
    settings: ModelComparisonSettings,
) -> float:
    if fit.max_parameter_correlation is not None:
        return fit.max_parameter_correlation
    free = model.free_parameters()
    if len(free) < 2:
        return 0.0
    base = dict(fit.parameter_values)
    jacobian = np.zeros((spectrum.size, len(free)), dtype=float)
    for column, parameter in enumerate(free):
        value = float(base[parameter.name])
        finite_span = parameter.upper - parameter.lower
        step = max(abs(value) * 1e-5, 1e-6)
        if np.isfinite(finite_span):
            step = max(step, finite_span * 1e-6)
        lower = max(parameter.lower, value - step)
        upper = min(parameter.upper, value + step)
        if upper <= lower:
            continue
        high = dict(base)
        low = dict(base)
        high[parameter.name] = upper
        low[parameter.name] = lower
        jacobian[:, column] = (_model_curve(spectrum, model, high) - _model_curve(spectrum, model, low)) / (upper - lower)
    if settings.weighting == "poisson":
        sigma = np.sqrt(np.maximum(spectrum.intensity * settings.count_scale, 1.0)) / settings.count_scale
        jacobian = jacobian / sigma[:, None]
    try:
        covariance = np.linalg.pinv(jacobian.T @ jacobian)
    except np.linalg.LinAlgError:
        return 1.0
    standard = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(standard, standard)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0.0)
    np.fill_diagonal(correlation, 0.0)
    return float(np.clip(np.max(np.abs(correlation)), 0.0, 1.0))


def _model_curve(spectrum: Spectrum, model: FitModel, values: Mapping[str, float]) -> NDArray[np.float64]:
    resolved = model.resolved_values(values)
    curve = model.background.copy()
    for component in model.components:
        prefix = component.name
        curve += evaluate_peak(
            spectrum.energy,
            profile=component.profile,
            center=resolved[f"{prefix}.center"],
            fwhm=resolved[f"{prefix}.fwhm"],
            area=resolved[f"{prefix}.area"],
            eta=resolved[f"{prefix}.eta"],
            asymmetry=resolved[f"{prefix}.asymmetry"],
            instrument_fwhm=resolved[f"{prefix}.instrument_fwhm"],
        )
    return curve
