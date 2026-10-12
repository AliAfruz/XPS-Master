"""Simultaneous, physics-constrained fitting of related XPS spectra.

The global-series engine tests one explicit hypothesis: related spectra share
selected physical line-shape parameters while retaining spectrum-specific peak
areas and, optionally, small rigid energy offsets.  It also fits the same
spectra independently and reports AICc/BIC evidence for the parameter sharing.

This is a numerical model-comparison tool.  Shared parameters do not prove a
chemical identity, and a favourable information criterion does not validate an
incorrect background, line shape, ROI, or component assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from ..core.spectrum import Spectrum
from .global_series_models import (
    GlobalComponentTrend,
    GlobalSeriesCancelled,
    GlobalSeriesResult,
    GlobalSeriesSettings,
    GlobalSpectrumFit,
)
from .model import FitModel, FitResult, ParameterTie, PeakComponent, shift_fit_result_energy
from .optimizer import fit_spectrum
from .statistics import (covariance_from_jacobian, information_criteria, pearson_residual,
                         poisson_deviance_derivative, poisson_deviance_residual, validate_poisson_spectrum)
from .peak_shapes import evaluate_peak

ProgressCallback = Callable[[int, str], None]
CancellationCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class _GlobalVariable:
    name: str
    parameter_name: str | None
    spectrum_index: int | None
    initial: float
    lower: float
    upper: float
    is_shift: bool = False


def fit_global_series(
    spectra: Sequence[Spectrum],
    backgrounds: Sequence[NDArray[np.float64]],
    components: Sequence[PeakComponent],
    ties: Sequence[ParameterTie] = (),
    *,
    workspace_indices: Sequence[int] | None = None,
    settings: GlobalSeriesSettings | None = None,
    progress_callback: ProgressCallback | None = None,
    cancellation_callback: CancellationCallback | None = None,
) -> GlobalSeriesResult:
    """Fit related spectra independently and as one shared-parameter model.

    Areas are always spectrum-specific.  Other independently variable
    parameters follow the sharing switches in ``settings``.  With shared
    centers, spectrum zero defines the reference energy and later spectra may
    receive a bounded rigid offset.  Existing :class:`ParameterTie` rules are
    resolved separately for every spectrum.
    """

    options = settings or GlobalSeriesSettings()
    series = tuple(spectra)
    baselines = tuple(np.asarray(background, dtype=float).reshape(-1) for background in backgrounds)
    hypothesis = tuple(components)
    constraints = tuple(ties)
    indices = tuple(range(len(series))) if workspace_indices is None else tuple(int(value) for value in workspace_indices)
    _validate_inputs(series, baselines, hypothesis, constraints, indices, options)

    template_model = FitModel(list(hypothesis), baselines[0], list(constraints), name="Global series template")
    dependencies = template_model.parameter_dependencies()
    if options.weighting == "poisson":
        for spectrum, background in zip(series, baselines, strict=True):
            validate_poisson_spectrum(spectrum, background)
    scaled_components = tuple(
        _scaled_components_for_spectrum(spectrum, background, hypothesis, series[0], baselines[0])
        for spectrum, background in zip(series, baselines, strict=True)
    )

    independent: list[FitResult] = []
    independent_objective = 0.0
    independent_parameter_count = 0
    for spectrum_index, (spectrum, background, seeded) in enumerate(
        zip(series, baselines, scaled_components, strict=True)
    ):
        _check_cancelled(cancellation_callback)
        _progress(
            progress_callback,
            int(round(32.0 * spectrum_index / len(series))),
            f"Independent reference fit {spectrum_index + 1}/{len(series)}: {spectrum.label}",
        )
        model = FitModel(list(seeded), background, list(constraints), name=f"Independent: {spectrum.label}")
        result = fit_spectrum(
            spectrum,
            model,
            weighting=options.weighting,  # type: ignore[arg-type]
            count_scale=options.count_scale,
            max_nfev=options.max_nfev,
        )
        if not result.success:
            raise ValueError(f"Independent reference fit did not converge for {spectrum.label!r}; revise the seeds or evaluation limit.")
        independent.append(result)
        independent_objective += result.chi_square
        independent_parameter_count += len(model.free_parameters())

    _check_cancelled(cancellation_callback)
    variables = _build_variables(template_model, scaled_components, options)
    initial = np.asarray([variable.initial for variable in variables], dtype=float)
    lower = np.asarray([variable.lower for variable in variables], dtype=float)
    upper = np.asarray([variable.upper for variable in variables], dtype=float)
    evaluation_count = 0
    last_percentage = -1

    def evaluate(vector: NDArray[np.float64]) -> tuple[
        tuple[dict[str, float], ...],
        tuple[dict[str, NDArray[np.float64]], ...],
        tuple[NDArray[np.float64], ...],
        tuple[float, ...],
    ]:
        supplied_by_spectrum = _supplied_values_by_spectrum(vector, variables, len(series))
        resolved_all: list[dict[str, float]] = []
        component_curves_all: list[dict[str, NDArray[np.float64]]] = []
        fitted_all: list[NDArray[np.float64]] = []
        shifts: list[float] = []
        for spectrum_index, (spectrum, background, supplied) in enumerate(
            zip(series, baselines, supplied_by_spectrum, strict=True)
        ):
            resolved = {name: offset if root is None else scale*supplied[root]+offset
                        for name, (root, scale, offset) in dependencies.items()}
            shift = _shift_for_spectrum(vector, variables, spectrum_index)
            if shift:
                resolved = {
                    name: value + shift if name.endswith(".center") else value
                    for name, value in resolved.items()
                }
            fitted = background.copy()
            component_curves: dict[str, NDArray[np.float64]] = {}
            for component in hypothesis:
                prefix = component.name
                curve = evaluate_peak(
                    spectrum.energy,
                    profile=component.profile,
                    center=resolved[f"{prefix}.center"],
                    fwhm=resolved[f"{prefix}.fwhm"],
                    area=resolved[f"{prefix}.area"],
                    eta=resolved[f"{prefix}.eta"],
                    asymmetry=resolved[f"{prefix}.asymmetry"],
                    instrument_fwhm=resolved[f"{prefix}.instrument_fwhm"],
                )
                component_curves[prefix] = curve
                fitted += curve
            resolved_all.append(resolved)
            component_curves_all.append(component_curves)
            fitted_all.append(fitted)
            shifts.append(shift)
        return tuple(resolved_all), tuple(component_curves_all), tuple(fitted_all), tuple(shifts)

    def residual_vector(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal evaluation_count, last_percentage
        _check_cancelled(cancellation_callback)
        evaluation_count += 1
        if evaluation_count == 1 or evaluation_count % 8 == 0:
            percentage = min(96, 32 + int(round(64.0 * evaluation_count / options.max_nfev)))
            if percentage != last_percentage:
                last_percentage = percentage
                _progress(
                    progress_callback,
                    percentage,
                    f"Simultaneous optimization: {evaluation_count} model evaluations",
                )
        _, _, fitted_all, _ = evaluate(vector)
        residuals = [
            (poisson_deviance_residual(spectrum.intensity, fitted, options.count_scale)
             if options.weighting == "poisson" else fitted-spectrum.intensity)
            for spectrum, fitted in zip(series, fitted_all, strict=True)
        ]
        return np.concatenate(residuals)

    _progress(progress_callback, 32, "Starting shared-parameter simultaneous fit")
    if variables:
        solution = least_squares(
            residual_vector,
            initial,
            bounds=(lower, upper),
            max_nfev=options.max_nfev,
            method="trf",
            jac="3-point",
            x_scale="jac",
        )
        best = np.asarray(solution.x, dtype=float)
        weighted = residual_vector(best)
        success = bool(solution.success)
        message = str(solution.message)
        optimizer_evaluations = int(solution.nfev)
        covariance_jacobian = np.asarray(solution.jac).copy()
        if options.weighting == "poisson":
            final_fitted = evaluate(best)[2]
            derivative = np.concatenate([poisson_deviance_derivative(s.intensity, f, options.count_scale)
                                         for s, f in zip(series, final_fitted, strict=True)])
            fisher_factor = np.concatenate([options.count_scale/np.sqrt(np.maximum(f*options.count_scale, 1e-12))
                                            for f in final_fitted])
            covariance_jacobian *= (fisher_factor/derivative)[:, None]
        covariance = covariance_from_jacobian(covariance_jacobian, weighted, absolute_sigma=options.weighting == "poisson")
        variable_errors = {variable.name: float(np.sqrt(max(covariance.matrix[index, index], 0.0)))
                           for index, variable in enumerate(variables)}
    else:
        best = initial
        weighted = residual_vector(best)
        success = True
        message = "Global model has no independently variable parameters."
        optimizer_evaluations = 0
        variable_errors = {}
        covariance = covariance_from_jacobian(np.zeros((sum(s.size for s in series), 0)), weighted)

    resolved_all, curves_all, fitted_all, shifts = evaluate(best)
    global_parameter_count = len(variables)
    total_samples = sum(spectrum.size for spectrum in series)
    degrees_of_freedom = max(0, total_samples - global_parameter_count)
    global_objective = float(np.dot(weighted, weighted))
    reduced_global = global_objective / degrees_of_freedom if degrees_of_freedom else float("nan")
    global_criteria = information_criteria(global_objective, total_samples, global_parameter_count, options.weighting)
    global_aicc, global_bic = global_criteria.aicc, global_criteria.bic
    independent_criteria = information_criteria(independent_objective, total_samples, independent_parameter_count, options.weighting)
    independent_aicc, independent_bic = independent_criteria.aicc, independent_criteria.bic
    delta_aicc = independent_aicc - global_aicc
    delta_bic = independent_bic - global_bic
    preference = _preference(delta_aicc, delta_bic)

    spectrum_fits: list[GlobalSpectrumFit] = []
    shared_parameter_count = sum(variable.spectrum_index is None for variable in variables)
    shared_base, shared_remainder = divmod(shared_parameter_count, len(series))
    for spectrum_index, (spectrum, background, resolved, curves, fitted, shift, independent_fit) in enumerate(
        zip(series, baselines, resolved_all, curves_all, fitted_all, shifts, independent, strict=True)
    ):
        raw_residual = fitted - spectrum.intensity
        spectrum_weighted = (poisson_deviance_residual(spectrum.intensity, fitted, options.count_scale)
                             if options.weighting == "poisson" else raw_residual)
        parameter_errors = _parameter_errors_for_spectrum(
            template_model,
            variables,
            covariance.matrix,
            spectrum_index,
            options,
        )
        local_parameter_count = sum(variable.spectrum_index == spectrum_index for variable in variables)
        # Allocate shared degrees of freedom once across the scans. Per-scan
        # reduced values are bookkeeping diagnostics; the combined statistic
        # above is the authoritative global result.
        spectrum_parameter_count = local_parameter_count + shared_base + int(spectrum_index < shared_remainder)
        spectrum_dof = max(0, spectrum.size - spectrum_parameter_count)
        objective = float(np.dot(spectrum_weighted, spectrum_weighted))
        reduced = objective / spectrum_dof if spectrum_dof else float("nan")
        pearson = pearson_residual(spectrum.intensity, fitted, options.count_scale) if options.weighting == "poisson" else None
        statistical = float(pearson @ pearson) if pearson is not None else None
        shift_name = f"s{spectrum_index}::energy_shift"
        shift_error = float(variable_errors.get(shift_name, 0.0 if spectrum_index == 0 else float("nan")))
        fit = FitResult(
            success=success,
            message=f"Global simultaneous fit: {message}",
            parameter_values=resolved,
            standard_errors=parameter_errors,
            background=background.copy(),
            components=curves,
            fitted_intensity=fitted,
            residual=raw_residual,
            weighted_residual=spectrum_weighted,
            chi_square=objective,
            reduced_chi_square=float(reduced),
            degrees_of_freedom=spectrum_dof,
            optimizer_evaluations=optimizer_evaluations,
            weighting=options.weighting,
            count_scale=float(options.count_scale) if options.weighting == "poisson" else None,
            residual_rms=float(np.sqrt(np.mean(np.square(raw_residual)))),
            statistical_chi_square=statistical,
            reduced_statistical_chi_square=statistical/spectrum_dof if statistical is not None and spectrum_dof else None,
            uncertainty_model=(
                f"Global Poisson maximum likelihood (deviance); 1 displayed unit = {options.count_scale:g} physical count(s). Expected-Fisher covariance includes shared-center/shift correlations."
                if options.weighting == "poisson"
                else "Global uniform residual weighting; objective is not a statistically calibrated chi-square. SVD covariance includes shared-center/shift correlations and is conditional on fixed backgrounds."
            ),
            objective_name="poisson_deviance" if options.weighting == "poisson" else "residual_sum_squares",
            objective_value=objective,
            free_parameter_count=spectrum_parameter_count,
            jacobian_rank=covariance.rank,
            jacobian_condition=covariance.condition,
            max_parameter_correlation=covariance.max_correlation,
            fit_warnings=(("Global model is rank-deficient; local uncertainties are undefined.",)
                          if covariance.rank < global_parameter_count else ()),
        )
        spectrum_fits.append(
            GlobalSpectrumFit(
                workspace_index=indices[spectrum_index],
                spectrum_label=spectrum.label,
                energy_shift_eV=float(shift),
                energy_shift_stderr_eV=shift_error,
                global_fit=fit,
                independent_fit=independent_fit,
            )
        )

    trends = _component_trends(
        hypothesis,
        spectrum_fits,
        template_model,
        variables,
        variable_errors,
    )
    _progress(progress_callback, 100, "Global series fit complete")
    return GlobalSeriesResult(
        settings=options,
        spectrum_fits=tuple(spectrum_fits),
        component_trends=trends,
        ties=constraints,
        success=success,
        message=message,
        optimizer_evaluations=optimizer_evaluations,
        total_samples=total_samples,
        global_parameter_count=global_parameter_count,
        independent_parameter_count=independent_parameter_count,
        degrees_of_freedom=degrees_of_freedom,
        global_objective=global_objective,
        reduced_global_objective=float(reduced_global),
        global_aicc=global_aicc,
        global_bic=global_bic,
        independent_objective=float(independent_objective),
        independent_aicc=independent_aicc,
        independent_bic=independent_bic,
        delta_aicc=float(delta_aicc),
        delta_bic=float(delta_bic),
        preference=preference,
        numerical_note=(
            "AICc/BIC compare only the tested independent and shared-parameter hypotheses. "
            "A preferred global model does not establish chemical identity and remains conditional on the ROI, "
            "backgrounds, line shapes, constraints, acquisition comparability, and analyst review."
            " Reported parameter uncertainties are local covariance approximations; the combined global degrees of "
            "freedom and reduced objective are authoritative over per-spectrum bookkeeping values."
        ),
    )


def shift_global_series_energy(result: GlobalSeriesResult, shift_eV: float) -> GlobalSeriesResult:
    """Translate a retained series after one project-wide energy calibration."""

    shift = float(shift_eV)
    if not np.isfinite(shift):
        raise ValueError("Energy shift must be finite.")
    spectrum_fits = tuple(
        replace(
            item,
            global_fit=shift_fit_result_energy(item.global_fit, shift),
            independent_fit=shift_fit_result_energy(item.independent_fit, shift),
        )
        for item in result.spectrum_fits
    )
    trends = tuple(
        replace(
            trend,
            reference_center_eV=trend.reference_center_eV + shift,
            centers_eV=tuple(value + shift for value in trend.centers_eV),
        )
        for trend in result.component_trends
    )
    shifted_ties = tuple(
        replace(
            tie,
            offset=tie.offset
            + (shift if tie.target.endswith(".center") else 0.0)
            - tie.scale * (shift if tie.source.endswith(".center") else 0.0),
        )
        for tie in result.ties
    )
    return replace(result, spectrum_fits=spectrum_fits, component_trends=trends, ties=shifted_ties)


def _validate_inputs(
    spectra: tuple[Spectrum, ...],
    backgrounds: tuple[NDArray[np.float64], ...],
    components: tuple[PeakComponent, ...],
    ties: tuple[ParameterTie, ...],
    workspace_indices: tuple[int, ...],
    settings: GlobalSeriesSettings,
) -> None:
    if len(spectra) < 2:
        raise ValueError("Global fitting requires at least two selected spectra.")
    if len(backgrounds) != len(spectra) or len(workspace_indices) != len(spectra):
        raise ValueError("Spectra, backgrounds, and workspace indices must be aligned.")
    if len(set(workspace_indices)) != len(workspace_indices) or any(index < 0 for index in workspace_indices):
        raise ValueError("Global-series workspace indices must be unique and non-negative.")
    if not components:
        raise ValueError("Global fitting requires at least one peak component.")
    scale = spectra[0].energy_scale
    for spectrum, background in zip(spectra, backgrounds, strict=True):
        if spectrum.energy_scale is not scale:
            raise ValueError("Selected spectra must use the same energy representation.")
        if background.size != spectrum.size or not np.all(np.isfinite(background)):
            raise ValueError(f"Background for {spectrum.label!r} is not finite or does not align with its spectrum.")
        if settings.weighting == "poisson" and np.any(spectrum.intensity < 0.0):
            raise ValueError("Poisson weighting requires non-negative raw count data in every selected spectrum.")
        low, high = spectrum.energy_range
        outside = [component.name for component in components if not low <= component.center <= high]
        if outside:
            raise ValueError(
                f"{spectrum.label!r} does not cover starting center(s): {', '.join(outside)}. "
                "Select spectra representing the same high-resolution region."
            )
    FitModel(list(components), backgrounds[0], list(ties), name="Global validation")
    if settings.allow_energy_shifts:
        for tie in ties:
            if tie.target.endswith(".center") and tie.source.endswith(".center") and not np.isclose(tie.scale, 1.0):
                raise ValueError("Energy-shift fitting supports center ties only when their scale is exactly 1.")


def _scaled_components_for_spectrum(
    spectrum: Spectrum,
    background: NDArray[np.float64],
    components: tuple[PeakComponent, ...],
    reference_spectrum: Spectrum,
    reference_background: NDArray[np.float64],
) -> tuple[PeakComponent, ...]:
    net = float(np.trapezoid(np.clip(spectrum.intensity - background, 0.0, None), spectrum.energy))
    reference_net = float(
        np.trapezoid(np.clip(reference_spectrum.intensity - reference_background, 0.0, None), reference_spectrum.energy)
    )
    scale = net / reference_net if reference_net > np.finfo(float).eps else 1.0
    scale = float(np.clip(scale, 1e-6, 1e6))
    seeded: list[PeakComponent] = []
    for component in components:
        area = float(np.clip(component.area * scale, component.area_bounds[0], component.area_bounds[1]))
        seeded.append(replace(component, area=area))
    return tuple(seeded)


def _build_variables(
    model: FitModel,
    component_sets: tuple[tuple[PeakComponent, ...], ...],
    settings: GlobalSeriesSettings,
) -> tuple[_GlobalVariable, ...]:
    free = model.free_parameters()
    values_per_spectrum = tuple({p.name: p for p in FitModel(list(components), np.zeros(4), list(model.ties)).free_parameters()}
                               for components in component_sets)
    variables: list[_GlobalVariable] = []
    for parameter in free:
        suffix = parameter.name.rsplit(".", 1)[1]
        shared = _parameter_is_shared(suffix, settings)
        if shared:
            variables.append(
                _GlobalVariable(
                    name=f"shared::{parameter.name}",
                    parameter_name=parameter.name,
                    spectrum_index=None,
                    initial=parameter.value,
                    lower=parameter.lower,
                    upper=parameter.upper,
                )
            )
        else:
            for spectrum_index, spectrum_parameters in enumerate(values_per_spectrum):
                seeded = spectrum_parameters[parameter.name]
                variables.append(
                    _GlobalVariable(
                        name=f"s{spectrum_index}::{parameter.name}",
                        parameter_name=parameter.name,
                        spectrum_index=spectrum_index,
                        initial=seeded.value,
                        lower=seeded.lower,
                        upper=seeded.upper,
                    )
                )
    if settings.allow_energy_shifts and settings.maximum_energy_shift_eV > 0.0:
        limit = float(settings.maximum_energy_shift_eV)
        for spectrum_index in range(1, len(component_sets)):
            variables.append(
                _GlobalVariable(
                    name=f"s{spectrum_index}::energy_shift",
                    parameter_name=None,
                    spectrum_index=spectrum_index,
                    initial=0.0,
                    lower=-limit,
                    upper=limit,
                    is_shift=True,
                )
            )
    return tuple(variables)


def _parameter_is_shared(suffix: str, settings: GlobalSeriesSettings) -> bool:
    return {
        "center": settings.share_centers,
        "fwhm": settings.share_fwhm,
        "eta": settings.share_eta,
        "asymmetry": settings.share_asymmetry,
        "instrument_fwhm": settings.share_instrument_fwhm,
        "area": False,
    }[suffix]


def _supplied_values_by_spectrum(
    vector: NDArray[np.float64], variables: tuple[_GlobalVariable, ...], spectrum_count: int
) -> tuple[dict[str, float], ...]:
    supplied = [dict() for _ in range(spectrum_count)]
    for variable, value in zip(variables, vector, strict=True):
        if variable.is_shift or variable.parameter_name is None:
            continue
        if variable.spectrum_index is None:
            for destination in supplied:
                destination[variable.parameter_name] = float(value)
        else:
            supplied[variable.spectrum_index][variable.parameter_name] = float(value)
    return tuple(supplied)


def _shift_for_spectrum(
    vector: NDArray[np.float64], variables: tuple[_GlobalVariable, ...], spectrum_index: int
) -> float:
    for variable, value in zip(variables, vector, strict=True):
        if variable.is_shift and variable.spectrum_index == spectrum_index:
            return float(value)
    return 0.0


def _parameter_errors_for_spectrum(
    model: FitModel,
    variables: tuple[_GlobalVariable, ...],
    variable_covariance: NDArray[np.float64],
    spectrum_index: int,
    settings: GlobalSeriesSettings,
) -> dict[str, float]:
    indices = {variable.name: index for index, variable in enumerate(variables)}
    errors: dict[str, float] = {}
    for name, (root, multiplier, _) in model.parameter_dependencies().items():
        gradient = np.zeros(len(variables))
        if root is not None and multiplier != 0.0:
            suffix = root.rsplit(".", 1)[1]
            key = f"shared::{root}" if _parameter_is_shared(suffix, settings) else f"s{spectrum_index}::{root}"
            gradient[indices[key]] = multiplier
        shift_key = f"s{spectrum_index}::energy_shift"
        if name.endswith(".center") and shift_key in indices:
            gradient[indices[shift_key]] += 1.0
        active = np.flatnonzero(gradient)
        if not active.size:
            errors[name] = 0.0
            continue
        variance = float(gradient[active] @ variable_covariance[np.ix_(active, active)] @ gradient[active])
        errors[name] = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else float("nan")
    return errors


def _component_trends(
    components: tuple[PeakComponent, ...],
    spectrum_fits: list[GlobalSpectrumFit],
    model: FitModel,
    variables: tuple[_GlobalVariable, ...],
    variable_errors: Mapping[str, float],
) -> tuple[GlobalComponentTrend, ...]:
    trends: list[GlobalComponentTrend] = []
    for component in components:
        prefix = component.name
        centers = tuple(float(item.global_fit.parameter_values[f"{prefix}.center"]) for item in spectrum_fits)
        center_errors = tuple(float(item.global_fit.standard_errors[f"{prefix}.center"]) for item in spectrum_fits)
        reference_center = centers[0] - spectrum_fits[0].energy_shift_eV
        reference_error = float(variable_errors.get(f"shared::{prefix}.center", center_errors[0]))
        trends.append(
            GlobalComponentTrend(
                name=prefix,
                profile=component.profile.value,
                reference_center_eV=float(reference_center),
                reference_center_stderr_eV=reference_error,
                centers_eV=centers,
                center_stderr_eV=center_errors,
                fwhm_eV=tuple(float(item.global_fit.parameter_values[f"{prefix}.fwhm"]) for item in spectrum_fits),
                fwhm_stderr_eV=tuple(float(item.global_fit.standard_errors[f"{prefix}.fwhm"]) for item in spectrum_fits),
                areas=tuple(float(item.global_fit.parameter_values[f"{prefix}.area"]) for item in spectrum_fits),
                area_stderr=tuple(float(item.global_fit.standard_errors[f"{prefix}.area"]) for item in spectrum_fits),
                eta=tuple(float(item.global_fit.parameter_values[f"{prefix}.eta"]) for item in spectrum_fits),
                asymmetry=tuple(float(item.global_fit.parameter_values[f"{prefix}.asymmetry"]) for item in spectrum_fits),
                instrument_fwhm_eV=tuple(
                    float(item.global_fit.parameter_values[f"{prefix}.instrument_fwhm"]) for item in spectrum_fits
                ),
                vary_eta=component.vary_eta,
                vary_asymmetry=component.vary_asymmetry,
                vary_instrument_fwhm=component.vary_instrument_fwhm,
            )
        )
    return tuple(trends)


def _preference(delta_aicc: float, delta_bic: float) -> str:
    if delta_aicc > 10.0 and delta_bic > 10.0:
        return "Strong numerical support for the shared-parameter global model"
    if delta_aicc > 2.0 and delta_bic > 2.0:
        return "Moderate numerical support for the shared-parameter global model"
    if delta_aicc < -10.0 and delta_bic < -10.0:
        return "Strong numerical support for independent fits"
    if delta_aicc < -2.0 and delta_bic < -2.0:
        return "Moderate numerical support for independent fits"
    return "AICc/BIC evidence is mixed or inconclusive"


def _check_cancelled(callback: CancellationCallback | None) -> None:
    if callback is not None and callback():
        raise GlobalSeriesCancelled("Global series fitting was cancelled.")


def _progress(callback: ProgressCallback | None, percentage: int, message: str) -> None:
    if callback is not None:
        callback(int(np.clip(percentage, 0, 100)), message)
