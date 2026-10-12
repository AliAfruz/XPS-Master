"""Scaled, constrained Gaussian/Poisson-likelihood fitting for XPS models."""
from __future__ import annotations
from typing import Literal
import numpy as np
from scipy.optimize import least_squares
from ..core.spectrum import Spectrum
from .model import FitModel, FitResult
from .evaluator import ModelEvaluator
from .statistics import (covariance_from_jacobian, pearson_residual, poisson_deviance_residual,
                         poisson_deviance_derivative, validate_poisson_spectrum)

Weighting = Literal['uniform', 'poisson']


def fit_spectrum(spectrum: Spectrum, model: FitModel, *, weighting: Weighting = 'uniform',
                 count_scale: float = 1.0, max_nfev: int = 10_000) -> FitResult:
    """Fit fixed-background components with explicit noise and constraints.

    Uniform mode minimizes RSS. Poisson mode minimizes count-scale-aware
    Poisson deviance, including zero-count bins, and reports Pearson chi-square
    separately as an asymptotic diagnostic. Covariance is conditional on the
    selected background and profile; it does not include model-choice error.
    """
    if spectrum.size != model.background.size:
        raise ValueError('Spectrum and model background must have the same number of samples.')
    if weighting not in ('uniform', 'poisson'):
        raise ValueError("weighting must be 'uniform' or 'poisson'.")
    if not np.isfinite(count_scale) or count_scale <= 0.0:
        raise ValueError('count_scale must be positive and finite.')
    if weighting == 'poisson':
        validate_poisson_spectrum(spectrum, model.background)
    if max_nfev < 1:
        raise ValueError('max_nfev must be positive.')
    evaluator = ModelEvaluator(spectrum.energy, model)
    free = evaluator.free
    if spectrum.size <= len(free):
        raise ValueError('Fitting requires more measured samples than independently variable parameters.')
    initial = np.asarray([p.value for p in free])
    lower = np.asarray([p.lower for p in free])
    upper = np.asarray([p.upper for p in free])

    def residual_fn(vector):
        fitted = evaluator.evaluate(vector)[2]
        return poisson_deviance_residual(spectrum.intensity, fitted, count_scale) if weighting == 'poisson' else fitted-spectrum.intensity

    def jacobian_fn(vector):
        jacobian = evaluator.jacobian(vector)
        if weighting == 'poisson':
            fitted = evaluator.evaluate(vector)[2]
            jacobian *= poisson_deviance_derivative(spectrum.intensity, fitted, count_scale)[:, None]
        return jacobian

    if free:
        solution = least_squares(residual_fn, initial, jac=jacobian_fn, bounds=(lower, upper),
                                 max_nfev=max_nfev, method='trf', x_scale='jac')
        best, success, message, evaluations = solution.x, bool(solution.success), str(solution.message), int(solution.nfev)
    else:
        best, success, message, evaluations = initial, True, 'Model has no independently variable parameters.', 0
    values, components, fitted = evaluator.evaluate(best)
    residual = fitted-spectrum.intensity
    weighted = residual_fn(best)
    objective = float(weighted @ weighted)
    dof = spectrum.size-len(free)
    model_jacobian = evaluator.jacobian(best)
    if weighting == 'poisson':
        model_jacobian *= (count_scale/np.sqrt(np.maximum(fitted*count_scale, 1e-12)))[:, None]
    covariance = covariance_from_jacobian(model_jacobian, weighted, absolute_sigma=weighting == 'poisson')
    errors = {}
    for name, (root, multiplier, _) in evaluator.dependencies.items():
        if root is None or multiplier == 0.0:
            errors[name] = 0.0
        else:
            variance = covariance.matrix[evaluator.columns[root], evaluator.columns[root]]
            errors[name] = float(abs(multiplier)*np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else float('nan')
    boundaries = []
    for parameter, value in zip(free, best):
        span = parameter.upper-parameter.lower
        tolerance = max(1e-8, abs(value)*1e-8, span*1e-6 if np.isfinite(span) else 0.0)
        if min(value-parameter.lower, parameter.upper-value) <= tolerance:
            boundaries.append(parameter.name)
    warnings = []
    declared = model.parameters()
    clipped = [parameter.name for parameter in free if parameter.value != declared[parameter.name].value]
    if clipped:
        warnings.append('Initial values were clipped to bounds imposed by parameter ties: '+', '.join(clipped))
    if not success:
        warnings.append('Optimizer did not converge; do not treat this result as a completed fit.')
    if covariance.rank < len(free):
        warnings.append('Rank-deficient model: independently variable parameters are not locally identifiable; standard errors are undefined.')
    elif covariance.condition > 1e8:
        warnings.append('Ill-conditioned model: local parameter uncertainty is sensitive to small perturbations.')
    if covariance.max_correlation > 0.95:
        warnings.append('Strong parameter correlation; review component identifiability.')
    if boundaries:
        warnings.append('Parameters reached bounds; symmetric covariance errors may be unreliable.')
    statistical = None
    if weighting == 'poisson':
        pearson = pearson_residual(spectrum.intensity, fitted, count_scale)
        statistical = float(pearson @ pearson)
        if np.median(fitted*count_scale) < 5.0:
            warnings.append('Low expected counts: Pearson chi-square is asymptotic; assess goodness of fit with count simulations.')
        uncertainty = f'Poisson maximum likelihood (deviance); 1 displayed unit = {count_scale:g} physical count(s). Expected-Fisher local covariance, without residual-variance rescaling.'
    else:
        uncertainty = 'Uniform Gaussian residual likelihood; no calibrated chi-square. Column-scaled SVD local covariance with estimated residual variance.'
    uncertainty += ' Errors are conditional on the fixed background and chosen line shapes.'
    return FitResult(success=success, message=message, parameter_values=values, standard_errors=errors,
                     background=model.background.copy(), components=components, fitted_intensity=fitted,
                     residual=residual, weighted_residual=weighted, chi_square=objective,
                     reduced_chi_square=objective/dof, degrees_of_freedom=dof, optimizer_evaluations=evaluations,
                     weighting=weighting, count_scale=float(count_scale) if weighting == 'poisson' else None,
                     residual_rms=float(np.sqrt(np.mean(residual**2))), statistical_chi_square=statistical,
                     reduced_statistical_chi_square=statistical/dof if statistical is not None else None,
                     uncertainty_model=uncertainty, objective_name='poisson_deviance' if weighting == 'poisson' else 'residual_sum_squares',
                     objective_value=objective, free_parameter_count=len(free), jacobian_rank=covariance.rank,
                     jacobian_condition=covariance.condition, max_parameter_correlation=covariance.max_correlation,
                     boundary_parameters=tuple(boundaries), fit_warnings=tuple(warnings))
