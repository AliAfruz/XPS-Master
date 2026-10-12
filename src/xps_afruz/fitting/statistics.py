"""Shared likelihood, covariance, and evidence calculations.

Compare information criteria only for the same observations and noise model.
Poisson deviance omits a data-only constant that cancels within comparisons.
Uniform fits estimate one additional residual-variance parameter.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class InformationCriteria:
    aic: float
    aicc: float
    bic: float
    parameter_count: int


@dataclass(frozen=True, slots=True)
class CovarianceEstimate:
    matrix: NDArray[np.float64]
    rank: int
    condition: float
    max_correlation: float


def poisson_deviance_residual(observed: ArrayLike, predicted: ArrayLike, count_scale: float = 1.0) -> NDArray[np.float64]:
    """Signed square-root Poisson deviance, including zero-count channels."""
    data = np.asarray(observed, dtype=float)*count_scale
    mean = np.asarray(predicted, dtype=float)*count_scale
    if not np.isfinite(count_scale) or count_scale <= 0.0:
        raise ValueError('Count scale must be positive and finite.')
    if data.shape != mean.shape or not np.all(np.isfinite(data)) or not np.all(np.isfinite(mean)):
        raise ValueError('Poisson data and model must be finite and aligned.')
    if np.any(data < 0.0) or np.any(mean < 0.0):
        raise ValueError('Poisson data and model must be non-negative.')
    mean = np.maximum(mean, 1e-12)
    terms = mean.copy()
    positive = data > 0.0
    delta = (mean[positive]-data[positive])/data[positive]
    near = np.abs(delta) < 1e-4
    term = np.empty_like(delta)
    z = delta[near]
    term[near] = z*z*(0.5-z/3.0+z*z/4.0-z*z*z/5.0)
    term[~near] = delta[~near]-(np.log(mean[positive][~near])-np.log(data[positive][~near]))
    terms[positive] = data[positive]*term
    return np.sign(mean-data)*np.sqrt(2.0*np.maximum(terms, 0.0))


def poisson_deviance_derivative(observed: ArrayLike, predicted: ArrayLike, count_scale: float = 1.0) -> NDArray[np.float64]:
    """Derivative of signed deviance residual with respect to model intensity."""
    data = np.asarray(observed, dtype=float)*count_scale
    mean = np.maximum(np.asarray(predicted, dtype=float)*count_scale, 1e-12)
    residual = poisson_deviance_residual(observed, predicted, count_scale)
    close = (data > 0.0) & (np.abs(mean-data) <= 1e-5*np.maximum(mean, data))
    derivative = np.empty_like(mean)
    derivative[close] = count_scale/np.sqrt(mean[close])
    derivative[~close] = count_scale*(1.0-data[~close]/mean[~close])/residual[~close]
    return derivative


def pearson_residual(observed: ArrayLike, predicted: ArrayLike, count_scale: float = 1.0) -> NDArray[np.float64]:
    """Pearson diagnostics use expected counts, never observed-count weights."""
    mean = np.maximum(np.asarray(predicted, dtype=float)*count_scale, 1e-12)
    return (np.asarray(predicted, dtype=float)-np.asarray(observed, dtype=float))*count_scale/np.sqrt(mean)


def information_criteria(objective: float, sample_count: int, free_parameter_count: int, weighting: str = 'uniform') -> InformationCriteria:
    """AIC/AICc/BIC from fitted likelihood and actual independent parameters."""
    if sample_count < 1 or free_parameter_count < 0 or not np.isfinite(objective) or objective < 0.0:
        raise ValueError('Information criteria require a finite non-negative objective and valid dimensions.')
    if weighting == 'poisson':
        likelihood, count = float(objective), free_parameter_count
    elif weighting == 'uniform':
        likelihood = sample_count*np.log(max(float(objective)/sample_count, np.finfo(float).tiny))
        count = free_parameter_count+1
    else:
        raise ValueError('Unknown likelihood weighting.')
    aic = likelihood+2.0*count
    denominator = sample_count-count-1
    correction = 2.0*count*(count+1)/denominator if denominator > 0 else float('inf')
    return InformationCriteria(float(aic), float(aic+correction), float(likelihood+count*np.log(sample_count)), count)


def fit_information_criteria(result, sample_count: int, parameter_count: int | None = None) -> InformationCriteria:
    """Use retained metadata; preserve readability of legacy fit results."""
    count = result.free_parameter_count if result.free_parameter_count is not None else parameter_count
    if count is None:
        count = max(0, sample_count-result.degrees_of_freedom)
    objective = result.objective_value
    if objective is None:
        if result.weighting == 'poisson':
            data = result.fitted_intensity-result.residual
            weighted = poisson_deviance_residual(data, result.fitted_intensity, result.count_scale or 1.0)
            objective = float(weighted @ weighted)
        else:
            objective = float(result.residual @ result.residual)
    return information_criteria(objective, sample_count, int(count), result.weighting)


def covariance_from_jacobian(jacobian: ArrayLike, residual: ArrayLike, *, absolute_sigma: bool = False) -> CovarianceEstimate:
    """Column-scaled SVD avoids squaring the Jacobian condition number.

    Rank-deficient models have undefined local uncertainties. Poisson errors
    retain absolute counting variance; uniform errors estimate noise variance.
    """
    matrix, weighted = np.asarray(jacobian, dtype=float), np.asarray(residual, dtype=float)
    rows, columns = matrix.shape
    if columns == 0:
        return CovarianceEstimate(np.zeros((0, 0)), 0, 1.0, 0.0)
    invalid = np.full((columns, columns), np.nan)
    norms = np.linalg.norm(matrix, axis=0)
    safe = np.where(norms > 0.0, norms, 1.0)
    try:
        _, singular, vectors = np.linalg.svd(matrix/safe, full_matrices=False)
    except np.linalg.LinAlgError:
        return CovarianceEstimate(invalid, 0, float('inf'), 1.0)
    tolerance = np.finfo(float).eps*max(rows, columns)*singular[0] if singular.size else 0.0
    rank = int(np.sum(singular > tolerance))
    condition = float(singular[0]/singular[-1]) if rank == columns else float('inf')
    if rank < columns or rows <= columns:
        return CovarianceEstimate(invalid, rank, condition, 1.0)
    inverse = (vectors.T/np.square(singular)) @ vectors
    covariance = inverse/np.outer(safe, safe)
    if not absolute_sigma:
        covariance *= float(weighted @ weighted)/(rows-columns)
    std = np.sqrt(np.maximum(np.diag(inverse), 0.0))
    denom = np.outer(std, std)
    correlation = np.divide(inverse, denom, out=np.zeros_like(inverse), where=denom > 0.0)
    np.fill_diagonal(correlation, 0.0)
    return CovarianceEstimate(covariance, rank, condition, float(np.clip(np.max(np.abs(correlation)), 0.0, 1.0)))


def validate_poisson_spectrum(spectrum, background: ArrayLike) -> None:
    """Only count-preserving ROI selection is compatible with raw Poisson data."""
    if np.any(spectrum.intensity < 0.0) or np.any(np.asarray(background) < 0.0):
        raise ValueError('Poisson likelihood requires non-negative raw counts and model background.')
    if any(entry.get('operation') != 'region_of_interest' for entry in spectrum.processing_history):
        raise ValueError('Poisson likelihood requires raw counts; use uniform weighting after intensity processing.')
