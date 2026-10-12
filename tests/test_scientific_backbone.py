"""Known-truth checks for likelihood, uncertainty, bounds, and derivatives."""
from dataclasses import replace
import json
import numpy as np
import pytest

from xps_afruz import BackgroundMethod, FitModel, PeakComponent, Spectrum, fit_spectrum
from xps_afruz.core.processing import normalize_spectrum
from xps_afruz.fitting.evaluator import ModelEvaluator
from xps_afruz.fitting.model import ParameterTie
from xps_afruz.fitting.peak_shapes import evaluate_peak
from xps_afruz.fitting.statistics import (covariance_from_jacobian, fit_information_criteria,
    information_criteria, poisson_deviance_residual, poisson_deviance_derivative)
from xps_afruz.io.project_io import WorkspaceItemData, save_project, load_project


def _area_only(area=15.0, profile='gaussian'):
    return PeakComponent('A', 0.0, 1.4, area, profile=profile, vary_center=False,
                         vary_fwhm=False, area_bounds=(0.0, 500.0))


@pytest.mark.parametrize('count_scale', [1.0, 1000.0])
def test_poisson_likelihood_recovers_exact_area_mle_with_zero_channels(count_scale):
    x = np.linspace(-4.0, 4.0, 160)
    unit = evaluate_peak(x, profile='gaussian', center=0.0, fwhm=1.4, area=1.0)
    counts = np.random.default_rng(51).poisson(12.0*unit)
    assert np.any(counts == 0)
    spectrum = Spectrum(x, counts/count_scale)
    result = fit_spectrum(spectrum, FitModel([_area_only(8.0/count_scale)], np.zeros_like(x)),
                          weighting='poisson', count_scale=count_scale)
    mle = float(counts.sum())/(count_scale*float(unit.sum()))
    assert result.success
    assert result.parameter_values['A.area'] == pytest.approx(mle, rel=1e-6)
    expected_error = np.sqrt(mle/(count_scale*unit.sum()))
    assert result.standard_errors['A.area'] == pytest.approx(expected_error, rel=1e-6)
    assert result.objective_name == 'poisson_deviance'
    assert result.free_parameter_count == result.jacobian_rank == 1
    assert result.standard_errors['A.center'] == 0.0


def test_exact_poisson_expectations_still_have_nonzero_counting_uncertainty():
    x = np.linspace(-3, 3, 100)
    mean = evaluate_peak(x, profile='gaussian', center=0, fwhm=1.4, area=25)
    result = fit_spectrum(Spectrum(x, mean), FitModel([_area_only(25)], np.zeros_like(x)), weighting='poisson')
    assert result.objective_value == pytest.approx(0, abs=1e-9)
    assert result.standard_errors['A.area'] > 0.0


def test_deviance_derivative_handles_zero_and_near_equal_counts():
    data = np.array([0.0, 2.0, 1e8, 4.0])
    mean = np.array([0.8, 2.000001, 1e8+1.0, 9.0])
    step = np.array([1e-5, 1e-5, 0.01, 1e-5])
    numeric = (poisson_deviance_residual(data, mean+step)-poisson_deviance_residual(data, mean-step))/(2*step)
    assert poisson_deviance_derivative(data, mean) == pytest.approx(numeric, rel=3e-5)


def test_likelihood_criteria_use_noise_model_and_variance_parameter():
    uniform = information_criteria(5.0, 10, 2, 'uniform')
    assert uniform.parameter_count == 3
    assert uniform.aic == pytest.approx(10*np.log(0.5)+6)
    assert uniform.aicc-uniform.aic == pytest.approx(4.0)
    poisson = information_criteria(5.0, 10, 2, 'poisson')
    assert poisson.parameter_count == 2
    assert poisson.aic == 9.0
    assert poisson.bic == pytest.approx(5+2*np.log(10))


def test_rank_deficient_fit_does_not_report_false_precision():
    x = np.linspace(-3, 3, 100)
    unit = evaluate_peak(x, profile='gaussian', center=0, fwhm=1.4, area=1)
    model = FitModel([_area_only(10), replace(_area_only(10), name='B')], np.zeros_like(x))
    result = fit_spectrum(Spectrum(x, 30*unit), model)
    assert result.success
    assert result.jacobian_rank == 1
    assert np.isnan(result.standard_errors['A.area'])
    assert np.isnan(result.standard_errors['B.area'])
    assert any('Rank-deficient' in warning for warning in result.fit_warnings)


def test_svd_covariance_is_invariant_to_parameter_units_and_known_variance():
    jacobian = np.column_stack([np.linspace(-1, 1, 100), np.ones(100)])
    residual = np.ones(100)
    original = covariance_from_jacobian(jacobian, residual, absolute_sigma=True)
    units = np.array([1e-9, 1e7])
    rescaled = covariance_from_jacobian(jacobian*units, residual, absolute_sigma=True)
    assert rescaled.rank == 2
    assert np.allclose(rescaled.matrix*np.outer(units, units), original.matrix)
    assert covariance_from_jacobian(jacobian, np.zeros(100), absolute_sigma=True).matrix == pytest.approx(original.matrix)


def test_tied_area_bounds_and_uncertainty_are_propagated():
    x = np.linspace(-3, 5, 200)
    a = _area_only(30)
    b = replace(a, name='B', center=2.0, area=15, area_bounds=(0, 20))
    tie = ParameterTie('B.area', 'A.area', scale=0.5)
    model = FitModel([a, b], np.zeros_like(x), [tie])
    assert model.free_parameters()[0].upper == 40
    signal = evaluate_peak(x, profile='gaussian', center=0, fwhm=1.4, area=100)
    signal += evaluate_peak(x, profile='gaussian', center=2, fwhm=1.4, area=50)
    result = fit_spectrum(Spectrum(x, signal), model)
    assert result.parameter_values['A.area'] <= 40.0
    assert result.parameter_values['B.area'] <= 20.0
    assert result.standard_errors['B.area'] == pytest.approx(0.5*result.standard_errors['A.area'])
    assert 'A.area' in result.boundary_parameters


def test_chained_negative_ties_propagate_bounds():
    a = replace(_area_only(35), area_bounds=(0, 100))
    b = replace(a, name='B', area=65, area_bounds=(60, 80))
    c = replace(a, name='C', area=32.5, area_bounds=(30, 35))
    ties = [ParameterTie('B.area', 'A.area', -1, 100), ParameterTie('C.area', 'B.area', 0.5)]
    free = FitModel([a, b, c], np.zeros(10), ties).free_parameters()
    assert len(free) == 1
    assert (free[0].lower, free[0].upper) == (30, 40)


def test_inconsistent_tied_bounds_are_rejected():
    a = replace(_area_only(1), area_bounds=(0, 3))
    b = replace(a, name='B', area=5.5, area_bounds=(5, 6))
    model = FitModel([a, b], np.zeros(10), [ParameterTie('B.area', 'A.area', offset=10)])
    with pytest.raises(ValueError, match='no feasible interval'):
        model.free_parameters()


@pytest.mark.parametrize('profile', ['gaussian', 'lorentzian', 'pseudo_voigt', 'voigt', 'asymmetric_pseudo_voigt'])
def test_model_derivatives_match_central_difference_with_parameter_ties(profile):
    x = np.linspace(-3, 5, 300)
    a = PeakComponent('A', 0.0, 1.2, 30, profile=profile, eta=0.4, asymmetry=0.15,
                      vary_eta=profile in ('pseudo_voigt', 'voigt'),
                      vary_asymmetry=profile == 'asymmetric_pseudo_voigt', center_bounds=(-2, 2))
    b = replace(a, name='B', center=2.0, area=15, center_bounds=(0, 4))
    ties = [ParameterTie('B.center', 'A.center', offset=2), ParameterTie('B.area', 'A.area', 0.5)]
    evaluator = ModelEvaluator(x, FitModel([a, b], np.zeros_like(x), ties))
    vector = np.array([p.value for p in evaluator.free])
    analytical = evaluator.jacobian(vector)
    for column in range(vector.size):
        step = 1e-5*max(abs(vector[column]), 1)
        high, low = vector.copy(), vector.copy()
        high[column] += step
        low[column] -= step
        numeric = (evaluator.evaluate(high)[2]-evaluator.evaluate(low)[2])/(2*step)
        assert np.allclose(analytical[:, column], numeric, rtol=2e-4, atol=1e-6)


def test_processed_intensities_cannot_use_raw_count_likelihood():
    x = np.linspace(-3, 3, 100)
    data = Spectrum(x, np.ones_like(x)*5)
    with pytest.raises(ValueError, match='raw counts'):
        fit_spectrum(normalize_spectrum(data), FitModel([_area_only()], np.zeros_like(x)), weighting='poisson')
    roi = data.region(-2, 2)
    assert fit_spectrum(roi, FitModel([_area_only()], np.zeros(roi.size)), weighting='poisson').success


def test_fit_diagnostics_round_trip_and_version_nine_migration(tmp_path):
    x = np.linspace(-3, 3, 100)
    spectrum = Spectrum(x, evaluate_peak(x, profile='gaussian', center=0, fwhm=1.4, area=30))
    result = fit_spectrum(spectrum, FitModel([_area_only()], np.zeros_like(x)), weighting='poisson')
    item = WorkspaceItemData(spectrum, BackgroundMethod.LINEAR, {}, fit_result=result)
    path = tmp_path/'diagnostics.xpsa'
    save_project(path, spectrum=spectrum, background_method='linear', peaks=(), workspace_items=(item,))
    loaded = load_project(path).workspace[0].fit_result
    assert loaded.objective_name == 'poisson_deviance'
    assert loaded.free_parameter_count == loaded.jacobian_rank == 1
    assert loaded.objective_value == result.objective_value
    assert loaded.fit_warnings == result.fit_warnings
    payload = json.loads(path.read_text())
    payload['version'] = 9
    fit = payload['workspace']['items'][0]['fit_result']
    for key in ('objective_name', 'objective_value', 'free_parameter_count', 'jacobian_rank', 'jacobian_condition',
                'max_parameter_correlation', 'boundary_parameters', 'fit_warnings'):
        fit.pop(key)
    path.write_text(json.dumps(payload))
    migrated = load_project(path).workspace[0].fit_result
    assert migrated.objective_value is None
    assert np.isfinite(fit_information_criteria(migrated, spectrum.size).bic)
