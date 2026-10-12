"""Compiled peak evaluation and constraint-aware model derivatives."""
from __future__ import annotations
import numpy as np
from .model import FitModel
from .peak_shapes import PeakProfile, evaluate_peak, gaussian, lorentzian

SHAPE_PARAMETERS = ('center', 'fwhm', 'area', 'eta', 'asymmetry', 'instrument_fwhm')


class ModelEvaluator:
    """Resolve ties once and reuse curves between residual and Jacobian calls."""
    def __init__(self, energy, model: FitModel):
        self.energy = np.asarray(energy, dtype=float)
        self.model = model
        self.free = model.free_parameters()
        self.dependencies = model.parameter_dependencies()
        self.columns = {parameter.name: index for index, parameter in enumerate(self.free)}
        self._last_vector = None
        self._last_evaluation = None

    def evaluate(self, vector):
        if self._last_vector is not None and np.array_equal(vector, self._last_vector):
            return self._last_evaluation
        supplied = dict(zip(self.columns, vector))
        values = {name: offset if root is None else scale*supplied[root]+offset
                  for name, (root, scale, offset) in self.dependencies.items()}
        curves = {}
        fitted = self.model.background.copy()
        for component in self.model.components:
            shape = {suffix: values[f'{component.name}.{suffix}'] for suffix in SHAPE_PARAMETERS}
            curve = evaluate_peak(self.energy, profile=component.profile, **shape)
            curves[component.name] = curve
            fitted += curve
        self._last_vector = np.asarray(vector).copy()
        self._last_evaluation = values, curves, fitted
        return self._last_evaluation

    def jacobian(self, vector):
        values, _, _ = self.evaluate(vector)
        jacobian = np.zeros((self.energy.size, len(self.free)))
        parameters = self.model.parameters()
        for component in self.model.components:
            shape = {suffix: values[f'{component.name}.{suffix}'] for suffix in SHAPE_PARAMETERS}
            analytical = self._analytic_derivatives(component.profile, shape)
            for suffix in SHAPE_PARAMETERS:
                name = f'{component.name}.{suffix}'
                root, multiplier, _ = self.dependencies[name]
                if root is None or multiplier == 0.0:
                    continue
                if suffix == 'area':
                    derivative = evaluate_peak(self.energy, profile=component.profile, **dict(shape, area=1.0))
                elif suffix in analytical:
                    derivative = analytical[suffix]
                else:
                    value = shape[suffix]
                    scale = max(shape['fwhm'], 1.0) if suffix == 'center' else max(abs(value), 1.0)
                    step = np.cbrt(np.finfo(float).eps)*scale
                    lower, upper = max(parameters[name].lower, value-step), min(parameters[name].upper, value+step)
                    if upper <= lower:
                        continue
                    high, low = dict(shape), dict(shape)
                    high[suffix], low[suffix] = upper, lower
                    derivative = (evaluate_peak(self.energy, profile=component.profile, **high)
                                  - evaluate_peak(self.energy, profile=component.profile, **low))/(upper-lower)
                jacobian[:, self.columns[root]] += multiplier*derivative
        return jacobian

    def _analytic_derivatives(self, profile, shape):
        if profile not in (PeakProfile.GAUSSIAN, PeakProfile.LORENTZIAN, PeakProfile.PSEUDO_VOIGT):
            return {}
        delta = self.energy-shape['center']
        width, area, eta = shape['fwhm'], shape['area'], shape['eta']
        g = gaussian(self.energy, center=shape['center'], fwhm=width, area=area)
        l = lorentzian(self.energy, center=shape['center'], fwhm=width, area=area)
        gc = g*8.0*np.log(2.0)*delta/width**2
        gw = g*(8.0*np.log(2.0)*(delta/width)**2-1.0)/width
        denominator = delta**2+(width/2.0)**2
        lc = l*2.0*delta/denominator
        lw = l*(1.0-width**2/(2.0*denominator))/width
        zero = np.zeros_like(delta)
        if profile == PeakProfile.GAUSSIAN:
            return {'center': gc, 'fwhm': gw, 'eta': zero, 'asymmetry': zero, 'instrument_fwhm': zero}
        if profile == PeakProfile.LORENTZIAN:
            return {'center': lc, 'fwhm': lw, 'eta': zero, 'asymmetry': zero, 'instrument_fwhm': zero}
        return {'center': (1.0-eta)*gc+eta*lc, 'fwhm': (1.0-eta)*gw+eta*lw,
                'eta': l-g, 'asymmetry': zero, 'instrument_fwhm': zero}
