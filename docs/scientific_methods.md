# Phase 14: fitting and statistical methods

Version 0.2.0 changes the numerical engine and its evidence calculations. It
keeps backgrounds and chemical hypotheses explicit. A more stable optimizer
does not establish the validity of an analyst's background or peak assignment.

## Objective and noise model

Uniform mode minimizes residual sum of squares (RSS). Model evidence uses the
Gaussian likelihood with an estimated common residual variance. Its parameter
count is the number of independent model parameters plus one variance parameter.

Poisson mode fits non-negative, unprocessed counts with a declared count scale
`s`: one displayed intensity unit represents `s` physical counts. With observed
counts `d` and model counts `m`, it minimizes

`D = 2 sum(m - d + d log(d / m))`.

The `d = 0` contribution is `2m`. A signed square-root deviance residual permits
bounded least-squares optimization of this likelihood. A series expansion near
`m = d` avoids numerical cancellation. A positive floor of `1e-12` physical
counts handles zero model predictions. The fixed background is included in
the predicted trace; it is not subtracted from count observations.

Pearson chi-square is calculated separately from **expected model counts**.
Its reference distribution is asymptotic; low-count goodness-of-fit decisions
require count simulations. Normalization, filtering, derivatives, despiking,
and spectrum arithmetic invalidate the raw Poisson assumption. An unmodified
ROI crop and energy calibration preserve counts.

See [Sherpa's C-statistic definition](https://sherpa.readthedocs.io/en/latest/statistics/api/sherpa.stats.CStat.html).

## Derivatives, scaling, and constraints

The evaluator compiles linear parameter ties once, caches the last curves, and
applies analytic area derivatives for every profile. Gaussian, Lorentzian, and
pseudo-Voigt center/width/mix derivatives are analytic. Other shape derivatives
use central or one-sided differences within declared bounds. The constraint
chain rule contributes each dependent component to its independent root.

Trust-region reflective optimization uses Jacobian column scaling. Dependent
parameter bounds are propagated back through chained ties, including negative
scales and offsets. Incompatible constraints fail before optimization. Seeds
clipped to a tighter tied interval receive a retained review warning.

See [SciPy's bounded optimizer and scaling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

## Uncertainty and identifiability

Covariance uses a column-scaled singular-value decomposition of the Jacobian,
avoiding the loss of precision caused by forming `J.T @ J` first. Uniform
covariance estimates noise with `RSS / (n - p)`. Poisson covariance uses expected
Fisher information with absolute counting variance; it is not rescaled by the
observed residual. Fixed parameters have zero local standard error, and tied
errors propagate through their affine multiplier.

Rank-deficient systems return undefined variable uncertainties rather than
precise-looking pseudoinverse errors. Large condition numbers, high parameter
correlation, and active bounds receive warnings. In global fits, a scan's center
variance includes the cross-covariance between its shared reference center and
its relative shift. Global rank refers to the combined system; per-scan degrees
of freedom remain allocated bookkeeping quantities.

These are local, symmetric approximations conditional on fixed backgrounds,
selected profiles and constraints. They exclude background-choice uncertainty,
calibration systematics, and nonlinear or one-sided confidence intervals. The
retained Stability-Consensus engine separately explores several of these
sensitivities. See [SciPy's covariance assumptions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html).

## Consistent model evidence

All fitting workflows use the same likelihood helper. Poisson information
criteria use deviance plus complexity penalties; uniform criteria use
`n log(RSS/n)`. AICc includes its finite-sample correction and BIC uses `p log(n)`.
Only independently variable parameters count, plus the uniform variance term.
The omitted data-only likelihood constant cancels only when comparing models
on the same observations with the same noise assumption.

Nonconverged smart-search fits are excluded. Candidate models that fail
convergence, rank, finite-evidence, or background checks have zero support and
cannot be accepted. If every candidate is ineligible, the dashboard explicitly
reports that none passed; its first row remains diagnostic only.

## Reproducibility and validation

Project format v10 retains objective type/value, independent parameter count,
Jacobian rank and scaled condition, maximum correlation, active bounds, and
review warnings. Versions 1–9 still load; legacy fit values are preserved rather
than rewritten as newly optimized results. Excel includes a separate diagnostic
section without changing the existing curve-column contract.

Regression tests cover exact Poisson area MLEs with zero channels and different
count scales, nonzero Poisson uncertainty at exact expected counts, derivative
agreement, unit-invariant covariance, unidentifiable peaks, chained tie bounds,
global center/shift covariance, global Poisson evidence, candidate eligibility,
and project/Excel round trips. See [benchmark results](performance_validation.md).
