# Changelog

## 0.2.0 — 12 October 2026

### Fitting and scientific methods

- Compiled affine parameter ties and cached model evaluation.
- Analytic derivatives for common peak profiles and scaled optimization.
- Raw-count Poisson maximum likelihood, including zero-count channels.
- Likelihood-consistent AICc/BIC across the fitting workflows.
- Propagated bounds and uncertainties for tied parameters.
- Scaled SVD covariance, Poisson expected-Fisher uncertainties, and correlated
  shared-center/energy-shift uncertainties in global fits.
- Retained rank, correlation, boundary, and convergence diagnostics.
- Project format v10 with support for reading older projects.

### Repository and distribution

- Software renamed to **XPS-master**, with updated interface, exports, commands,
  repository links, screenshots, and fitting figures.
- Author signature and contact details for Ali Afruz.
- Illustrated README, real application screenshots, and reproducible synthetic
  fitting showcase.
- Citation metadata, contribution guide, issue templates, and upload guide.
- Desktop command `xps-afruz` and clean source-distribution packaging.
- Resizable workspace height for inspecting complete spectrum cards.
- Citation metadata and instructions for a later Zenodo DOI release.

### Validation

- 102 scientific and offscreen GUI tests passed for the prepared release.
- Synthetic common-profile fitting benchmarks measured 2.56–3.36x speed ratios.
- Included real VGD import and narrow-region fitting smoke checks passed.

See [scientific methods](docs/scientific_methods.md) and
[benchmark conditions](docs/performance_validation.md).
