# Current architecture

`xps_afruz` is deliberately split into independent scientific layers:

The public software and distribution are named **XPS-master**. The Python import
namespace, saved-project format identifiers, RSF library identifiers, and
existing Qt preference keys retain their original values for compatibility.

```text
importer -> Spectrum -> background estimator -> FitModel -> optimizer -> FitResult
```

- `core`: validated one-dimensional spectrum model. Energy is normalized to ascending order internally.
- `backgrounds`: named baseline estimators. The result is supplied explicitly to a fit model.
- fitting.peak_shapes: area-normalized Gaussian, Lorentzian, Gaussian-Lorentzian sum/product, Voigt, empirical asymmetric pseudo-Voigt, and finite-window Doniach-Sunjic-type metallic-state profiles. The profile metadata labels mathematical curves separately from the metallic-state model.
- fitting.model: components, bounds, and linear parameter ties. Components explicitly retain center, FWHM, area, eta, asymmetry alpha, and instrument FWHM. A spin-orbit helper enforces binding-energy separation, area ratio, width, and compatible shape-parameter ties.
- `fitting.optimizer`: SciPy bounded least-squares plus residuals, degrees of freedom, and local standard-error estimates.
- `fitting.evaluator`: compiled affine ties, cached model curves, analytic common-profile derivatives, and bound-aware shape derivatives.
- `fitting.statistics`: shared Gaussian/Poisson likelihoods, information criteria, count diagnostics, and scaled-SVD covariance. See [scientific methods](scientific_methods.md).
- `fitting.global_series` and `fitting.global_series_models`: simultaneous optimization across spectra with shared or scan-specific parameter scopes, bounded relative energy offsets, independent-fit references, explicit free-parameter counting, local covariance uncertainty, and global-vs-independent AICc/BIC evidence.
- `fitting.model_comparison`: immutable candidate hypotheses, controlled alternative generation, cancellable fitting, AICc/BIC support, residual/boundary/overlap/background diagnostics, and finite-difference parameter-correlation checks. `fitting.constraint_templates` supplies generic orbital-degeneracy ratios without assigning chemistry or inventing an experimental splitting.
- `fitting.stability_consensus`: deterministic, cancellable perturbation ensembles over noise, calibration uncertainty, ROI edges, accepted backgrounds, line shapes, randomized starts, and neighbouring component counts. Paired BIC/AICc support and Hungarian component matching feed retained per-component stability metrics and Trust Scores.
- `fitting.stability_models`: validated ensemble settings and retained result contracts, separated from numerical orchestration while re-exported through the original public import path.
- `ui.consensus_worker`: isolates long consensus calculations from the Qt event loop and applies only complete results; cancellation never mutates the retained deconvolution.
- `ui.consensus_controller` and `ui.consensus_report`: separate background-job lifecycle from presentation. `ui.theme` and `ui.widgets` isolate visual styling and reusable safe-input controls from the main workflow controller.
- `ui.interface_builder`, `ui.background_controller`, `ui.energy_controller`, and `ui.fitting_controller`: keep Qt construction, baseline workflows, energy provenance, peak editing, and fit-worker orchestration out of the desktop composition root.
- `ui.model_comparison_worker`, `ui.model_comparison_controller`, and `ui.model_comparison_report`: separate the long candidate batch, retained per-card lifecycle, overlay dashboard, and explicit acceptance action. A completed comparison does not mutate the active fit.
- `ui.global_series_worker`, `ui.global_series_controller`, and `ui.global_series_report`: isolate the simultaneous optimizer, series selection and retained project state, and the global/independent fit, residual, shift, and component-trend dashboard. Applying global curves to cards is explicit.
- `ui.workspace_controller`, `ui.survey_controller`, `ui.processing_controller`, and `ui.project_controller`: keep multi-card state, elemental/RSF workflows, non-destructive processing, and persistence/export independent from the desktop composition root.
- `io`: conservative delimited-text and Thermo/VG VGD importers. VAMAS and additional vendor importers remain future extensions.
- `manuscript_plotter`: a separately launched Tk/Matplotlib publication editor adapted from XPS Plot Studio 1.5. Its reader accepts Avantage-style workbooks and the stable XPS-master Excel-export contract, while its renderer keeps the main spectrum and residual axes synchronized. Launching it in a detached process prevents the Tk and Qt event loops from competing.

## Reproducibility rules

The current engine does not silently select chemical states, a background, or a reference energy. Those inputs are explicit. Project files preserve the spectrum provenance, calibration, processing history, background selection/parameters, peak profile and shape values, every active tie, every retained model-comparison curve/metric, aggregate Stability-Consensus settings/evidence, and complete global/independent series results. Consensus replica curves are not serialized.

## Scientific limits

The engine is not an autonomous chemical-assignment tool. Tougaard handling remains exploratory until validated against the laboratory's chosen method. Asymmetric and Doniach-Sunjic-type profiles are selectable models, not evidence that a component has a metallic state; their use must be justified by the specimen, acquisition conditions, and laboratory method.

Stability Trust Scores are numerical robustness summaries, not probabilities of elemental identity, oxidation state, or physical peak existence. Model-comparison and consensus BIC/AICc weights compare their tested numerical families only; they cannot rescue omitted physics or an inappropriate ROI/background family.

Global parameter sharing is also a tested numerical hypothesis. A favourable AICc/BIC comparison does not prove common chemical identity or acquisition comparability. The analyst must justify series membership, backgrounds, calibration bounds, peak constraints, and line shapes.

## Phase 10A reliability boundary

The desktop entry point configures bounded rotating diagnostics before constructing Qt widgets. Unhandled GUI-thread exceptions are recorded with a visible log-file location; spectra and fitted arrays are deliberately not written to logs. Headless GUI smoke tests exercise the responsive dock layout and multi-spectrum board in addition to the numerical test suite. GitHub Actions runs the complete suite on Windows with Python 3.10 and 3.12.
