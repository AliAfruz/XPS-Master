# Python code organization

XPS-master uses small domain modules around one desktop composition root. New work should extend the narrowest relevant module instead of adding another responsibility to `ui/main_window.py`.

## Current boundaries

| Area | Module | Responsibility |
|---|---|---|
| Desktop composition | `ui/main_window.py` | Connect high-level fit, survey, workspace, processing, calibration, and project workflows. |
| Interface construction | `ui/interface_builder.py` | Build menus, workflow tabs, plot panels, tables, and analysis docks. |
| Visual design | `ui/theme.py` | Metallic-dark Qt stylesheet only. |
| Reusable controls | `ui/widgets.py` | Input widgets that prevent accidental wheel changes. |
| Fitting workflow | `ui/fitting_controller.py` | Peak editing, constraints, ordinary fitting, and smart-deconvolution lifecycle. |
| Manual-fit identifiability | `fitting/seed_protection.py` | Ordered local center zones, local FWHM bounds, tie-aware seed protection, and post-fit pile-up diagnostics. |
| Model-comparison calculation | `fitting/model_comparison.py` | Candidate contracts, controlled alternatives, fits, diagnostics, and AICc/BIC support. |
| Constraint templates | `fitting/constraint_templates.py` | Generic p/d/f degeneracy ratios with analyst-supplied splitting. |
| Model-comparison workflow | `ui/model_comparison_controller.py` | Cancellable worker lifecycle, per-card retention, and explicit candidate acceptance. |
| Model-comparison presentation | `ui/model_comparison_report.py` | Ranked copyable evidence table with envelope/component/residual overlays. |
| Background workflow | `ui/background_controller.py` | Model settings, recomputation, diagnostics, and background-quality presentation. |
| Energy workflow | `ui/energy_controller.py` | Scale conversion, charge correction, calibration provenance, and project-wide shifts. |
| Consensus lifecycle | `ui/consensus_controller.py` | Start, cancel, track, and safely retain a background consensus job. |
| Consensus presentation | `ui/consensus_report.py` | Copyable Trust Report table and Trust Map dialog. |
| Worker boundary | `ui/consensus_worker.py` | Qt signals and cooperative cancellation around the scientific engine. |
| Workspace workflow | `ui/workspace_controller.py` | Per-card spectrum, background, fit, survey, constraint, and Trust state plus navigator/card rendering. |
| Survey workflow | `ui/survey_controller.py` | Element scope, trace recovery, screening tables, editable RSFs, quantification, and fit seeding. |
| Processing workflow | `ui/processing_controller.py` | ROI, normalization, filtering, derivatives, despiking, spectrum arithmetic, SNR, and undo. |
| Project workflow | `ui/project_controller.py` | Save/open/autosave/recovery, analysis-setting restoration, and complete Excel export. |
| Consensus contracts | `fitting/stability_models.py` | Validated settings, retained results, and internal observation records. |
| Consensus calculation | `fitting/stability_consensus.py` | Perturbations, fitting ensemble, model support, matching, and metrics. |

## Maintenance rules

- Keep scientific calculations independent of Qt.
- Keep dialogs independent of project/workspace mutation.
- Long calculations run behind a cancellable worker boundary.
- Result/configuration dataclasses live separately from orchestration when a scientific module becomes large.
- Preserve old public import paths when moving types, so saved projects and user scripts do not break.
- Add focused tests before extracting the next controller.

## Completed controller extraction

Workspace, survey/quantification, project/export, processing, background, energy, fitting, and interface-construction workflows now live behind independent mixins. Three organization passes reduced `ui/main_window.py` from roughly 3,900 physical lines to about 490 without changing the project format or public scientific imports.

The composition root now owns only application state, recent-project/window state, shared clipboard and plot helpers, spectrum loading, final plot rendering, and shutdown/error handling. Further extraction should be driven by a concrete feature or maintenance need rather than file length alone, and every boundary change must pass the full regression suite.
