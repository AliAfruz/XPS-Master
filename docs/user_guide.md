# XPS-master — detailed user guide

Scientific, reproducible X-ray Photoelectron Spectroscopy (XPS) analysis software built in Python.

## Status

Phase 14 (0.2.0) strengthens the fitting and statistical backbone: compiled peak derivatives, scaled optimization, raw-count Poisson maximum likelihood, consistent model evidence, enforced tie bounds, and identifiable local uncertainties. It retains Global Multi-Spectrum Fitting, Physics-Constrained Model Comparison, the Stability-Consensus Engine, calibration, Excel export, and complete project sessions. See [scientific methods and validation](scientific_methods.md).

## Scope and scientific note

This package helps analyze XPS spectra but does not establish chemical assignments automatically. Fit models, calibration references, constraints, and results must be reviewed by a qualified analyst.

## Quick start

```powershell
py -3.12 -m venv .venv
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python -m pip install -e '.[dev]'
& $python -m pytest
& $python run.py
```

On Windows, you can also double-click `run.bat` to launch the same Python runner.
The launcher uses this checkout's source and validates available environments.
If a moved `.venv` references a removed interpreter, it can use a compatible
Codex bundled Python with the retained dependencies when that runtime exists.

Activation is optional; the commands above work even when PowerShell execution
policy prevents running `Activate.ps1`. This project intentionally stays as a
Python application during development. Windows `.exe` packaging comes only
after the desktop app and its scientific validation are complete.

## Included in Phases 1–14

- Validated XPS spectrum model that safely normalizes descending energy axes.
- CSV/TSV/two-column text import with header and comment handling.
- Thermo/VG Scientific `.VGD` binary-file import, including kinetic-to-binding-energy conversion and multi-spectrum selection.
- Text-import metadata preservation for header fields such as source/photon energy, pass energy, step size, dwell time, sweeps, analyzer settings, take-off angle, and acquisition date when present in an export.
- Explicit binding/kinetic-energy representation, stored photon energy, original imported axis, calibration/charge-correction history, and quadrature energy-uncertainty tracking.
- Linear, Shirley, and explicit-parameter Tougaard-style backgrounds.
- Area-normalized Gaussian, Lorentzian, Gaussian-Lorentzian sum and product, Voigt, empirical asymmetric pseudo-Voigt, and finite-window Doniach-Sunjic-type metallic-state components.
- Scaled bounded fitting with explicit noise likelihoods: uniform mode minimizes RSS; Poisson mode minimizes signed count deviance, including zero-count channels, with an explicit displayed-unit count scale. Pearson χ² uses fitted expected counts and is labelled as an asymptotic diagnostic. Intensity-processed spectra cannot silently use a raw-count likelihood.
- Phase 14 numerical and statistical reliability: cached model evaluation and analytic Gaussian/Lorentzian/pseudo-Voigt derivatives with constraint chain rules; bound-aware numerical derivatives for other profiles; propagated bounds through chained linear ties; column-scaled SVD covariance; absolute expected-Fisher covariance for Poisson counts; standard errors for tied parameters; and combined center/shift covariance in global fits. Rank, correlation, and boundary diagnostics are visible and retained in project format v10 and Excel. Older v1–v9 projects remain readable.
- Phase 14 evidence corrections: smart search, consensus, candidate comparison, and global fitting share one likelihood-aware AICc/BIC implementation. Fixed and tied parameters do not incur independent-parameter penalties; uniform likelihoods include the estimated noise variance. Nonconverged smart candidates are excluded, and failed eligibility checks never receive normalized candidate support or permit acceptance.
- Explicit linear parameter ties, including a spin-orbit doublet helper for fixed splitting, area ratio, width, and shape.
- Fit residuals, chi-square, degrees of freedom, and local covariance-based standard errors.
- Synthetic examples and unit tests.
- Desktop interface with spectrum import, interactive fit and residual plots, peak editing, constrained doublets, and self-contained `.xpsa` project files.
- Modern warm-metal 2D interface with flat controls, larger tables and form fields, scroll-safe workflow panels, tabbed Survey/Composition consoles, optional dock visibility, and a workspace board that opens only when multi-spectrum context is useful.
- Phase 10A reliability boundary: 2 MB rotating diagnostic logs with four backups, a user-visible unexpected-error report path, Windows CI on Python 3.10/3.12, dependency validation, and headless GUI tests for responsive docks and multi-spectrum workspace behavior. Logs contain application lifecycle/failure diagnostics, not spectrum arrays or scientific results.
- Right-click clipboard tools on all data tables: copy a selected cell range, copy the complete table as TSV, or copy the visible table as an image. Both scientific plots also provide **Copy plot as image**.
- Wheel-safe dropdown and numeric controls: scrolling over a control does not silently change its selected background, peak profile, or match tolerance.
- Survey material finder: reproducible peak detection plus a curated, inspectable library of common XPS core lines. A checked-element scope restricts the search; when narrowed to 12 or fewer elements, an optional local-SNR trace pass can recover weak expected lines missed by global detection. Targeted evidence is explicitly labelled and never promoted to an automatic confirmation.
- Survey consistency checking: maximum and centroid positions, local screening areas, SNR/detection proxy, photoelectron/Auger distinction, overlap warnings, expected companion lines, unassigned features, and removable candidates.
- RSF-based atomic-percent preview with editable factors, transmission/analyzer correction fields, uncertainty proxy, JSON instrument/laboratory RSF libraries, and reproducible factor snapshots in `.xpsa` projects.
- Non-destructive high-resolution processing: numerical ROI, normalization, Gaussian and Savitzky–Golay smoothing, first/second derivatives, robust spike detection/despiking, spectrum addition/subtraction, replicate averaging, noise/SNR, and Undo.
- Background configurations with visual inspection: linear (including optional constrained endpoint levels), Shirley, endpoint-anchored Tougaard U2/U3 with explicit B/C/D plus automatic or constrained scale, and energy-aligned imported custom traces. Peak-safe lower-quantile edge anchors avoid strong edge features; diagnostics flag material over-background without changing the analyst's selected model. The selected model and parameters are retained in `.xpsa` projects and fit summaries.
- Phase 6 peak controls for center, FWHM, area, Gaussian-Lorentzian mix eta, asymmetry alpha, and optional instrument FWHM. Mathematical and empirical profiles are visibly labelled separately from the metallic-state Doniach-Sunjic model, and all selected values are preserved in `.xpsa` projects.
- Phase 7 multi-survey workspace: load several text or VGD spectra into a visual card board, screen all cards for possible elements, and select any card to make it the active spectrum for the established fitting and processing tools. Processing/calibration steps remain attached to their selected workspace card during the session.
- Phase 8 retained-fit workspace: each board card independently retains its peak components, parameter ties, background choice, and latest fit result. Compact board plots show saved colourful component envelopes, while Fit, Survey, and Workspace tabs keep the primary workflow uncluttered.
- Phase 9 smart deconvolution: choose an exact component count, a starting line-shape family, and the number of candidate models. A responsive worker evaluates diverse bounded starting fits, reports progress, and selects the numerical model with the best BIC-plus-residual-structure-plus-collapse score. It proposes a fit only; the analyst must still justify component count, background, line shape, and chemical assignment.
- Smart seed protection for ordinary manual fits is enabled by default. Each manually placed component receives an ordered local center zone and a width ceiling scaled to its seed and neighbouring peaks, preventing exchangeable components from crossing or collecting at one residual minimum. The fit summary reports unresolved overlap, negligible area, center/width boundary hits, and structured residuals; the protection can be disabled explicitly for unconstrained exploration.
- Phase 10B complete project sessions: one versioned `.xpsa` file retains every workspace spectrum, processing/calibration provenance, background configuration, editable component/tie setup, latest fit arrays and statistics, survey evidence, active card, workflow tab, weighting choice, and relevant dock visibility. Atomic Save/Save As, recent projects, 60-second dirty-session recovery snapshots, and save/discard/cancel protection are covered by round-trip and GUI reopen tests. Versions 1–5 migrate into a single-card workspace.
- Phase 10C complete Excel export: **File → Export complete workspace to Excel…** creates one `.xlsx` workbook with a Summary and one worksheet per workspace plot. Each scientific sheet includes native measured/fit and residual charts; active and original energy axes; measured intensity, background, formula-driven net signal, fit envelope and residual; weighted residual; every net component and component-plus-background curve; fitted values and standard errors; constraints; background parameters; processing/calibration history; survey evidence; and acquisition metadata. Sheet names are sanitized and unique, numerical columns remain machine-readable, formula cells are auditable, and the target is replaced atomically only after a valid workbook closes successfully.
- Integrated XPS-master Manuscript Plotter 2.0: the maintained XPS Plot Studio 1.5 source is preserved as a separate module with its original Avantage-style workbook support plus direct XPS-master complete-workbook detection. It ignores the Summary and comparison-audit columns, loads the accepted measured/background/envelope/deconvoluted component curves, and renders a synchronized manuscript-style residual panel. The dedicated **Publish** runner tab and **File → Manuscript Plotter** menu can export the current workspace and open it immediately, open an existing Avantage/XPS-master workbook, or launch an empty plot editor. PNG, TIFF, PDF, SVG, per-sheet styles, annotations, reusable templates, and batch export remain available.
- Phase 10D project-wide calibration: **Energy → Apply signed shift to entire project…** adds one documented eV offset to every loaded active axis in a single atomic operation. Imported axes remain unchanged, uncertainty is propagated, retained component centers and linear constraints translate with the spectra, fit/residual intensity arrays stay valid, and previously screened surveys are recalculated at their retained tolerance and element scope.
- Phase 11 Stability-Consensus: a cancellable worker evaluates paired perturbation scenarios across empirical/Poisson noise replicas, recorded calibration uncertainty, ROI-edge variation, current/linear/Shirley/Tougaard U2/Tougaard U3 backgrounds, conservative line-shape alternatives, randomized bounded starts, and optional K−1/K/K+1 component counts. Hungarian assignment matches comparable components between models. Each retained Trust Report records component survival, separate normalized BIC and AICc support plus their labelled consensus weight, center/FWHM/area dispersion, background and calibration sensitivity, boundary/collapse frequency, ensemble parameter correlation, an explicit Stability Trust Score, exact settings, random seed, and a non-chemical identifiability label. Reports are retained per workspace card, were introduced in project format v7, remain preserved in v8, are restored on reopen, shown in a copyable table and Trust Map, and exported to Excel.
- Phase 12 Physics-Constrained Model Comparison: the analyst's exact current model is compared with controlled Linear/Shirley/Tougaard, Gaussian/pseudo-Voigt, and optional K−1/K+1 alternatives in a cancellable worker. Each complete candidate retains its fitted curves, components, constraints, background parameters, AICc/BIC and normalized weights, residual autocorrelation and Durbin–Watson statistic, boundary-hit and overlap risk, numerical parameter correlation, background diagnostics, eligibility guardrail, warnings, and identifiability label. The dashboard overlays every envelope and residual but never replaces the active fit until **Accept selected candidate** is clicked. Generic p/d/f degeneracy templates create linked doublets while leaving splitting and ratio review under analyst control. Per-card alternatives are saved in project format v8, translated during project-wide calibration, and fully exported to Excel.
- Phase 13 Global Multi-Spectrum Fitting: choose related workspace spectra from the same high-resolution region and fit them simultaneously. Component centers, FWHM and fitted line-shape terms can be shared, while every spectrum retains its own peak areas, background and bounded rigid energy offset. The engine also performs fully independent reference fits and compares the hypotheses using AICc/BIC with their actual free-parameter counts. Global and independent curves, residuals, covariance-based uncertainty, component trends, settings and scientific-boundary note are retained in project format v9, translated by project-wide calibration, displayed in a review dashboard, and exported to a dedicated Excel sheet with typed trend and curve data. Applying global fits to workspace cards remains an explicit analyst action.

## Design principles

Every background and constraint is visible to the analyst. The package does not make an unreviewed chemical-state assignment or claim that a numerical fit establishes a material interpretation. See [the architecture notes](architecture.md) for the scientific boundaries and extension plan.

## Survey material finder and composition preview

Open a wide-range survey (including `.VGD`) and select **Find possible elements**. The dock lists the measured maximum/centroid, approximate reference energy, SNR, energy error, feature type, overlap candidates, and companion lines expected for each element. A candidate can be removed with **Remove selected element from survey**; this excludes every line for that element from both the current survey screen and the composition preview. **Restore removed elements** reverses that choice.

Use **Element search scope** to check the elements you want to investigate. With **Trace recovery** enabled and no more than 12 elements checked, the finder performs a second, hypothesis-driven search at their expected reference energies. A weak line must exceed the configured local SNR threshold; recovered rows are marked `trace`, and their atomic-percent values use `≈` plus an uncertainty that includes an inverse-SNR term. This mode helps retain low-concentration signals but does not prove elemental identity. If the generic library has no RSF for a recovered line, select its result row and use **Set RSF for selected line…** to enter a validated instrument/laboratory value; it is saved as an explicit user-defined factor rather than silently invented.

Boron screening retains separate low- and high-binding-energy B 1s reference positions. These alternatives prevent weak B near 191–192 eV from being discarded solely because it is far from the low-BE reference; their labels are screening contexts, not chemical-state assignments. Narrow the element scope to include B so the local-SNR trace pass is enabled.

The composition dock uses `corrected intensity = area / (RSF × transmission × analyzer)` followed by normalization to atomic percent. Its generic Al Kα RSFs are editable starting values, not universal instrument calibration. Use a validated laboratory/instrument RSF JSON library through the Python API, or edit the displayed RSF; the project stores the selected values and correction history. Confirm candidates with the [NIST XPS Database (SRD 20)](https://www.nist.gov/srd/surface-data), appropriate companion lines, instrument-specific RSFs, and narrow-region fitting before reporting a material assignment.

## Energy calibration

The **Energy** menu converts binding/kinetic energy only when a photon energy is recorded, offers charge correction against an analyst-selected reference, and provides **Apply signed shift to entire project…** for a shared calibration offset. A positive shift moves a 284.800 eV active coordinate to 286.000 eV when `+1.200 eV` is entered; the dialog states this direction explicitly. The signed offset is applied to every loaded spectrum, including retained high-resolution fit centers. The imported axes are preserved, uncertainty and project scope are recorded in each spectrum's history, and previously generated survey matches are recalculated. A grey dashed curve shows the uncorrected coordinate after calibration. **Show calibration provenance** displays source energy, active/imported scales, uncertainty, operation history, and the complete-workspace scope.

## High-resolution processing and backgrounds

The **Process** menu never overwrites the active data: each selected operation creates a labelled derived spectrum with its parameters recorded, and **Undo last processing step** restores the prior spectrum. Use smoothing and derivatives cautiously—they are exploratory transformations, not substitutes for acquisition quality or fitting validation.

Use **Configure background…** to make its parameters explicit. Tougaard follows the internally ascending energy axis, starts at its low-energy anchor, and automatically scales its loss tail to the high-energy anchor unless you enter a numeric constraint. **Inspect background quality** reports both anchors, net-signal floor, robust noise, and the fraction of background above the measured trace. A custom background trace must have exactly the same energy axis and number of points as the active spectrum; this makes the imported constraint inspectable and reproducible.

## Peak fitting and line shapes

The peak table exposes position, FWHM, integrated area, eta, asymmetry alpha, and instrument FWHM. The **Profile** selector labels Gaussian, Lorentzian, Gaussian-Lorentzian sum/product, Voigt, and asymmetric pseudo-Voigt as mathematical or empirical curves. Use these only when their constraints are scientifically defensible; a smaller residual alone is not a physical justification.

**Doniach-Sunjic** is separately labelled as a finite-window metallic-state model. Its alpha parameter controls a low-binding-energy tail on the internally ascending binding-energy axis, while instrument FWHM applies an explicit Gaussian broadening convolution. Use **Fit → Peak line-shape guide** for the documented interpretation of each option. Review every model choice, parameter bound, background, and residual before reporting a chemical-state result.

## Physics-Constrained Model Comparison

Prepare a reviewed component model and explicit constraints, then choose **Compare physical models** in the Fit tab or **Fit → Run physics-constrained model comparison**. The current candidate is preserved exactly. Optional switches add background-family, line-shape, and neighbouring-count hypotheses one controlled choice at a time. The left workflow is scrollable, so the comparison and Stability-Consensus controls retain their intended size instead of compressing.

The background worker reports progress and retains only a complete candidate batch. The dashboard ranks eligible candidates using the arithmetic mean of separately normalized AICc and BIC weights, overlays every candidate envelope and residual, and displays background, correlation, boundary, overlap, and residual-structure warnings. A background that exceeds the signal beyond the documented limit is excluded from support normalization instead of silently becoming the preferred result. **Accept selected candidate** is the only action that replaces the editable model; every alternative remains attached to the spectrum card for later review.

AICc/BIC support is conditional on the tested candidate family. It is not the probability of an oxidation state, elemental identity, or physical peak existence. Omitted physics, an inappropriate ROI, an invalid background, or unjustified constraints can make every compared candidate wrong.

## Global Multi-Spectrum Fitting

Prepare the reviewed peak hypothesis on the active reference spectrum, then choose **Select series and fit globally…** in the Fit tab or **Fit → Run global multi-spectrum fit**. The selection dialog enables only spectra that use the same energy representation and cover every active component center. Keep the active reference selected and include only scans acquired for the same chemical region and a scientifically comparable series.

Peak areas always remain spectrum-specific. Centers are represented in one shared reference frame, and each non-reference spectrum may receive a bounded rigid energy offset. FWHM and fitted eta, asymmetry, or instrument broadening can be shared independently. Existing doublet and other linear parameter ties are resolved in every spectrum. Each selected card keeps its own retained background model and parameters.

The worker first calculates independent fits, then the simultaneous fit. The dashboard shows the global and independent envelopes and residuals for every scan, fitted relative shifts with uncertainty, component-area trends, parameter counts, and ΔAICc/ΔBIC defined as `independent − global`; positive values support the more parsimonious global hypothesis. **Apply global fits to workspace** is the only action that replaces per-card fits. The independent reference fits remain in the retained dashboard and Excel export.

Parameter sharing is an explicit numerical hypothesis, not proof that components have the same chemical origin. Review acquisition comparability, energy calibration, ROI, backgrounds, line shapes, constraints, and residual structure before interpreting a preferred global model.

## Stability-Consensus Trust Reports

Choose the component count and starting profile in **Smart deconvolution**, then use **Run stability consensus**. The recommended neighbouring-count option fits K−1, K, and K+1 against each identical perturbed replica, making their BIC/AICc comparison paired rather than comparing different noise realizations. Only scenarios that successfully complete every tested K contribute to model-count weights, and the report records that paired-scenario count. Separate normalized BIC and AICc weights are retained; the displayed consensus support is their arithmetic mean so criterion disagreement remains visible. The retained deconvolution is never replaced. Progress is reported for every scenario/model combination and **Stop consensus analysis** cooperatively exits without applying a partial result.

For uniform-weight spectra, high-frequency point differences are robustly clipped and resampled; raw-count workflows use Poisson replicas with the displayed-unit count scale. Calibration perturbations use the spectrum's retained energy uncertainty. Alternative backgrounds that materially exceed the perturbed signal are rejected using the established background diagnostic rather than silently admitted. Component clusters are matched to the best requested-K reference model with linear-sum assignment using center and width distance.

**Survival** means recurrence among valid requested-K replicas. **Model support** combines recurrence with the tested-count consensus of normalized BIC and AICc weights. The Trust Score is a transparent engineering summary of survival, model support, center/area stability, background/calibration sensitivity, boundary hits, collapse risk, and ensemble parameter correlation. It is not a calibrated probability that a peak is physically real, and BIC/AICc support is not a Bayesian probability of a chemical state. The implementation is informed by published XPS uniqueness and Bayesian model-selection work: [uniqueness plots](https://doi.org/10.1016/j.apsusc.2016.06.070), [BIC decomposition and confidence intervals](https://doi.org/10.1080/14686996.2020.1773210), and [Bayesian Si 2p analysis](https://doi.org/10.1016/j.apsusc.2024.162001).

Changing the spectrum, active background, background parameters, or applying a project-wide calibration invalidates the affected stale Trust Report. Completed aggregate reports and their exact ensemble settings are serialized; transient replica spectra and curves are deliberately not stored, keeping project files auditable and reasonably sized.

## Multi-survey workspace

Use **File → Load surveys into workspace…** to load several text exports and/or every spectrum from a VGD file. The **MULTI-SURVEY WORKSPACE** board shows a card for each loaded spectrum. Choose a card, its **Work on this spectrum** button, or a navigator row to make that spectrum active for fitting, processing, calibration, and survey analysis.

Use **Workspace → Screen all workspace surveys** to add the possible-element summary and peak markers to every card. Removing a card does not modify its source file. **File → Save project** stores the complete board and every retained per-card result; reopening the `.xpsa` reconstructs the saved workspace instead of appending it to an unrelated session. Use **Save project as…** for a new path. Unsaved work is marked with `*` in the title and receives a recovery snapshot every 60 seconds.

Use **File → Export complete workspace to Excel…** (`Ctrl+Shift+E`) to create an auditable workbook for all loaded plots. The export does not alter or mark the project dirty. Fitted sheets include deconvoluted peak curves and a residual chart; unfitted sheets still include the measured trace, calculated retained background, net-signal formula, metadata, and an explicit note that no residual is available.

## Manuscript Plotter

Open the main workflow's **Publish** tab for the integrated runner. **Export workspace and open plotter…** becomes available when at least one project spectrum is loaded and reports the number of retained fits that will be included. **Open Avantage or XPS-master Excel…** handles an existing workbook, while **Launch empty plotter** starts the editor without data. The same actions remain available through **File → Manuscript Plotter**. Only scientific data sheets are listed: `Export Summary` is skipped, and `COMPARE · …` columns remain audit data rather than being mistaken for the accepted deconvolution.

The fitted envelope, background, accepted colourful components, and measured spectrum share the established XPS Plot Studio manuscript template. Residuals use a linked lower panel with a zero reference line and the same font, axis, normalization, DPI, template, preview, and export settings. The panel can be disabled under **Plot & Axes → Separate residual panel**. The original plotter directory is not modified.

Raster figures export at **600 DPI minimum** (up to 2400 DPI); the renderer raises older or manually entered lower values to 600. PDF and SVG remain available for vector-first manuscript workflows.

The standalone command is:

```powershell
& .\.venv\Scripts\python.exe -m xps_afruz.manuscript_plotter.studio
```

## Diagnostics and development checks

On Windows, diagnostic logs are stored at `%LOCALAPPDATA%\XPS-master\logs\xps-master.log`. Each file is limited to 2 MB and four older files are retained. An unexpected GUI-thread error displays this path so the failure can be investigated without modifying the source spectrum.

Run the same validation used by continuous integration with:

```powershell
& .\.venv\Scripts\python.exe -m compileall -q src tests
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m pip check
```
