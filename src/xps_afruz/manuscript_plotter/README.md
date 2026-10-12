# XPS-master Manuscript Plotter

This package is the maintained XPS Plot Studio 1.5 source adapted for XPS
XPS-master. The original standalone files remain unchanged.

Supported workbook layouts:

- Thermo Avantage-style Excel exports with binding/kinetic energy, envelope,
  background, residual, and fitted-peak/component columns.
- XPS-master complete-workspace exports. The importer skips the summary sheet,
  reads the active energy axis, measured intensity, accepted fit envelope,
  background and residual, and maps each accepted `+ background` component.
  Physics-comparison helper columns remain available in the audit workbook but
  are not mistaken for the accepted manuscript model.

Fitted spectra show a separate residual panel by default. It uses the same
font, axis, DPI, normalization, template, live-preview, and export system as
the main plot. Main and residual axes share one font family, 19-point axis
labels, and 14-point tick labels. The default panel title is 20.4 points,
exactly 20% larger than the original 17-point manuscript preset.
Both panels also share one X-axis object and identical physical left/right
boundaries, so energy ticks, toolbar zoom/pan, reversed limits, and pixel
coordinates remain synchronized even when the main panel is square.
Raster publication export defaults to 600 DPI and enforces 600 DPI as the
minimum; analysts may select higher values up to 2400 DPI. Older projects or
templates that retained a lower value are safely raised to 600 during export.

Run it directly with:

```powershell
& .\.venv\Scripts\python.exe -m xps_afruz.manuscript_plotter.studio
```

Or use the **Publish** runner tab (export current workspace, open an existing
workbook, or launch empty) from XPS-master. The same actions are mirrored
under **File → Manuscript Plotter**.
