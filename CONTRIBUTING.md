# Contributing to XPS-master

Thanks for helping improve reproducible XPS analysis.

## Development setup

Create a Python 3.10 or newer virtual environment, then install the project:

```shell
python -m pip install -e ".[dev]"
python -m pytest -q
```

GUI tests run offscreen. On Linux, set `QT_QPA_PLATFORM=offscreen` when a display
is unavailable. The GitHub workflow validates Windows Python 3.10 and 3.12.

## Scientific changes

Describe the problem, the mathematical assumptions, and the resulting behavior.
Add a known-truth synthetic regression case for changes to fitting, weighting,
constraints, calibration, uncertainty, or model scoring. Include a primary
method reference when a change depends on a published statistical method.

Keep calculations independent of Qt. Extend the relevant controller rather
than adding unrelated responsibilities to `ui/main_window.py`. Preserve saved
project compatibility, explicit analyst choices, cancellation behavior, and
the accepted-curve contract used by Excel export and the Manuscript Plotter.

## Reporting an issue

Include the software version, operating system, Python version, reproduction
steps, expected behavior, and the actual result. A small synthetic example is
usually the most useful way to reproduce a fitting issue.

For questions about the project, contact **Ali Afruz** at
[a.a.afruz@gmail.com](mailto:a.a.afruz@gmail.com) or visit
[chemitro.ir](https://chemitro.ir).
