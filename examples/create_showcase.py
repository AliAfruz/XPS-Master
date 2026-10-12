"""Reproduce the README figures with synthetic counts and the real fit engine.

Run from the repository root after installing the project. All generated data
are synthetic; component numbers do not represent chemical-state assignments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from xps_afruz import FitModel, PeakComponent, Spectrum, fit_spectrum
from xps_afruz.fitting.peak_shapes import evaluate_peak
from xps_afruz.metadata import AUTHOR_NAME, AUTHOR_EMAIL, AUTHOR_WEBSITE

SEED = 41407
COLOURS = ("#65c6b6", "#ef977e", "#b5a0eb")
CANVAS = "#101a24"
TEXT = "#eef2ef"
MUTED = "#9cabb8"
GOLD = "#deb573"


def synthetic_fit():
    energy = np.linspace(279.0, 293.0, 1000)
    background = 58.0 + 2.6 * (energy-energy[0])
    truth = ((284.45, 1.15, 1580.0, 0.30),
             (286.05, 1.40, 740.0, 0.42),
             (288.35, 1.55, 440.0, 0.50))
    expected = background.copy()
    guesses = []
    for index, (center, width, area, eta) in enumerate(truth, 1):
        expected += evaluate_peak(energy, profile="pseudo_voigt", center=center,
                                  fwhm=width, area=area, eta=eta)
        guesses.append(PeakComponent(f"Component {index:02d}", center-0.12, width*1.12,
            area*0.86, eta=0.40, vary_eta=True, center_bounds=(center-0.6, center+0.6),
            fwhm_bounds=(0.55, 2.4), area_bounds=(1.0, 4000.0)))
    counts = np.random.default_rng(SEED).poisson(expected).astype(float)
    spectrum = Spectrum(energy, counts, label="Synthetic fitting showcase",
        metadata={"source": "synthetic", "random_seed": SEED, "noise": "Poisson counts"})
    result = fit_spectrum(spectrum, FitModel(guesses, background), weighting="poisson")
    if not result.success or result.jacobian_rank != 12:
        raise RuntimeError("Showcase fit did not converge with identifiable parameters.")
    return spectrum, background, guesses, result, truth


def style_axis(axis):
    axis.set_facecolor(CANVAS)
    axis.spines[["top", "right"]].set_visible(False)
    for spine in axis.spines.values():
        spine.set_color("#41505e")
    axis.tick_params(colors=MUTED, labelsize=11, length=4, pad=7)
    axis.grid(axis="y", color="#354553", alpha=0.38, linewidth=0.7)
    axis.set_axisbelow(True)


def showcase(output: Path, spectrum, background, result):
    figure = plt.figure(figsize=(13.2, 8.7), facecolor=CANVAS)
    grid = figure.add_gridspec(2, 1, height_ratios=(4.1, 1.1),
        left=0.095, right=0.97, top=0.77, bottom=0.30, hspace=0.06)
    main = figure.add_subplot(grid[0])
    residual = figure.add_subplot(grid[1], sharex=main)
    x = spectrum.energy
    for axis in (main, residual):
        style_axis(axis)
    main.scatter(x, spectrum.intensity, s=5, c="#cad4dc", alpha=0.56,
                 linewidths=0, label="Measured counts", zorder=5)
    for (name, curve), colour in zip(result.components.items(), COLOURS):
        main.fill_between(x, background, background+curve, color=colour, alpha=0.25)
        main.plot(x, background+curve, color=colour, lw=1.8, label=name)
    main.plot(x, result.fitted_intensity, color=TEXT, lw=2.0, label="Total fit", zorder=6)
    main.plot(x, background, color=GOLD, lw=1.3, linestyle=(0, (5, 4)), label="Background")
    main.set_ylabel("Intensity / counts", color=TEXT, fontsize=12, labelpad=12)
    main.set_ylim(0, np.max(spectrum.intensity)*1.18)
    main.set_xlim(292.8, 279.2)
    main.tick_params(labelbottom=False)
    legend = main.legend(loc="upper left", ncol=3, fontsize=9.5, frameon=False,
                         handlelength=2.0, columnspacing=1.2)
    for text in legend.get_texts():
        text.set_color(TEXT)
    residual.axhline(0, color=GOLD, alpha=0.7, lw=0.9)
    residual.plot(x, result.residual, color="#83b1c6", lw=0.72)
    residual.set_ylabel("Fit − data", color=TEXT, fontsize=11, labelpad=12)
    residual.set_xlabel("Binding energy / eV", color=TEXT, fontsize=12, labelpad=12)
    for index, ((name, _), colour) in enumerate(zip(result.components.items(), COLOURS)):
        center = result.parameter_values[f"{name}.center"]
        figure.text(0.105+index*0.30, 0.145, f"{index+1:02d}  {name.upper()}", color=colour,
                    fontsize=11, weight="bold")
        figure.text(0.105+index*0.30, 0.105,
            f"{center:.3f} eV  ·  FWHM {result.parameter_values[f'{name}.fwhm']:.3f} eV",
            color=TEXT, fontsize=10.5)
        figure.text(0.105+index*0.30, 0.073,
            f"Area {result.parameter_values[f'{name}.area']:.1f}  ·  η {result.parameter_values[f'{name}.eta']:.3f}",
            color=MUTED, fontsize=10)
    figure.text(0.095, 0.94, "XPS-MASTER", color=GOLD, fontsize=12, weight="bold",
                fontfamily="DejaVu Sans")
    figure.text(0.095, 0.875, "See the spectrum. Inspect the fit.", color=TEXT,
                fontsize=27, weight="bold")
    figure.text(0.095, 0.826, "Synthetic Poisson counts  /  3 pseudo-Voigt components  /  retained residuals",
                color=MUTED, fontsize=11)
    figure.text(0.97, 0.94, "SCIENTIFIC FITTING SHOWCASE", color=MUTED, fontsize=9, ha="right")
    figure.text(0.095, 0.205,
        f"POISSON LIKELIHOOD   ·   Pearson χ² / DOF {result.reduced_statistical_chi_square:.3f}"
        f"   ·   Jacobian rank {result.jacobian_rank}/{result.free_parameter_count}",
        color=GOLD, fontsize=10.5)
    figure.text(0.095, 0.027, f"{AUTHOR_NAME}  ·  {AUTHOR_EMAIL}  ·  chemitro.ir", color=MUTED, fontsize=9)
    figure.text(0.97, 0.027, "Reproducible synthetic example · not an experimental assignment", color=MUTED,
                fontsize=8.5, ha="right")
    figure.savefig(output/"fitting-showcase.png", dpi=180, facecolor=CANVAS,
                   metadata={"Author": AUTHOR_NAME, "Title": "XPS-master synthetic fitting showcase"})
    figure.savefig(output/"fitting-showcase.svg", facecolor=CANVAS,
                   metadata={"Creator": AUTHOR_NAME, "Title": "Reproducible synthetic XPS fit"})
    svg_path = output/"fitting-showcase.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8", newline="\n")
    plt.close(figure)


def banner(output: Path, spectrum, background, result):
    figure = plt.figure(figsize=(14.4, 4.5), facecolor=CANVAS)
    axis = figure.add_axes((0.46, 0.19, 0.51, 0.69), facecolor=CANVAS)
    x = spectrum.energy
    for (name, curve), colour in zip(result.components.items(), COLOURS):
        axis.fill_between(x, background, background+curve, color=colour, alpha=0.38)
        axis.plot(x, background+curve, color=colour, lw=1.8)
    axis.scatter(x[::3], spectrum.intensity[::3], color="#c8d4de", s=6, alpha=0.40, linewidths=0)
    axis.plot(x, result.fitted_intensity, color=TEXT, lw=1.9)
    axis.plot(x, background, color=GOLD, lw=0.9, linestyle=(0, (4, 4)))
    axis.set_xlim(292.8, 279.2)
    axis.set_ylim(0, spectrum.intensity.max()*1.12)
    axis.axis("off")
    figure.text(0.055, 0.83, "XPS ANALYSIS  /  SCIENTIFIC SOFTWARE", color=GOLD, fontsize=11, weight="bold")
    figure.text(0.055, 0.59, "XPS-master", color=TEXT, fontsize=39, weight="bold")
    figure.text(0.055, 0.40, "SURFACE ANALYSIS", color=MUTED, fontsize=15, weight="bold")
    figure.text(0.055, 0.255, "Fit  ·  Survey  ·  Workspace  ·  Publish", color=MUTED, fontsize=13)
    figure.text(0.055, 0.095, f"{AUTHOR_NAME}  |  chemitro.ir", color=GOLD, fontsize=11)
    figure.text(0.97, 0.095, "Real engine · synthetic spectrum", color=MUTED, fontsize=9, ha="right")
    figure.savefig(output/"xps-master-banner.png", dpi=180, facecolor=CANVAS,
                   metadata={"Author": AUTHOR_NAME, "Title": "XPS-master"})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=Path(__file__).resolve().parents[1]/"docs"/"assets")
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    spectrum, background, _, result, truth = synthetic_fit()
    showcase(options.output, spectrum, background, result)
    banner(options.output, spectrum, background, result)
    data = Path(__file__).resolve().parent/"data"
    data.mkdir(exist_ok=True)
    csv_path = data/"synthetic_showcase.csv"
    np.savetxt(csv_path, np.column_stack((spectrum.energy, spectrum.intensity)), delimiter=",",
               header="Binding Energy (eV),Intensity (counts)", comments="", fmt=("%.8f", "%.0f"))
    provenance = {"kind": "synthetic illustration", "author": AUTHOR_NAME,
        "email": AUTHOR_EMAIL, "website": AUTHOR_WEBSITE, "random_seed": SEED,
        "samples": spectrum.size, "profile": "pseudo_voigt", "weighting": "poisson",
        "count_scale": 1.0, "background": {"method": "known synthetic linear", "intercept": 58.0, "slope": 2.6},
        "truth": [{"center": c, "fwhm": w, "area": a, "eta": e} for c,w,a,e in truth],
        "parameters": dict(result.parameter_values), "standard_errors": dict(result.standard_errors),
        "objective": result.objective_name, "deviance": result.objective_value,
        "pearson_chi_square": result.statistical_chi_square,
        "degrees_of_freedom": result.degrees_of_freedom,
        "jacobian_rank": result.jacobian_rank, "fit_warnings": result.fit_warnings,
        "data_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}
    (options.output/"showcase-fit.json").write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Created figures at {options.output}; Poisson fit converged, rank {result.jacobian_rank}.")


if __name__ == "__main__":
    main()
