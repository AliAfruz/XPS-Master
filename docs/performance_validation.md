# Phase 14 performance validation

Measured locally on 12 October 2026 with the same Python 3.12 runtime and
installed NumPy/SciPy dependencies for both versions. Each case uses 1,000
energy samples, two pseudo-Voigt components, eight independent parameters,
identical starting values and bounds, and three sequential repetitions.
Times below are medians; interpreter startup is excluded.

| Main peak area | Original engine | Phase 14 engine | Speed ratio |
|---:|---:|---:|---:|
| 300 | 46.7 ms | 18.2 ms | 2.56x |
| 3,000,000 | 49.1 ms | 18.8 ms | 2.62x |
| 300,000,000 | 61.2 ms | 18.2 ms | 3.36x |

Both engines converged in 12 optimizer residual evaluations in every case.
Both recovered the known peak centers to the printed floating-point precision.
Phase 14 relative RMS errors ranged from zero to `6.3e-17`.

This benchmark demonstrates reduced derivative/evaluation overhead for this
synthetic common-profile workload. It does not establish a universal speedup,
the speed of every line shape, or the scientific validity of real-data models.

The complete suite passes: **102 tests**, including 23 new scientific regression
cases added to the original 79-test suite. The included `XPS Survey.VGD` also
passed a real-file import and a uniform narrow-region fitting smoke check
(1,361 imported points; 20 ROI samples; four identifiable fit parameters).
Validation includes the offscreen GUI, project migration, advanced fitting,
Excel exports, and publication rendering. Runtime paths and user settings are
isolated during test execution.
