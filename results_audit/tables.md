### Table A. Coverage in transition windows (window-based regimes of Wang et al.), 60 minutes, target 0.90

| Dataset, forecaster | Share of transition windows | Split conformal | CQR | CQR, per-regime correction | Historical baseline | Same, 30 most congestion-prone sensors: CQR, per-regime |
|---|---|---|---|---|---|---|
| PeMS04 GRU | 0.115 | 0.462 | 0.772 | 0.773 | 0.727 | 0.807 |
| PeMS04 STID | 0.115 | 0.489 | 0.813 | 0.827 | 0.727 | 0.855 |
| PeMS08 GRU | 0.101 | 0.415 | 0.766 | 0.775 | 0.713 | 0.800 |
| PeMS08 STID | 0.101 | 0.454 | 0.780 | 0.813 | 0.713 | 0.867 |
| METR-LA GRU | 0.378 | 0.732 | 0.836 | 0.844 | 0.843 | 0.905 |
| METR-LA STID | 0.378 | 0.750 | 0.876 | 0.880 | 0.843 | 0.903 |
| PEMS-BAY GRU | 0.145 | 0.488 | 0.797 | 0.805 | 0.717 | 0.869 |
| PEMS-BAY STID | 0.145 | 0.497 | 0.840 | 0.843 | 0.717 | 0.869 |

For reference, Wang et al. report 60-minute transition coverage of 0.68 to 0.80 for three foundation models on METR-LA and 0.55 to 0.72 on PEMS-BAY, rising to at most 0.85 and 0.77 with their corrections (their Table 2; 30 sensors per dataset; read from the arXiv HTML and not yet checked against the PDF).

### Table B. Calibrating on predicted transition risk

| Dataset, forecaster | Classifier AUC | Transition-window coverage: CQR per origin regime | CQR per risk bin | Winkler: CQR per origin regime | CQR per risk bin |
|---|---|---|---|---|---|
| PeMS04 GRU | 0.960 | 0.773 | 0.810 | 18.05 | 17.85 |
| PeMS04 STID | 0.968 | 0.825 | 0.830 | 12.52 | 12.51 |
| PeMS08 GRU | 0.970 | 0.773 | 0.806 | 14.33 | 14.21 |
| PeMS08 STID | 0.976 | 0.816 | 0.831 | 10.03 | 9.97 |
| METR-LA GRU | 0.930 | 0.843 | 0.854 | 34.68 | 34.48 |
| METR-LA STID | 0.940 | 0.887 | 0.897 | 22.14 | 22.16 |
| PEMS-BAY GRU | 0.970 | 0.804 | 0.846 | 19.26 | 19.17 |
| PEMS-BAY STID | 0.975 | 0.844 | 0.854 | 12.93 | 12.92 |

### Table C. Coverage inside risk deciles, split by what happened next (CQR per risk bin)

**METR-LA STID**

| Risk decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Share of windows that were transitions | 0.01 | 0.02 | 0.03 | 0.06 | 0.11 | 0.21 | 0.40 | 0.68 | 0.93 | 1.00 |
| Coverage, all points | 0.91 | 0.90 | 0.91 | 0.91 | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 |
| Coverage, a transition followed | 0.65 | 0.68 | 0.70 | 0.71 | 0.75 | 0.80 | 0.86 | 0.91 | 0.93 | 0.92 |
| Coverage, no transition followed | 0.92 | 0.91 | 0.92 | 0.92 | 0.94 | 0.95 | 0.96 | 0.94 | 0.84 | 0.61 |

**PEMS-BAY STID**

| Risk decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Share of windows that were transitions | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.02 | 0.07 | 0.42 | 0.96 |
| Coverage, all points | 0.89 | 0.89 | 0.88 | 0.89 | 0.90 | 0.90 | 0.90 | 0.90 | 0.91 | 0.90 |
| Coverage, a transition followed | 0.44 | 0.34 | 0.32 | 0.29 | 0.34 | 0.36 | 0.39 | 0.52 | 0.85 | 0.91 |
| Coverage, no transition followed | 0.89 | 0.89 | 0.89 | 0.90 | 0.90 | 0.91 | 0.91 | 0.92 | 0.95 | 0.75 |

### Table C2. Coverage by predicted transition-risk decile: lowest decile value, and value in the highest-risk decile

| Dataset, forecaster | Split conformal | Mondrian on origin regime | CQR | CQR per origin regime | CQR per risk bin |
|---|---|---|---|---|---|
| PeMS04 GRU | 0.479 / 0.479 | 0.707 / 0.826 | 0.835 / 0.889 | 0.835 / 0.888 | 0.899 / 0.901 |
| PeMS04 STID | 0.512 / 0.512 | 0.683 / 0.777 | 0.870 / 0.871 | 0.870 / 0.888 | 0.889 / 0.889 |
| PeMS08 GRU | 0.466 / 0.466 | 0.755 / 0.769 | 0.853 / 0.853 | 0.861 / 0.862 | 0.897 / 0.898 |
| PeMS08 STID | 0.495 / 0.495 | 0.729 / 0.729 | 0.833 / 0.833 | 0.853 / 0.875 | 0.888 / 0.895 |
| METR-LA GRU | 0.665 / 0.736 | 0.758 / 0.911 | 0.841 / 0.901 | 0.845 / 0.914 | 0.885 / 0.895 |
| METR-LA STID | 0.697 / 0.769 | 0.762 / 0.907 | 0.896 / 0.909 | 0.903 / 0.916 | 0.905 / 0.916 |
| PEMS-BAY GRU | 0.490 / 0.490 | 0.678 / 0.835 | 0.842 / 0.882 | 0.842 / 0.892 | 0.900 / 0.907 |
| PEMS-BAY STID | 0.502 / 0.502 | 0.679 / 0.772 | 0.878 / 0.895 | 0.878 / 0.899 | 0.885 / 0.903 |

### Table D. Online calibration with a 60-minute feedback delay (Winkler score)

| Dataset, forecaster | Score | Minutes misses stay correlated | Delay-to-memory ratio | Split or CQR | Batch per regime | ACI, delayed (best of four step sizes) | Per-sensor per-regime tracker, delayed (best of three) | ACI, one-step feedback |
|---|---|---|---|---|---|---|---|---|
| PeMS04 GRU | absolute residual | 45 | 1.3 | 27.80 | 21.59 | 24.81 | 20.49 | 14.98 |
| PeMS04 GRU | CQR | 20 | 3.0 | 18.05 | 18.05 | 17.98 | 18.04 | 15.90 |
| PeMS04 STID | absolute residual | 30 | 2.0 | 20.41 | 17.61 | 18.75 | 16.27 | 12.45 |
| PeMS04 STID | CQR | 10 | 6.0 | 12.54 | 12.52 | 12.54 | 12.43 | 13.38 |
| PeMS08 GRU | absolute residual | 30 | 2.0 | 22.26 | 17.75 | 19.65 | 16.49 | 12.54 |
| PeMS08 GRU | CQR | 20 | 3.0 | 14.34 | 14.33 | 14.40 | 14.31 | 13.30 |
| PeMS08 STID | absolute residual | 35 | 1.7 | 18.29 | 15.25 | 16.11 | 13.16 | 10.53 |
| PeMS08 STID | CQR | 10 | 6.0 | 10.10 | 10.03 | 10.20 | 9.94 | 10.73 |
| METR-LA GRU | absolute residual | 35 | 1.7 | 49.99 | 40.55 | 45.42 | 38.88 | 30.89 |
| METR-LA GRU | CQR | 10 | 6.0 | 34.70 | 34.67 | 34.71 | 34.45 | 30.76 |
| METR-LA STID | absolute residual | 20 | 3.0 | 39.81 | 35.49 | 37.03 | 33.86 | 25.61 |
| METR-LA STID | CQR | 5 | 12.0 | 22.12 | 22.11 | 22.08 | 22.09 | 22.15 |
| PEMS-BAY GRU | absolute residual | 70 | 0.9 | 32.87 | 23.97 | 30.62 | 23.06 | 16.38 |
| PEMS-BAY GRU | CQR | 15 | 4.0 | 19.26 | 19.25 | 19.20 | 19.26 | 17.50 |
| PEMS-BAY STID | absolute residual | 45 | 1.3 | 22.38 | 18.62 | 21.41 | 17.70 | 13.36 |
| PEMS-BAY STID | CQR | 10 | 6.0 | 12.93 | 12.93 | 13.21 | 12.71 | 13.81 |

Step sizes were picked after seeing test results, which favours the online methods.

### Table E. Where the rule of thumb's error comes from (split conformal coverage by regime)

| Regime | Mean gap: Laplace rule | Common shape, no parametric form | Regime's own calibration distribution | Mean size of: drift | shape difference | Laplace misfit |
|---|---|---|---|---|---|---|
| free | 0.009 | 0.004 | 0.004 | 0.004 | 0.003 | 0.006 |
| transition | 0.039 | 0.117 | 0.017 | 0.017 | 0.111 | 0.109 |
| congested | 0.112 | 0.151 | 0.012 | 0.012 | 0.142 | 0.108 |

| Dataset, forecaster | Distance between congested and pooled residual shape (KS) | Shape term, congested | Laplace term, congested |
|---|---|---|---|
| PeMS04 GRU | 0.284 | -0.265 | +0.077 |
| PeMS04 STID | 0.060 | -0.023 | +0.119 |
| PeMS08 GRU | 0.266 | -0.223 | +0.084 |
| PeMS08 STID | 0.088 | -0.053 | +0.162 |
| METR-LA GRU | 0.180 | -0.156 | +0.078 |
| METR-LA STID | 0.059 | -0.044 | +0.135 |
| PEMS-BAY GRU | 0.347 | -0.335 | +0.108 |
| PEMS-BAY STID | 0.070 | -0.039 | +0.098 |

### Table F. Day-block bootstrap, 95% intervals (congested regime at the origin)

| Dataset, forecaster | Test days | Split conformal | CQR | CQR, per-regime correction |
|---|---|---|---|---|
| PeMS04 GRU | 12 | 0.154 [0.143, 0.165] | 0.911 [0.899, 0.920] | 0.913 [0.901, 0.922] |
| PeMS04 STID | 12 | 0.463 [0.414, 0.501] | 0.874 [0.858, 0.887] | 0.895 [0.881, 0.906] |
| PeMS08 GRU | 13 | 0.114 [0.096, 0.131] | 0.918 [0.901, 0.935] | 0.919 [0.902, 0.935] |
| PeMS08 STID | 13 | 0.272 [0.221, 0.325] | 0.789 [0.749, 0.831] | 0.912 [0.878, 0.934] |
| METR-LA GRU | 24 | 0.481 [0.465, 0.496] | 0.884 [0.873, 0.895] | 0.899 [0.889, 0.908] |
| METR-LA STID | 24 | 0.648 [0.625, 0.671] | 0.885 [0.867, 0.904] | 0.901 [0.883, 0.918] |
| PEMS-BAY GRU | 37 | 0.159 [0.153, 0.165] | 0.905 [0.897, 0.914] | 0.908 [0.900, 0.916] |
| PEMS-BAY STID | 37 | 0.427 [0.409, 0.444] | 0.899 [0.881, 0.913] | 0.894 [0.876, 0.908] |

### Table G (lowest coverage over regime x 4-hour cells), CQR score with five ways of computing the correction

| Dataset, forecaster | pooled | regime | regime x hour | sensor x regime | multigroup |
|---|---|---|---|---|---|
| PeMS04 GRU | 0.851 | 0.850 | 0.888 | 0.845 | 0.875 |
| PeMS04 STID | 0.862 | 0.878 | 0.882 | 0.868 | 0.886 |
| PeMS08 GRU | 0.818 | 0.845 | 0.880 | 0.865 | 0.852 |
| PeMS08 STID | 0.728 | 0.790 | 0.844 | 0.803 | 0.790 |
| METR-LA GRU | 0.819 | 0.832 | 0.884 | 0.840 | 0.851 |
| METR-LA STID | 0.865 | 0.879 | 0.883 | 0.874 | 0.887 |
| PEMS-BAY GRU | 0.653 | 0.693 | 0.884 | 0.676 | 0.661 |
| PEMS-BAY STID | 0.764 | 0.779 | 0.872 | 0.778 | 0.789 |

### Table G (coverage of the worst sensor), CQR score with five ways of computing the correction

| Dataset, forecaster | pooled | regime | regime x hour | sensor x regime | multigroup |
|---|---|---|---|---|---|
| PeMS04 GRU | 0.781 | 0.782 | 0.776 | 0.825 | 0.777 |
| PeMS04 STID | 0.511 | 0.518 | 0.512 | 0.581 | 0.520 |
| PeMS08 GRU | 0.785 | 0.794 | 0.795 | 0.834 | 0.806 |
| PeMS08 STID | 0.635 | 0.632 | 0.629 | 0.623 | 0.618 |
| METR-LA GRU | 0.749 | 0.751 | 0.741 | 0.854 | 0.740 |
| METR-LA STID | 0.807 | 0.816 | 0.816 | 0.768 | 0.820 |
| PEMS-BAY GRU | 0.820 | 0.817 | 0.812 | 0.820 | 0.817 |
| PEMS-BAY STID | 0.417 | 0.417 | 0.414 | 0.697 | 0.426 |

### Table G (Winkler score), CQR score with five ways of computing the correction

| Dataset, forecaster | pooled | regime | regime x hour | sensor x regime | multigroup |
|---|---|---|---|---|---|
| PeMS04 GRU | 18.05 | 18.05 | 17.87 | 17.79 | 17.87 |
| PeMS04 STID | 12.54 | 12.52 | 12.52 | 12.40 | 12.51 |
| PeMS08 GRU | 14.34 | 14.33 | 14.21 | 14.12 | 14.22 |
| PeMS08 STID | 10.10 | 10.03 | 10.00 | 9.95 | 10.01 |
| METR-LA GRU | 34.70 | 34.67 | 34.47 | 34.10 | 34.53 |
| METR-LA STID | 22.12 | 22.11 | 22.09 | 22.09 | 22.11 |
| PEMS-BAY GRU | 19.26 | 19.25 | 19.09 | 19.04 | 19.08 |
| PEMS-BAY STID | 12.93 | 12.93 | 12.92 | 12.70 | 12.92 |

### Table H. Typical distance of conditional coverage from 0.90 (classifier-based diagnostic, lower is better)

| Dataset, forecaster | split conformal | Mondrian on regime | CQR | CQR per regime | CQR per sensor x regime |
|---|---|---|---|---|---|
| PeMS04 GRU | 0.179 | 0.135 | 0.106 | 0.105 | 0.106 |
| PeMS04 STID | 0.164 | 0.136 | 0.070 | 0.070 | 0.070 |
| PeMS08 GRU | 0.206 | 0.170 | 0.148 | 0.148 | 0.141 |
| PeMS08 STID | 0.181 | 0.148 | 0.094 | 0.091 | 0.092 |
| METR-LA GRU | 0.205 | 0.147 | 0.109 | 0.108 | 0.099 |
| METR-LA STID | 0.164 | 0.130 | 0.038 | 0.034 | 0.027 |
| PEMS-BAY GRU | 0.210 | 0.160 | 0.108 | 0.106 | 0.101 |
| PEMS-BAY STID | 0.192 | 0.154 | 0.080 | 0.080 | 0.058 |

### Table I. STID at 15, 30 and 60 minutes

| Dataset | Horizon (min) | MAE | Split conformal congested coverage | CQR per-regime congested coverage | Minutes misses stay correlated | Delay-to-memory ratio | Winkler: Mondrian | ACI, delayed | ACI, one-step feedback |
|---|---|---|---|---|---|---|---|---|---|
| PeMS04 | 15 | 1.30 | 0.595 | 0.898 | 10 | 1.50 | 10.98 | 11.05 | 10.17 |
| PeMS04 | 30 | 1.57 | 0.515 | 0.898 | 20 | 1.50 | 14.06 | 14.78 | 11.74 |
| PeMS04 | 60 | 1.88 | 0.463 | 0.895 | 30 | 2.00 | 17.61 | 18.75 | 12.45 |
| PeMS08 | 15 | 1.03 | 0.479 | 0.890 | 10 | 1.50 | 9.49 | 8.79 | 8.09 |
| PeMS08 | 30 | 1.31 | 0.336 | 0.906 | 20 | 1.50 | 12.33 | 12.41 | 9.30 |
| PeMS08 | 60 | 1.56 | 0.272 | 0.912 | 35 | 1.71 | 15.25 | 16.11 | 9.95 |
| METR-LA | 15 | 2.81 | 0.719 | 0.902 | 5 | 3.00 | 24.17 | 25.70 | 23.67 |
| METR-LA | 30 | 3.20 | 0.671 | 0.898 | 10 | 3.00 | 29.48 | 31.34 | 24.98 |
| METR-LA | 60 | 3.62 | 0.648 | 0.901 | 20 | 3.00 | 35.49 | 37.03 | 25.61 |
| PEMS-BAY | 60 | 2.05 | 0.427 | 0.894 | 45 | 1.33 | 18.62 | 21.41 | 13.36 |

### Table J. Predicting a transition within the hour from a free-flow origin (PeMS04, PeMS08)

| Dataset | Features | With STID forecast | AUC | Average precision | Share caught at 5% false alarms |
|---|---|---|---|---|---|
| PeMS04 | A speed | no | 0.918 | 0.489 | 0.631 |
| PeMS04 | A speed | yes | 0.929 | 0.567 | 0.692 |
| PeMS04 | B + occupancy, flow | no | 0.927 | 0.526 | 0.669 |
| PeMS04 | B + occupancy, flow | yes | 0.934 | 0.605 | 0.708 |
| PeMS04 | C + FD | no | 0.931 | 0.522 | 0.682 |
| PeMS04 | C + FD | yes | 0.934 | 0.587 | 0.705 |
| PeMS08 | A speed | no | 0.943 | 0.557 | 0.731 |
| PeMS08 | A speed | yes | 0.950 | 0.640 | 0.772 |
| PeMS08 | B + occupancy, flow | no | 0.948 | 0.589 | 0.761 |
| PeMS08 | B + occupancy, flow | yes | 0.953 | 0.627 | 0.789 |
| PeMS08 | C + FD | no | 0.948 | 0.582 | 0.764 |
| PeMS08 | C + FD | yes | 0.955 | 0.643 | 0.798 |
