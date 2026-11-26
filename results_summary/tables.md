Speed regimes at the forecast origin: free flow at 55 mph or more, congested below 25 mph. 60-minute horizon, target coverage 0.90.

**Test error and regime shares**

| Dataset, forecaster | Test points | MAE (mph) | MAE free / transition / congested | Share free / transition / congested |
|---|---|---|---|---|
| PeMS04 GRU | 1,039,809 | 2.61 | 2.01 / 9.35 / 15.63 | 0.928 / 0.059 / 0.012 |
| PeMS04 STID | 1,039,809 | 1.88 | 1.50 / 6.26 / 9.01 | 0.928 / 0.059 / 0.012 |
| PeMS08 GRU | 605,200 | 2.07 | 1.62 / 8.32 / 15.58 | 0.939 / 0.055 / 0.006 |
| PeMS08 STID | 605,200 | 1.56 | 1.20 / 6.26 / 14.31 | 0.939 / 0.055 / 0.006 |
| METR-LA GRU | 1,209,566 | 5.22 | 3.62 / 9.42 / 14.78 | 0.775 / 0.170 / 0.055 |
| METR-LA STID | 1,209,566 | 3.62 | 2.66 / 6.07 / 9.64 | 0.775 / 0.170 / 0.055 |
| PEMS-BAY GRU | 3,384,116 | 2.92 | 2.03 / 9.73 / 16.33 | 0.902 / 0.079 / 0.020 |
| PEMS-BAY STID | 3,384,116 | 2.05 | 1.54 / 6.06 / 9.38 | 0.902 / 0.079 / 0.020 |

**Winkler interval score (lower is better)**

| Method | PeMS04 GRU | PeMS04 STID | PeMS08 GRU | PeMS08 STID | METR-LA GRU | METR-LA STID | PEMS-BAY GRU | PEMS-BAY STID |
|---|---|---|---|---|---|---|---|---|
| Split conformal | 27.80 | 20.41 | 22.26 | 18.29 | 49.99 | 39.81 | 32.87 | 22.38 |
| Mondrian on regime | 21.59 | 17.61 | 17.75 | 15.25 | 40.55 | 35.49 | 23.97 | 18.62 |
| Normalized conformal | 19.60 | 15.52 | 15.87 | 13.28 | 39.72 | 32.40 | 21.46 | 15.58 |
| CQR | 18.05 | 12.54 | 14.34 | 10.10 | 34.70 | 22.12 | 19.26 | 12.93 |
| CQR, per-regime correction | 18.05 | 12.52 | 14.33 | 10.03 | 34.67 | **22.11** | 19.25 | 12.93 |
| Refined CQR, per-regime correction | **14.90** | **12.14** | 11.71 | **9.73** | **26.10** | 22.22 | **14.78** | **12.32** |
| Two-stage on point forecast | 14.90 | 13.25 | **11.65** | 10.88 | 26.17 | 24.57 | 14.90 | 13.22 |
| Climatology (no forecast) | 24.98 | 24.98 | 21.46 | 21.46 | 39.80 | 39.80 | 27.81 | 27.81 |

**Coverage in the congested regime (target 0.90)**

| Method | PeMS04 GRU | PeMS04 STID | PeMS08 GRU | PeMS08 STID | METR-LA GRU | METR-LA STID | PEMS-BAY GRU | PEMS-BAY STID |
|---|---|---|---|---|---|---|---|---|
| Split conformal | 0.154 | 0.463 | 0.114 | 0.272 | 0.481 | 0.648 | 0.159 | 0.427 |
| Mondrian on regime | 0.902 | 0.892 | 0.919 | 0.853 | 0.903 | 0.889 | 0.907 | 0.887 |
| Normalized conformal | 0.939 | 0.881 | 0.851 | 0.798 | 0.839 | 0.851 | 0.952 | 0.887 |
| CQR | 0.911 | 0.874 | 0.918 | 0.789 | 0.884 | 0.885 | 0.905 | 0.899 |
| CQR, per-regime correction | 0.913 | 0.895 | 0.919 | 0.912 | 0.899 | 0.901 | 0.908 | 0.894 |
| Refined CQR, per-regime correction | 0.883 | 0.889 | 0.917 | 0.913 | 0.893 | 0.903 | 0.895 | 0.887 |
| Two-stage on point forecast | 0.894 | 0.888 | 0.928 | 0.926 | 0.896 | 0.893 | 0.896 | 0.886 |
| Climatology (no forecast) | 0.924 | 0.924 | 0.886 | 0.886 | 0.909 | 0.909 | 0.911 | 0.911 |

**Split conformal coverage by regime: measured against the formula (free / transition / congested)**

| Dataset, forecaster | Measured | Formula | Rule of thumb |
|---|---|---|---|
| PeMS04 GRU | 0.957 / 0.447 / 0.154 | 0.942 / 0.489 / 0.326 | 0.900 / 0.418 / 0.273 |
| PeMS04 STID | 0.936 / 0.527 / 0.463 | 0.938 / 0.524 / 0.394 | 0.900 / 0.459 / 0.339 |
| PeMS08 GRU | 0.938 / 0.382 / 0.114 | 0.930 / 0.418 / 0.245 | 0.900 / 0.374 / 0.216 |
| PeMS08 STID | 0.927 / 0.465 / 0.272 | 0.929 / 0.439 / 0.235 | 0.900 / 0.395 / 0.208 |
| METR-LA GRU | 0.955 / 0.699 / 0.481 | 0.960 / 0.701 / 0.524 | 0.900 / 0.578 / 0.411 |
| METR-LA STID | 0.937 / 0.761 / 0.648 | 0.951 / 0.735 / 0.568 | 0.900 / 0.637 / 0.473 |
| PEMS-BAY GRU | 0.963 / 0.488 / 0.159 | 0.953 / 0.470 / 0.326 | 0.900 / 0.381 / 0.258 |
| PEMS-BAY STID | 0.942 / 0.535 / 0.427 | 0.946 / 0.530 / 0.394 | 0.900 / 0.450 / 0.327 |

Mean absolute gap between measured and mixture formula over 24 regime values: 0.040
