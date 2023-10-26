# Neighbourhood ensemble copula coupling

The code implements 3 variants of ensemble copula coupling:

1. Standard ECC
2. Smoothed ECC
3. Neighborhood ECC

The script to produce the calibrated ensembles is `ecc.py`. It requires as inputs the gridded raw ensemble, and the gridded calibrated probabilistic forecast (e.g. from Rainforests, reliability calibration, or any other method).
The script `probability_output.py` thresholds the calibrated forecasts and produces gridded and site output for quantitative verification.
