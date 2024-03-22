# Neighbourhood ensemble copula coupling

The code implements 3 variants of ensemble copula coupling:

1. Standard ECC
2. Smoothed ECC
3. Neighborhood ECC
4. Tricube smoothing method of Scheuerer and Hamill (2018)
5. Regularized calibration method of Scheuerer and Hamill (2018)

The script to produce the calibrated ensembles is `ecc.py`. It requires as inputs the gridded raw ensemble, and the gridded calibrated probabilistic forecast (e.g. from Rainforests, reliability calibration, or any other method).
The script `probability_output.py` thresholds the calibrated forecasts and produces gridded and site output for quantitative verification.

### References

Scheuerer and Hamill (2018), Generating calibrated ensembles of physically realistic, high-resolution
precipitation forecast fields based on GEFS model output. Journal of Hydrometeorology, 19. https://doi.org/10.1175/JHM-D-18-0067.1
