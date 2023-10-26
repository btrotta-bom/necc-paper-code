"""Calculate thresholded probabilities from processed ensembles, suitable 
for calculating the fractional skill score. Produce gridded and site outputs."""

from improver.cli import spot_extract, threshold
import os
import iris
from improver.utilities.save import save_netcdf
from multiprocessing import Pool
import numpy as np
from numba import njit
import datetime as dt
import pandas as pd

model = "ecmwf"
ecc_dir = f"/path/to/ecc/outputs"
input_dir_raw = f"/path/to/nwp/forecasts/"

thresholds = np.sort(np.array([500, 450, 400, 350, 300, 250, 200, 150, 125, 100, 75, 50, 35, 25, 15, 10, 7, 5, 2, 1, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01, 0.0])) / 1000
spot_cube = iris.load_cube("path/to/neighborhood/cube")
start_date = dt.datetime(2021, 9, 1)
end_date = dt.datetime(2022, 9, 1)


def process(input_path, output_dir_grid, output_dir_spot, thresholds):
    input_cube = iris.load_cube(input_path)
    try:
        # mask NaNs in input cube
        input_cube.data = np.ma.masked_where(np.isnan(input_cube.data), input_cube.data)
        threshold_cube = threshold.process(input_cube, threshold_values=thresholds, comparison_operator=">=", collapse_coord="realization")
        output_cube = threshold_cube
        filename = os.path.basename(input_path)
        save_netcdf(output_cube, os.path.join(output_dir_grid, filename))        
        output_cube = spot_extract.process(output_cube, spot_cube)
        save_netcdf(output_cube, os.path.join(output_dir_spot, filename))
    except:
        print(f"failed on {input_path}")


if __name__ == "__main__":

    methods = ["ecc", "smoothed_ecc", "necc", "raw_ensemble"]
    for method in methods:
        base_dir = os.path.split(ecc_dir)[0]
        output_dir_grid = os.path.join(base_dir, "grid_probability", method)
        output_dir_site = os.path.join(base_dir, "site_probability", method)
        for dir in [output_dir_grid, output_dir_site]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    args = []
    forecast_type = []
    input_dirs = {method: os.path.join(ecc_dir, method) for method in ["ecc", "smoothed_ecc", "necc"]}
    input_dirs["raw_ensemble"] = input_dir_raw
    for method, input_dir in input_dirs.items():
        for forecast_date in pd.date_range(start_date, end_date):
            valid_date = forecast_date + dt.timedelta(days=1)
            forecast_date_formatted = forecast_date.strftime("%Y%m%dT0000Z")
            valid_date_formatted = valid_date.strftime("%Y%m%dT0000Z")
            filename = f"{valid_date_formatted}-PT0024H00M-precipitation_accumulation-PT24H.nc"
            if "combine" in input_dir:
                input_path = os.path.join(input_dir, forecast_date_formatted, filename)
            else:
                input_path = os.path.join(input_dir, filename)
            dirs = os.path.split(ecc_dir)[0]
            output_dir_grid = os.path.join(base_dir, "grid_probability", method)
            output_dir_site = os.path.join(base_dir, "site_probability", method)
            if os.path.exists(input_path):
                args += [[input_path, output_dir_grid, output_dir_site, thresholds]]

    with Pool(70) as p:
        p.starmap(process, args)