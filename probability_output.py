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
from pathlib import Path

# paths
spot_cube_path = "/path/to/neighbor/cube"
working_dir = "/path/to/ecc/outputs/"
raw_ensemble_dir = "path/to/nwp/forecasts/"

models = ["ecmwf", "accessge3"]

thresholds = np.sort(np.array([500, 450, 400, 350, 300, 250, 200, 150, 125, 100, 75, 50, 35, 25, 15, 10, 7, 5, 2, 1, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01, 0.0])) / 1000
spot_cube = iris.load_cube(spot_cube_path)
start_date = dt.datetime(2021, 9, 1)
end_date = dt.datetime(2022, 9, 1)


def process(input_path, output_dir_grid, output_dir_spot, output_dir_site_ensemble, thresholds):
    input_cube = iris.load_cube(input_path)
    try:
        filename = os.path.basename(input_path)
        # mask NaNs in input cube
        input_cube.data = np.ma.masked_where(np.isnan(input_cube.data), input_cube.data)
        # site-extract ensemble
        output_cube = spot_extract.process(input_cube, spot_cube)
        save_netcdf(output_cube, os.path.join(output_dir_site_ensemble, filename))
        threshold_cube = threshold.process(input_cube, threshold_values=thresholds, comparison_operator=">=", collapse_coord="realization")
        output_cube = threshold_cube
        save_netcdf(output_cube, os.path.join(output_dir_grid, filename))        
        output_cube = spot_extract.process(output_cube, spot_cube)
        save_netcdf(output_cube, os.path.join(output_dir_spot, filename))
    except Exception as e:
        print(e)
        print(f"failed on {input_path}")


if __name__ == "__main__":

    args = []
    forecast_type = []
    overwrite_existing = True
    for model in models:
        base_dir = Path(working_dir) / model / "PT24H" / "regridded"
        input_dirs = list(base_dir.rglob("**/grid/"))
        input_dirs.append(Path(raw_ensemble_dir) / model / "combine" / "PT24H")
        for input_dir in input_dirs:
            if "combine" in str(input_dir):
                base_dir = Path(working_dir) / model / "PT24H" / "regridded" / "raw_ensemble"
            else:
                base_dir = os.path.split(input_dir)[0]
            output_dir_grid = os.path.join(base_dir, "grid_probability")
            output_dir_site = os.path.join(base_dir, "site_probability")
            output_dir_site_ensemble = os.path.join(base_dir, "site_ensemble")
            for dir in [output_dir_grid, output_dir_site, output_dir_site_ensemble]:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            for forecast_date in pd.date_range(start_date, end_date):
                valid_date = forecast_date + dt.timedelta(days=1)
                forecast_date_formatted = forecast_date.strftime("%Y%m%dT0000Z")
                valid_date_formatted = valid_date.strftime("%Y%m%dT0000Z")
                filename = f"{valid_date_formatted}-PT0024H00M-precipitation_accumulation-PT24H.nc"
                if "combine" in str(input_dir):
                    input_path = os.path.join(input_dir, forecast_date_formatted, filename)
                else:
                    input_path = os.path.join(input_dir, filename)
                output_files = [os.path.join(dir, filename) for dir in [output_dir_grid, output_dir_site, output_dir_site_ensemble]]
                all_exist = all([os.path.exists(p) for p in output_files])
                if os.path.exists(input_path) and (not(all_exist) or overwrite_existing):
                    args += [[input_path, output_dir_grid, output_dir_site, output_dir_site_ensemble, thresholds]]

    with Pool(70) as p:
        p.starmap(process, args)