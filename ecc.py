import iris
import numpy as np
import os
from multiprocessing import Pool
from numba import njit
import pandas as pd
import datetime as dt

from improver.ensemble_copula_coupling.ensemble_copula_coupling import ConvertProbabilitiesToPercentiles
from improver.utilities.save import save_netcdf
from scipy.signal import convolve2d

model = "ecmwf"
input_dir_raw = f"/path/to/nwp/forecasts/"
input_dir_cal = f"/path/to/calibrated/probability/forecasts/"
output_dir = f"/path/to/output/"
start_date = dt.datetime(2021, 9, 1)
end_date = dt.datetime(2022, 9, 1)
patch_width = 9
stride = 1


def ecc(raw_ensemble: iris.cube.Cube, calibrated_probabilities: iris.cube.Cube) -> iris.cube.Cube:
    """Apply ECC to a single forecast.

    Args:
        raw_ensemble: cube with dimensions realization, x dim, y dim
        calibrated_probabilities: cube with dimensions threshold, x dim, y dim
    Returns:
        cube with same dimensions as raw_ensemble
    """

    # convert probabilties to realizations
    num_realizations = len(raw_ensemble.coord("realization").points)
    calibrated_ensemble = ConvertProbabilitiesToPercentiles().process(
            calibrated_probabilities, no_of_percentiles=num_realizations
        )

    # apply ecc
    raw_ensemble_sort_ind = np.argsort(raw_ensemble.data, axis=0, kind="stable")
    raw_ensemble_reverse_sort_ind = np.argsort(raw_ensemble_sort_ind, axis=0, kind="stable")
    calibrated_ensemble_sorted = np.take_along_axis(calibrated_ensemble.data, raw_ensemble_reverse_sort_ind, axis=0)

    # put nans back in original locations
    original_nan_ind = np.isnan(raw_ensemble.data)
    new_nan_ind = np.isnan(calibrated_ensemble_sorted)
    calibrated_ensemble_sorted[~original_nan_ind] = calibrated_ensemble_sorted[~new_nan_ind]
    calibrated_ensemble_sorted[original_nan_ind] = np.nan

    output_cube = raw_ensemble.copy(data=calibrated_ensemble_sorted)
    output_cube.data = np.where(np.isnan(raw_ensemble.data), np.nan, output_cube.data)
    return output_cube


@njit
def calibrate_by_patch(raw_ensemble, cal_ensemble, patch_width, offset_i, offset_j):
    dim_i, dim_j  = raw_ensemble.shape[1], raw_ensemble.shape[2]
    output = raw_ensemble.copy()
    for i in range(offset_i, dim_i, patch_width):
        for j in range(offset_j, dim_j, patch_width):
            min_i = max(0, (i - patch_width // 2))
            max_i = min(dim_i, i + patch_width // 2 + 1)
            min_j = max(0, (j - patch_width // 2))
            max_j = min(dim_j, j + patch_width // 2 + 1)
            patch_raw_flat = raw_ensemble[:,  min_i:max_i, min_j:max_j].flatten()
            patch_cal_flat = cal_ensemble[:, min_i:max_i, min_j:max_j].flatten()
            raw_sort_ind = np.argsort(patch_raw_flat, kind="mergesort")
            raw_reverse_sort_ind = np.argsort(raw_sort_ind, kind="mergesort")
            patch_ecc = np.take(np.sort(patch_cal_flat), raw_reverse_sort_ind)

            # put nans back in original locations
            original_non_nan_ind = np.nonzero(~np.isnan(patch_raw_flat))
            new_non_nan_ind = np.nonzero(~np.isnan(patch_ecc))
            patch_ecc[original_non_nan_ind] = patch_ecc[new_non_nan_ind]
            original_nan_ind = np.nonzero(np.isnan(patch_raw_flat))
            patch_ecc[original_nan_ind] = np.nan

            patch_ecc = np.reshape(patch_ecc, (raw_ensemble.shape[0],  max_i - min_i, max_j - min_j))
            output[:, min_i:max_i, min_j:max_j] = patch_ecc
    return output


def calculate_necc(raw_ensemble, calibrated_ensemble, patch_width, stride):

    output_data = np.zeros(raw_ensemble.shape)
    num_valid = np.zeros((1, ) + output_data.shape[1:])
    for offset_i in range(0, patch_width, stride):
        for offset_j in range(0, patch_width, stride):
            cal_data = calibrate_by_patch(raw_ensemble, calibrated_ensemble, patch_width, offset_i, offset_j)
            output_data += cal_data
            num_valid += ~np.any(np.isnan(cal_data), axis=0)[np.newaxis, :, :]
    output_data /= num_valid

    return output_data



def necc(raw_ensemble: iris.cube.Cube, calibrated_probabilities: iris.cube.Cube, patch_width, stride) -> iris.cube.Cube:
    """Apply Neigbhourhood-ECC to a single forecast.

    Args:
        raw_ensemble: cube with dimensions realization, x dim, y dim
        calibrated_probabilities: cube with dimensions threshold, x dim, y dim
    Returns:
        cube with same dimensions as raw_ensemble
    """

    # convert probabilties to realizations
    num_realizations = len(raw_ensemble.coord("realization").points)
    calibrated_ensemble = ConvertProbabilitiesToPercentiles().process(
            calibrated_probabilities, no_of_percentiles=num_realizations
        )

    output_data = calculate_necc(raw_ensemble.data, calibrated_ensemble.data, patch_width, stride)

    output_cube = raw_ensemble.copy(data=output_data.astype(np.float32))
    return output_cube


def process(input_path_raw, input_path_calibrated, output_path_ecc, output_path_smoothed_ecc, output_path_n_ecc, patch_width, stride):
    print(input_path_raw)
    ensemble_cube = iris.load_cube(input_path_raw)
    calibrated_cube = iris.load_cube(input_path_calibrated)

    # fill masked data with nans
    ensemble_cube.data = ensemble_cube.data.filled(np.nan)
    nan_ind = np.any(np.isnan(ensemble_cube.data), axis=0)[np.newaxis, :, :]
    calibrated_cube.data = np.where(nan_ind, np.nan, calibrated_cube.data)

    # calculate smoothed raw ensemble to define an ordering for points with the same values
    ensemble_smoothed = ensemble_cube.data.copy()
    ensemble_smoothed = np.nan_to_num(ensemble_smoothed)
    num_realizations = len(ensemble_cube.coord("realization").points)
    for i in range(num_realizations):
        conv_filter = np.ones((patch_width, patch_width))
        ensemble_smoothed[i, :, :] = convolve2d(ensemble_smoothed[i, :, :], conv_filter, mode="same")
        num_valid = convolve2d((~np.isnan(ensemble_cube.data[i, :, :])).astype(int), conv_filter, mode="same")
        ensemble_smoothed[i, :, :] = ensemble_smoothed[i, :, :] / num_valid
        ensemble_smoothed[i, :, :] = np.where(np.isnan(ensemble_cube.data[i, :, :]), np.nan, ensemble_smoothed[i, :, :])
    ensemble_cube_smoothed = ensemble_cube.copy(data=ensemble_smoothed)

    output = ecc(ensemble_cube, calibrated_cube)
    save_netcdf(output, output_path_ecc)
    output = ecc(ensemble_cube_smoothed, calibrated_cube)
    save_netcdf(output, output_path_smoothed_ecc)
    output =  necc(ensemble_cube_smoothed, calibrated_cube, patch_width, stride)
    save_netcdf(output, output_path_n_ecc)


if __name__ == "__main__":

    file_list = []

    output_dir_ecc = os.path.join(output_dir, "ecc")
    output_dir_smoothed_ecc = os.path.join(output_dir, "smoothed_ecc")
    output_dir_necc = os.path.join(output_dir, "necc")

    for dir in [output_dir_ecc, output_dir_smoothed_ecc, output_dir_necc]:
        if not(os.path.exists(dir)):
            os.makedirs(dir)

    for forecast_date in pd.date_range(start_date, end_date):
        valid_date = forecast_date + dt.timedelta(days=1)
        forecast_date_formatted = forecast_date.strftime("%Y%m%dT0000Z")
        valid_date_formatted = valid_date.strftime("%Y%m%dT0000Z")
        filename = f"{valid_date_formatted}-PT0024H00M-precipitation_accumulation-PT24H.nc"
        input_path_raw = os.path.join(input_dir_raw, forecast_date_formatted, filename)
        input_path_cal = os.path.join(input_dir_cal, filename)
        if not(os.path.exists(input_path_raw)) or not(os.path.exists(input_path_cal)):
            continue
        output_path_ecc = os.path.join(output_dir_ecc, filename)
        output_path_ordered_ecc = os.path.join(output_dir_smoothed_ecc, filename)
        output_path_necc = os.path.join(output_dir_necc, filename)
        file_list.append([input_path_raw, input_path_cal, output_path_ecc, output_path_ordered_ecc, output_path_necc, patch_width, stride])

    print(len(file_list))

    with Pool(70) as p:
        p.starmap(process, file_list)

