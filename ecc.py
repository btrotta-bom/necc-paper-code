"""Implements variants of ECC, including that described in Section 3f of 
Scheuerer and Hamill, Generating calibrated ensembles of physically realistic, high-resolution
precipitation forecast fields based on GEFS model output. Journal of Hydrometeorology, 2018."""

import iris
import numpy as np
import os
from multiprocessing import Pool
from numba import njit
import pandas as pd
import datetime as dt
from pathlib import Path

from improver.ensemble_copula_coupling.ensemble_copula_coupling import ConvertProbabilitiesToPercentiles
from improver.utilities.save import save_netcdf
from scipy.signal import convolve2d

models = ["ecmwf", "accessge3"]
input_dir_raw = f"/path/to/nwp/forecasts/"  # parameters {model}
input_dir_cal = f"/path/to/calibrated/probability/forecasts/"  # parameters {model}
output_dir = f"/path/to/output/"  # parameters {model}, {method}, {width_str}, {lambda_str}
start_date = dt.datetime(2021, 9, 1)
end_date = dt.datetime(2022, 9, 1)
stride = 1


def ecc(raw_ensemble: iris.cube.Cube, calibrated_probabilities: iris.cube.Cube, ensemble_order: iris.cube.Cube = None, regularize: bool = False, reg_lambda: float = None) -> iris.cube.Cube:
    """Apply ECC to a single forecast.

    Args:
        raw_ensemble: cube with dimensions realization, x dim, y dim
        calibrated_probabilities: cube with dimensions threshold, x dim, y dim
        ensemble_order: cube with dimensions realization, x dim, y dim, used to re-order the calibrated ensemble.
            If None, use raw_ensemble
        regularize: if True, apply the technique of Scheuerur and Hamill to fit a regularized 
            piecewise-linear regression mapping the raw ensemble values to the calibrated probabilities
        reg_lambda: lambda parameter for regularization, only required if regularize is True
    Returns:
        cube with same dimensions as raw_ensemble
    """

    if ensemble_order is None:
        ensemble_order = raw_ensemble

    # convert probabilties to realizations
    num_realizations = len(raw_ensemble.coord("realization").points)
    calibrated_ensemble = ConvertProbabilitiesToPercentiles().process(
            calibrated_probabilities, no_of_percentiles=num_realizations
        )

    # apply ecc
    ensemble_sort_ind = np.argsort(ensemble_order.data, axis=0, kind="stable")
    raw_ensemble_reverse_sort_ind = np.argsort(ensemble_sort_ind, axis=0, kind="stable")
    if regularize:
        sorted_raw_ensemble = np.take_along_axis(raw_ensemble.data, ensemble_sort_ind, axis=0)
        smoothed_data = regularized_fit(sorted_raw_ensemble, calibrated_ensemble.data, reg_lambda)
        calibrated_ensemble_sorted = np.take_along_axis(smoothed_data, raw_ensemble_reverse_sort_ind, axis=0)
    else:
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
def regularized_fit(raw_ensemble, calibrated_ensemble, reg_lambda):
    """Apply the technique of Scheuerer and Hamill to fit a regularized piecewise-linear regression 
    mapping the raw ensemble values to the calibrated ensemble.
    
    Args:
        raw_ensemble: 3-d numpy array with dimensions realization, x dim, y dim, 
            non-decreasing along first dimension
        calibrated_ensemble: 3-d numpy array with dimensions realization, x dim, y dim, 
            non-decreasing along first dimension
        reg_lambda: regularization parameter
    Returns:
        cube with same dimensions as raw_ensemble
    """

    num_realizations, dim_x, dim_y = raw_ensemble.shape
    output = calibrated_ensemble.copy()
    for x in range(dim_x):
        for y in range(dim_y):
            raw_values = raw_ensemble[:, x, y] * 1000  # convert to mm
            if np.any(np.isnan(raw_values)):
                output[:, x, y] = np.nan
                continue
            cal_values = calibrated_ensemble[:, x, y] * 1000
            if cal_values[-2] == 0:
                # there is at most one non-zero value; do not modify the calibrated ensemble
                continue
            else:
                K = num_realizations - 1   # last index
                n_0 = cal_values.nonzero()[0][0] - 1  # index of last zero realization
                # in the paper, realizations are indexed 1, 2, ..., num_realizations, 
                # whereas ours are indexed 0, 1, ... num_realizations - 1
                if n_0 == -1:
                    n_0 = 0
                num_sq_errors = K - n_0 + 1  # number of terms in first sum of Eqn(5)
                num_reg_coeffs = K - n_0 - 1  # number of coeffs subject to regularization
                a = np.zeros((num_sq_errors + num_reg_coeffs, num_reg_coeffs + 2), np.float32)
                # we solve the regularized least squares problem by extending the matrix a used in 
                # unregularized least squares so that the last num_coeffs rows of a correspond to the 
                # regularization term; see 
                # https://inst.eecs.berkeley.edu/~ee127/sp21/livebook/l_ols_rls_def.html
                a[:num_sq_errors, 0] = 1
                a[:num_sq_errors, 1] = raw_values[n_0:]
                for k in range(n_0, K + 1):
                    for j in range(n_0 + 1, k):
                        a[k - n_0, j - n_0 + 1] = raw_values[k] - raw_values[j]
                for coeff_idx, k in enumerate(range(n_0 + 1, K)):
                    a[coeff_idx + num_sq_errors, coeff_idx + 2] = np.sqrt(reg_lambda * (np.maximum(1, cal_values[k])))
                b = np.concatenate((cal_values[n_0:], np.zeros(num_reg_coeffs, np.float32)))
                res = np.linalg.lstsq(a, b)
                coeffs = res[0]
                adj_values = np.concatenate((np.full(n_0, coeffs[0]), np.dot(a[:num_sq_errors, :], coeffs).flatten()))
                adj_values = np.maximum(0, adj_values)
                output[:, x, y] = adj_values * 0.001  # convert to m
    return output


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


def calculate_smoothed_ensemble(ensemble_cube, conv_filter):
    ensemble_smoothed = ensemble_cube.data.copy()
    ensemble_smoothed = np.nan_to_num(ensemble_smoothed)
    num_realizations = len(ensemble_cube.coord("realization").points)
    for i in range(num_realizations):
        ensemble_smoothed[i, :, :] = convolve2d(ensemble_smoothed[i, :, :], conv_filter, mode="same")
        num_valid = convolve2d((~np.isnan(ensemble_cube.data[i, :, :])).astype(int), conv_filter, mode="same")
        ensemble_smoothed[i, :, :] = ensemble_smoothed[i, :, :] / num_valid
        ensemble_smoothed[i, :, :] = np.where(np.isnan(ensemble_cube.data[i, :, :]), np.nan, ensemble_smoothed[i, :, :])
    ensemble_cube_smoothed = ensemble_cube.copy(data=ensemble_smoothed)
    return ensemble_cube_smoothed


def process(input_path_raw, input_path_calibrated, output_path_dict, patch_width, reg_lambda, stride):
    print(input_path_raw)
    ensemble_cube = iris.load_cube(str(input_path_raw))
    calibrated_cube = iris.load_cube(str(input_path_calibrated))

    # fill masked data with nans
    ensemble_cube.data = ensemble_cube.data.filled(np.nan)
    nan_ind = np.any(np.isnan(ensemble_cube.data), axis=0)[np.newaxis, :, :]
    calibrated_cube.data = np.where(nan_ind, np.nan, calibrated_cube.data)

    if "smoothed_ecc" in output_path_dict or "necc" in output_path_dict: 
        # calculate smoothed raw ensemble to define an ordering for points with the same values
        conv_filter = np.ones((patch_width, patch_width))
        ensemble_cube_smoothed_uniform = calculate_smoothed_ensemble(ensemble_cube, conv_filter)

    if "tricube_ecc" in output_path_dict or "tricube_ecc_reg" in output_path_dict:
        # use tricube kernel to replace zeros with smooth field of negative values, using technique from Scheuerer and Hamill
        tricube_kernel = np.empty((patch_width, patch_width))
        for i in range(0, patch_width):
            for j in range(0, patch_width):
                tricube_kernel[i, j] = np.power(1 - np.power(np.abs(i - patch_width // 2) / (patch_width // 2), 3), 3) *  np.power(1 - np.power(np.abs(j - patch_width // 2) / (patch_width // 2), 3), 3)
        uniform_sample = ensemble_cube.copy(data=np.random.uniform(low=-1, high=0, size=ensemble_cube.data.shape))
        uniform_sample.data = np.where(ensemble_cube.data == 0, 0, uniform_sample.data)
        uniform_sample_smoothed = calculate_smoothed_ensemble(uniform_sample, tricube_kernel)
        ensemble_cube_smoothed_tricube = ensemble_cube.copy()
        ensemble_cube_smoothed_tricube.data = np.where(ensemble_cube_smoothed_tricube.data == 0, uniform_sample_smoothed.data, ensemble_cube.data)

    # calculate ecc
    if "ecc" in output_path_dict:
        output = ecc(ensemble_cube, calibrated_cube, None, False)
        save_netcdf(output, str(output_path_dict["ecc"]))
    if "smoothed_ecc" in output_path_dict:
        output = ecc(ensemble_cube, calibrated_cube, ensemble_cube_smoothed_uniform, False)
        save_netcdf(output, str(output_path_dict["smoothed_ecc"]))
    if "tricube_ecc" in output_path_dict:
        output = ecc(ensemble_cube, calibrated_cube, ensemble_cube_smoothed_tricube, False)
        save_netcdf(output, str(output_path_dict["tricube_ecc"]))
    if "tricube_ecc_reg" in output_path_dict:
        output = ecc(ensemble_cube, calibrated_cube, ensemble_cube_smoothed_tricube, True, reg_lambda=reg_lambda)
        save_netcdf(output, str(output_path_dict["tricube_ecc_reg"]))
    if "necc" in output_path_dict:
        output = necc(ensemble_cube_smoothed_uniform, calibrated_cube, patch_width, stride)
        save_netcdf(output, str(output_path_dict["necc"]))


if __name__ == "__main__":

    file_list = []
    methods = ["ecc", "smoothed_ecc", "necc", "tricube_ecc", "tricube_ecc_reg"]
    patch_widths = [5, 9, 19]
    reg_lambdas = [0.1, 0.5, 1, 2, 5, 10, 50]

    # maintain a list of output paths to avoid writing the same file twice
    output_paths = set()
    for forecast_date in pd.date_range(start_date, end_date):
        valid_date = forecast_date + dt.timedelta(days=1)
        forecast_date_formatted = forecast_date.strftime("%Y%m%dT0000Z")
        valid_date_formatted = valid_date.strftime("%Y%m%dT0000Z")
        filename = f"{valid_date_formatted}-PT0024H00M-precipitation_accumulation-PT24H.nc"
        for model in models:
            for reg_lambda in reg_lambdas:
                for patch_width in patch_widths:
                    output_path_dict = {}
                    for method in methods:
                        if method == "ecc":
                            width_str = ""
                        else:
                            width_str = f"width_{patch_width}"
                        if method == "tricube_ecc_reg":
                            if patch_width != 9:
                                # only do width 9
                                continue
                            lambda_str = f"lambda_{reg_lambda:0.1f}"
                        else:
                            lambda_str = ""
                        curr_output_dir = output_dir.format(model=model, width_str=width_str, lambda_str=lambda_str, method=method)
                        if not(os.path.exists(curr_output_dir)):
                            os.makedirs(curr_output_dir)
                        input_path_raw = Path(input_dir_raw.format(model=model)) / forecast_date_formatted / filename
                        input_path_cal = Path(input_dir_cal.format(model=model)) / filename
                        output_path = Path(curr_output_dir) / filename
                        if not(input_path_raw.exists()) or not(input_path_cal.exists()) or output_path in output_paths:
                            continue
                        output_paths.add(output_path)
                        output_path_dict[method] = Path(curr_output_dir) / filename
                        file_list.append([input_path_raw, input_path_cal, output_path_dict, patch_width, reg_lambda, stride])

    print(len(file_list))

    with Pool(70) as p:
        p.starmap(process, file_list)

