# This program is to make CloudSat into ERA5 grid 
# Import package
import os
import sys
import numpy as np
import datetime
from glob import glob

from pyhdf.HC  import HC
from pyhdf.SD  import SD, SDC
from pyhdf.HDF import HDF

from joblib import Parallel, delayed

from netCDF4 import Dataset

from matplotlib import pyplot as plt

from scipy.interpolate import interp1d

# function for examine the existence of path
def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File path '{filepath}' does not exist.")
        sys.exit(1)  # Shut down the script with a non-zero exit code

# load CloudSat data
def load_cs_data(fname):
    from pyhdf.VS  import VS

    hdf = HDF(fname, SDC.READ)
    file_sd = SD(fname, SDC.READ)
    
    lat     = np.array(hdf.vstart().attach("Latitude")[:]).squeeze()
    lon     = np.array(hdf.vstart().attach("Longitude")[:]).squeeze()
    hgt_all = np.array(file_sd.select("Height")[:])
    qr_all  = np.array(file_sd.select("QR")[:])

    file_sd.end()

    lon[lon <= 0] += 360

    qlw, qsw = qr_all[:2] / 100.
    # Filtered out invalid data
    qlw[qlw < -50] = np.nan
    qsw[qsw < -50] = np.nan
    
    valid_cols = ~np.all(np.isnan(qlw)&np.isnan(qsw), axis=0)

    return {
        "lon": lon, "lat": lat,
        "hgt": hgt_all[:, valid_cols],
        "qlw": qlw[:, valid_cols],
        "qsw": qsw[:, valid_cols]
    }

# Interpolate the CLoudSat data to a pressure coordinate
def process_file(f, lat_era5, lon_era5, z_data):
    """Process individual CloudSat file and match to ERA5 grid."""
    cs_lat, cs_lon, cs_hgt, cs_qlw, cs_qsw = f["lat"], f["lon"], f["hgt"], f["qlw"], f["qsw"]

    # Find nearest ERA5 grid index
    lat_idx = np.searchsorted(lat_era5, cs_lat, side="right") - 1
    lon_idx = np.searchsorted(lon_era5, cs_lon, side="right") - 1

    # Pair up indices and find unique ones
    coord = np.column_stack((lat_idx, lon_idx))
    unique_pairs, inverse_indices = np.unique(coord, axis=0, return_inverse=True)

    # Store indices where each unique (lat_idx, lon_idx) occurs
    pair_indices = {
        tuple(pair): np.where(inverse_indices == i)[0].tolist()
        for i, pair in enumerate(unique_pairs)}

    # Store results
    qlw_interp = np.empty((8, len(lat_era5), len(lon_era5)), dtype=np.float32)
    qsw_interp = np.empty((8, len(lat_era5), len(lon_era5)), dtype=np.float32)
    qlw_interp.fill(np.nan)
    qsw_interp.fill(np.nan)
    num_interp = np.zeros((len(lat_era5), len(lon_era5)), dtype=int)

    for (lat_id, lon_id), indices in pair_indices.items():
        if len(indices) > 1:
            # Get mean ERA5 data for the matched points
            era5_values = z_data[:, lat_id, lon_id]  # Extract ERA5 at this grid point
            itp_qlw, itp_qsw = np.zeros((8), dtype=np.float32), np.zeros((8), dtype=np.float32)

            for idx in indices:
                try:
                    cs_qlw_valid = ~np.isnan(cs_qlw[idx])
                    cs_qsw_valid = ~np.isnan(cs_qsw[idx])

                    if not np.any(cs_qlw_valid) or not np.any(cs_qsw_valid):
                        continue  # Skip if no valid values

                    itp_qlw += interp1d(cs_hgt[idx][cs_qlw_valid], cs_qlw[idx][cs_qlw_valid], kind="linear", fill_value="extrapolate")(era5_values)
                    itp_qsw += interp1d(cs_hgt[idx][cs_qsw_valid], cs_qsw[idx][cs_qsw_valid], kind="linear", fill_value="extrapolate")(era5_values)
                
                except ValueError:
                    continue  # Skip errors in interpolation

            itp_qlw /= len(indices)
            itp_qsw /= len(indices)
            qlw_interp[:, lat_id, lon_id] = itp_qlw
            qsw_interp[:, lat_id, lon_id] = itp_qsw
            num_interp[lat_id, lon_id] = len(indices)

    return qlw_interp, qsw_interp

# main function
def main():
    year, date = np.array(sys.argv[1:3], dtype=int)
    #year, date = np.array([2006, 163], dtype=int)
    
    # Load ERA5 data
    with Dataset(f"/work/b11209013/2024_Research/nstcCCKW/z/z_{year}.nc", "r") as f:
        dims = {key: f.variables[key][:] for key in f.dimensions.keys() if key in f.variables.keys()}

        z = np.array(f.variables["z"][int(date-1)].squeeze()) / 9.81

    dims["lev"] = np.array([1000, 925, 850, 700, 500, 250, 200, 100])

    # Load CloudSat data
    ## file path
    path = f"/work/DATA/Satellite/CloudSat/{year}/{date:03d}/"

    if not os.path.exists(path):
        with Dataset(f"/work/b11209013/2024_Research/CloudSat/CloudSat_Interp/CloudSat_Interpolated_{year:04d}_{date:03d}.nc", "w") as f:
            # Add global attributes that CDO uses for metadata
            f.title = f"CloudSat CDO Data for {year}-{date}"
            f.institution = "NTUAS"
            f.source = "CloudSat"
            f.history = f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f.conventions = "CF-1.8"  # Climate and Forecast conventions
            f.createDimension("time", None)
            f.createDimension("lev", len(dims["lev"]))
            f.createDimension("lat", len(dims["lat"]))
            f.createDimension("lon", len(dims["lon"]))
            # Create coordinate variables with attributes
            time_var = f.createVariable("time", np.float64, ("time",))
            time_var.units = "hours since 1900-01-01 00:00:00"  # Standard time units
            time_var.calendar = "standard"
            time_var.standard_name = "time"
            time_var.long_name = "time"
            time_var.axis = "T"
            lev_var = f.createVariable("lev", np.float32, ("lev",))
            lev_var.units = "hPa"  # Assuming pressure levels
            lev_var.standard_name = "air_pressure"
            lev_var.long_name = "pressure level"
            lev_var.axis = "Z"
            lev_var.positive = "down"  # For pressure levels
            lat_var = f.createVariable("lat", np.float32, ("lat",))
            lat_var.units = "degrees_north"
            lat_var.standard_name = "latitude"
            lat_var.long_name = "latitude"
            lat_var.axis = "Y"
            lon_var = f.createVariable("lon", np.float32, ("lon",))
            lon_var.units = "degrees_east"
            lon_var.standard_name = "longitude"
            lon_var.long_name = "longitude"
            lon_var.axis = "X"
            # Assign coordinate values
            f.variables["time"][:] = dims["time"][int(date-1)]
            f.variables["lev"][:] = dims["lev"]
            f.variables["lat"][:] = dims["lat"]
            f.variables["lon"][:] = dims["lon"]
            # Create data variables with attributes
            qlw = f.createVariable("qlw", np.float32, ("time", "lev", "lat", "lon"), 
                                zlib=True, complevel=4)  # Compression for efficiency
            qlw.units = "K/day"
            qlw.standard_name = "tendency_of_air_temperature_due_to_longwave_heating"
            qlw.long_name = "Longwave Heating Rate"
            qlw[:, :, :, :] = np.full_like(z[None, :], np.nan, dtype=np.float32)  # More concise
            
            qsw = f.createVariable("qsw", np.float32, ("time", "lev", "lat", "lon"),
                                zlib=True, complevel=4)
            qsw.units = "K/day"
            qsw.standard_name = "tendency_of_air_temperature_due_to_shortwave_heating"
            qsw.long_name = "Shortwave Heating Rate"
            qsw[:, :, :, :] = np.full_like(z[None, :], np.nan, dtype=np.float32)  # More concise

        print(f"{year}_{date:03d} is empty")
        sys.exit(1)

    file_list = glob(path + "*.hdf")

    ## Load CloudSat data
    read_file = Parallel(n_jobs=-1, backend="loky")(
        delayed(load_cs_data)(fname)
        for fname in file_list
    )

    # Merge all the array into one
    concat_dict = {
        "lon": [read_file[0]["lon"]], "lat": [read_file[0]["lat"]],
        "hgt": [read_file[0]["hgt"]], "qlw": [read_file[0]["qlw"]],
        "qsw": [read_file[0]["qsw"]]
    }

    for key in concat_dict.keys():
        for i in range(1, len(read_file)):
            if read_file[i]["hgt"].shape[1] == concat_dict["hgt"][0].shape[1]:
                concat_dict[key].append(read_file[i][key])
            else:
                concat_dict[key].append(read_file[i][key][:, concat_dict["hgt"][0].shape[1]])

    concat_dict = {
        key: np.concatenate(concat_dict[key], axis=0)
        for key in concat_dict
    }

    itp_cs = process_file(concat_dict, dims["lat"], dims["lon"], z)

    print(itp_cs[0].shape)
    print(f"{year:04d}-{date:03d} done")

    with Dataset(f"/work/b11209013/2024_Research/CloudSat/CloudSat_Interp/CloudSat_Interpolated_{year:04d}_{date:03d}.nc", "w") as f:
        # Add global attributes that CDO uses for metadata
        f.title = f"CloudSat CDO Data for {year}-{date}"
        f.institution = "Your Institution"
        f.source = "CloudSat"
        f.history = f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f.conventions = "CF-1.8"  # Climate and Forecast conventions
        
        f.createDimension("time", None)
        f.createDimension("lev", len(dims["lev"]))
        f.createDimension("lat", len(dims["lat"]))
        f.createDimension("lon", len(dims["lon"]))

        # Create coordinate variables with attributes
        time_var = f.createVariable("time", np.float64, ("time",))
        time_var.units = "hours since 1900-01-01 00:00:00"  # Standard time units
        time_var.calendar = "standard"
        time_var.standard_name = "time"
        time_var.long_name = "time"
        time_var.axis = "T"

        lev_var = f.createVariable("lev", np.float32, ("lev",))
        lev_var.units = "hPa"  # Assuming pressure levels
        lev_var.standard_name = "air_pressure"
        lev_var.long_name = "pressure level"
        lev_var.axis = "Z"
        lev_var.positive = "down"  # For pressure levels

        lat_var = f.createVariable("lat", np.float32, ("lat",))
        lat_var.units = "degrees_north"
        lat_var.standard_name = "latitude"
        lat_var.long_name = "latitude"
        lat_var.axis = "Y"
        
        lon_var = f.createVariable("lon", np.float32, ("lon",))
        lon_var.units = "degrees_east"
        lon_var.standard_name = "longitude"
        lon_var.long_name = "longitude"
        lon_var.axis = "X"
        
        # Assign coordinate values
        f.variables["time"][:] = dims["time"][int(date-1)]
        f.variables["lev"][:] = dims["lev"]
        f.variables["lat"][:] = dims["lat"]
        f.variables["lon"][:] = dims["lon"]

        # Create data variables with attributes
        qlw = f.createVariable("qlw", np.float32, ("time", "lev", "lat", "lon"), 
                            zlib=True, complevel=4)  # Compression for efficiency
        qlw.units = "K/day"
        qlw.standard_name = "tendency_of_air_temperature_due_to_longwave_heating"
        qlw.long_name = "Longwave Heating Rate"
        qlw[:, :, :, :] = itp_cs[0][None, :, :, :]

        qsw = f.createVariable("qsw", np.float32, ("time", "lev", "lat", "lon"),
                            zlib=True, complevel=4)
        qsw.units = "K/day"
        qsw.standard_name = "tendency_of_air_temperature_due_to_shortwave_heating"
        qsw.long_name = "Shortwave Heating Rate"
        qsw[:, :, :, :] = itp_cs[1][None, :, :, :]


if __name__ == "__main__":
    main()

