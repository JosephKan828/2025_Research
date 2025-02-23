# This program is to interpolate the CloudSat data to a pressure coordinate
# import packages
import os 
import sys
import glob
import numpy as np
import xarray as xr

from pyhdf.HC  import HC
from pyhdf.SD  import SD, SDC
from pyhdf.VS  import VS
from pyhdf.HDF import HDF
import joblib
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
#year = int(sys.argv[1])
#date = int(sys.argv[2])

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File path '{filepath}' does not exist.")
        sys.exit(1)  # Shut down the script with a non-zero exit code

def load_cs_data(fname):
    # open data
    from pyhdf.VS  import VS
    hdf = HDF(fname, SDC.READ)
    file_sd = SD(fname, SDC.READ)
    
    lat = np.array(hdf.vstart().attach("Latitude")[:]).squeeze()
    lon = np.array(hdf.vstart().attach("Longitude")[:]).squeeze()
    hgt_all = np.array(file_sd.select("Height")[:])
    qr_all = np.array(file_sd.select("QR")[:])

    file_sd.end()

    lon[lon <= 0] += 360
    
    valid_mask = (lat >= -15) & (lat <= 15) & (lon >= 160) & (lon <= 260)

    if not np.any(valid_mask):
        return None  # Skip empty files
    
    lat, lon = lat[valid_mask], lon[valid_mask]
    
    hgt = hgt_all[valid_mask, :]
    qlw, qsw = (qr_all[0, valid_mask] / 100.), (qr_all[1, valid_mask] / 100.)
    
    # Filtered out invalid data
    qlw[qlw < -50] = np.nan
    qsw[qsw < -50] = np.nan
    
    valid_cols = ~np.all(np.isnan(qlw)&np.isnan(qsw), axis=0)

    return {
        "lon": lon, "lat": lat,
        "hgt": hgt[:, valid_cols],
        "qlw": qlw[:, valid_cols],
        "qsw": qsw[:, valid_cols]
    }

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

    return qlw_interp, qsw_interp, num_interp

def main():
    year, date = int(sys.argv[1]), int(sys.argv[2])
    #year, date = 2006, 163

    path = f"/work/DATA/Satellite/CloudSat/{year}/{date:03d}/"

    check_file_exists(path)
    
    # Load CloudSat data
    file_list = glob.glob(path + "*.hdf")
    
    limit_data = Parallel(n_jobs = -1, backend="loky")(
        delayed(load_cs_data)(fname)
        for fname in file_list
    )

    new_data = [f for f in limit_data if f]

    # Load ERA5 data
    with xr.open_dataset(f"/work/b11209013/2024_Research/nstcCCKW/z/z_{year}.nc") as f:
        dims = f.coords
        z_data = (f.z.sel(time=dims["time"][date-1]) / 9.81).values  # Convert to geopotential height

    lat_era5, lon_era5 = np.array(dims["lat"]) - 0.25, np.array(dims["lon"]) - 0.3125

    # Process each CloudSat file in parallel
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_file)(f, lat_era5, lon_era5, z_data)
        for f in new_data
    )
    

    qlw_sync = np.nanmean(np.array([r[0] for r in results]), axis=0)
    qsw_sync = np.nanmean(np.array([r[1] for r in results]), axis=0)

    # Save processed data
    ds = xr.Dataset(
        {
            "qlw_sync": (["pressure_level", "latitude", "longitude"], qlw_sync),
            "qsw_sync": (["pressure_level", "latitude", "longitude"], qsw_sync),
        },
        coords={
            "pressure_level": np.array([1000, 925, 850, 700, 500, 250, 200, 100]),  # Assuming 8 levels; modify if needed
            "latitude": lat_era5,
            "longitude": lon_era5,
        },
        attrs={
            "title": "Interpolated CloudSat Data on ERA5 Pressure Coordinates",
            "source": "CloudSat & ERA5",
            "description": "Longwave (qlw) and Shortwave (qsw) interpolated fluxes.",
            "date_processed": f"{year}-{date:03d}"
        }
    )

    # Save to NetCDF
    output_filename = f"/work/b11209013/2024_Research/CloudSat/CS_sel/CloudSat_Interpolated_{year}_{date:03d}.nc"
    ds.to_netcdf(output_filename, format="NETCDF4", mode="w")
    
    print(f"✅ aved {output_filename}")


if __name__ == "__main__":
    main()
