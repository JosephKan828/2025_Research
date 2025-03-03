# This program is to merge CloudSat data
# Import packages
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from datetime import datetime

# Compute time difference
def time_itv_vectorized(time_array):
    base_time = np.datetime64("1900-01-01T00:00:00")
    return (time_array.astype("datetime64[s]") - base_time).astype("timedelta64[h]").astype(np.float32)

def process_CS_for_CDO(input, output):
    with xr.open_dataset(input, engine="netcdf4") as f:
        lat = f.latitude.values.astype(np.float32)
        lon = f.longitude.values.astype(np.float32)
        lev = f.pressure_level.values.astype(np.float32)
        time = f.time.values  # Keep raw format for fast conversion
        qlw = f.qlw.load().astype(np.float32)
        qsw = f.qsw.load().astype(np.float32)
        
    # Convert time
    time_itv_sec = time_itv_vectorized(time)
    
    # Create output file
    with nc.Dataset(output, "w", format="NETCDF4") as f:
        # Define dimensions
        f.createDimension("time", len(time_itv_sec))
        f.createDimension("lev", len(lev))
        f.createDimension("lat", len(lat))
        f.createDimension("lon", len(lon))

        # Create variables with compression
        f.createVariable("time", "f4", ("time",), zlib=True)
        f.createVariable("lev", "f4", ("lev",), zlib=True)
        f.createVariable("lat", "f4", ("lat",), zlib=True)
        f.createVariable("lon", "f4", ("lon",), zlib=True)
        f.createVariable("qlw", "f4", ("time", "lev", "lat", "lon"), zlib=True, chunksizes=(1, len(lev), len(lat), len(lon)))
        f.createVariable("qsw", "f4", ("time", "lev", "lat", "lon"), zlib=True, chunksizes=(1, len(lev), len(lat), len(lon)))

        # Assign values
        f.variables["time"][:] = time_itv_sec
        f.variables["lev"][:] = lev
        f.variables["lat"][:] = lat
        f.variables["lon"][:] = lon
        f.variables["qlw"][:] = qlw
        f.variables["qsw"][:] = qsw

        # Add attributes for CDO compatibility
        f.variables["time"].units = "hours since 1900-01-01 00:00:00"
        f.variables["time"].calendar = "gregorian"
        f.variables["time"].long_name = "time"

        f.variables["lev"].long_name = "pressure_level"
        f.variables["lev"].units = "hPa"

        f.variables["lat"].long_name = "latitude"
        f.variables["lat"].units = "degrees_north"

        f.variables["lon"].long_name = "longitude"
        f.variables["lon"].units = "degrees_east"

        f.variables["qlw"].long_name = "Longwave heating rate"
        f.variables["qlw"].units = "K/day"

        f.variables["qsw"].long_name = "Shortwave heating rate"
        f.variables["qsw"].units = "K/day"

    print(f"Processed file saved: {output}")
    
def main():
    file_list = glob.glob("/work/b11209013/2024_Research/CloudSat/interpolated_output_*.nc")
    
    for f in file_list:
        output_file = f.replace("interpolated_output_", "CloudSat_CDO_")

        process_CS_for_CDO(f, output_file)
if __name__ == "__main__":
    
    main()