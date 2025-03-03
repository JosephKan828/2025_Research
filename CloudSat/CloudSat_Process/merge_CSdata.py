import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
from re import search

def extract_time(fname):
    """
    Extracts time from filename formatted as YYYY_DDD.
    """
    match = search(r"(\d{4})_(\d{3})", fname)
    
    if match:
        year, doy = match.groups()
        return pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=int(doy) - 1)
    else:
        raise ValueError(f"Could not extract time from {fname}")

def main():
    year = sys.argv[1]  # Input year

    # File path
    path = f"/work/b11209013/2024_Research/CloudSat/CloudSat_Interp/CloudSat_Interpolated_{year}_*.nc"
    print(f"Processing files in: {path}")

    file_list = sorted(glob.glob(path))
    if not file_list:
        print("No files found. Check file path and naming convention.")
        sys.exit(1)

    extracted_times = []
    dataset = []

    # Read NetCDF files lazily to reduce memory usage
    for f in file_list:
        with xr.open_dataset(f, engine="netcdf4") as ds:
            time = extract_time(f)
            ds = ds.expand_dims("time")
            ds = ds.assign_coords(time=[time])
            dataset.append(ds)
            extracted_times.append(time)

    # Concatenate along time dimension
    data = xr.concat(dataset, dim="time", combine_attrs="override")
    del dataset  # Free memory

    # Generate full daily time range (insert NaN for missing dates)
    full_time_range = pd.date_range(start=min(extracted_times), end=max(extracted_times), freq="D")
    data = data.reindex(time=full_time_range)

    # Convert to NumPy before interpolation to prevent Dask chunk explosion
    data = data.compute()

    # Define standard pressure levels (1000 hPa to 100 hPa, 37 levels)
    new_levels = np.linspace(1000, 100, 37)

    # Ensure `level` is recognized as pressure in hPa
    interpolated_data = xr.Dataset()

    if "qlw_sync" in data:
        interpolated_data["qlw"] = data["qlw_sync"].interp(pressure_level=new_levels, method="linear").astype(np.float32)
    if "qsw_sync" in data:
        interpolated_data["qsw"] = data["qsw_sync"].interp(pressure_level=new_levels, method="linear").astype(np.float32)

    # Rename coordinates if necessary
    if "latitude" in data.coords:
        data = data.rename({"latitude": "lat"})
    if "longitude" in data.coords:
        data = data.rename({"longitude": "lon"})

    # Copy coordinates and preserve metadata
    interpolated_data = interpolated_data.assign_coords({
        "time": data.time,
        "level": new_levels,
        "lat": data.lat,
        "lon": data.lon
    })

    # Ensure time units are CDO-compatible (CF conventions)
    #interpolated_data["time"].attrs = {
        #"units": f"days since {year}-01-01 00:00:00",
        #"calendar": "standard",
        #"long_name": "Time"
    #}

    # Assign metadata for CDO compatibility
    interpolated_data["level"].attrs = {
        "units": "hPa",
        "positive": "down",
        "long_name": "Pressure level"
    }

    interpolated_data["lat"].attrs = {
        "units": "degrees_north",
        "long_name": "Latitude"
    }

    interpolated_data["lon"].attrs = {
        "units": "degrees_east",
        "long_name": "Longitude"
    }

    if "qlw" in interpolated_data:
        interpolated_data["qlw"].attrs = {
            "long_name": "Longwave heating rate",
            "units": "K/day",
            "description": "Interpolated longwave heating rate over standard pressure levels (1000 to 100 hPa)"
        }
    
    if "qsw" in interpolated_data:
        interpolated_data["qsw"].attrs = {
            "long_name": "Shortwave heating rate",
            "units": "K/day",
            "description": "Interpolated shortwave heating rate over standard pressure levels (1000 to 100 hPa)"
        }

    # Assign global attributes for CF compliance
    interpolated_data.attrs = {
        "title": "Interpolated CloudSat Dataset",
        "description": "Merged CloudSat data with NaNs for missing dates and interpolated over levels.",
        "source": "CloudSat observations, processed with xarray",
        "processed_by": "Yu-Chuan Kan / Lab. of Chaos and Predictability NTUAS",
        "Conventions": "CF-1.6",
        "institution": "NTUAS",
        "history": f"Processed on {pd.Timestamp.now()}",
    }

    # Save dataset using chunking to optimize memory
    output_file = f"/work/b11209013/2024_Research/CloudSat/interpolated_output_{year}.nc"
    interpolated_data.to_netcdf(
        output_file, 
        format="NETCDF4_CLASSIC",  # Ensures CDO compatibility
        encoding={
            "qlw": {"zlib": True, "complevel": 4, "dtype": "float32"},
            "qsw": {"zlib": True, "complevel": 4, "dtype": "float32"},
            "time": {"dtype": "int32"},
            "level": {"dtype": "float32"},
            "lat": {"dtype": "float32"},
            "lon": {"dtype": "float32"}
        },
        compute=True
    )

    print(f"Interpolated dataset saved to: {output_file}")

if __name__ == "__main__":
    main()
