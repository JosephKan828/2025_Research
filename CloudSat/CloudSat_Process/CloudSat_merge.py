# This program is to merge CloudSat data into a single NetCDF file
# import package
import xarray as xr
import numpy as np
import pandas as pd
import os
import re

def main():
    path = "/work/b11209013/2024_Research/CloudSat/CloudSat_Interp/"

    name_pat = r"CloudSat_Interpolated_(\d{4})_(\d{3})\.nc"

    files = [f for f in os.listdir(path) if re.match(name_pat, f)]

    # Extract dates from filenames
    dates_available = []
    datasets = {}

    for file in files:
        match = re.match(name_pat, file)
        if match:
            year = int(match.group(1))
            day_of_year = int(match.group(2))
            date = pd.to_datetime(f"{year}-{day_of_year}", format="%Y-%j")  # Convert to datetime
            dates_available.append(date)

            try:
                datasets[date] = xr.open_dataset(os.path.join(path, file), engine="netcdf4")
            except OSError:
                continue

    # Generate a complete date range
    date_range = pd.date_range(start=min(dates_available), end=max(dates_available), freq="D")

    # Merge datasets, inserting NaN where data is missing
    merged_datasets = []
    for date in date_range:
        if date in datasets:
            merged_datasets.append(datasets[date])
        else:
            # Create an empty dataset with NaN values, matching existing dimensions
            empty_ds = datasets[next(iter(datasets))].copy(deep=True)  # Copy structure
            for var in empty_ds.data_vars:
                empty_ds[var].values[:] = np.nan  # Set all values to NaN
            merged_datasets.append(empty_ds)

    # Concatenate along time dimension
    merged_ds = xr.concat(merged_datasets, dim="time")

    # Save the merged dataset to a new NetCDF file
    merged_ds.to_netcdf("/work/b11209013/2024_Research/CloudSat/Merged_CloudSat_Interpolated.nc")

    print("Merging completed. Missing dates filled with NaN.")

    with xr.open_dataset("/work/b11209013/2024_Research/CloudSat/Merged_CloudSat_Interpolated.nc", engine="netcdf4") as f:
        print(f"Dimensions: {f.dims}")
        print(f"Variables: {f.variables.keys()}")
        print(date_range.shape)
        print(f.time.shape)

if __name__ == "__main__":
    main()
