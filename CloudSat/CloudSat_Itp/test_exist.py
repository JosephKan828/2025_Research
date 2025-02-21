# This program is to interpolate the selected CloudSat data with geopotential
# import package
import os
import sys
import numpy as np
import joblib as jl
import xarray as xr
import pandas as pd

def main():

    # Load selected events
    with xr.open_dataset("/home/b11209013/2025_Research/CloudSat/CloudSat_Itp/CCKW_select.nc") as ds:
        time_label = ds.time.values

    print(len(time_label))
    year_list, month_list, day_list = [], [], []

    for t in time_label:
        year, month, day = t.astype(str).split("-")

        year_list.append(int(year))
        month_list.append(int(month))
        day_list.append(int(day[:2]))

    # Load CloudSat
    cs_data = []
    date_idx = []
    output_year = []
    output_date = []

    for i in range(len(year_list)):
        date = pd.date_range(f"{year_list[i]}-01-01", f"{year_list[i]}-12-31", freq="D")
        day_idx = np.where((date.month == month_list[i]) & (date.day == day_list[i]))[0][0]
        date_idx.append(day_idx)

        fname = f"/work/b11209013/2024_Research/CloudSat/CS_sel/CS_{year_list[i]}_{(day_idx+1):03d}.joblib"

        if not os.path.exists(fname):
            print(f"{fname} do not exist")
            continue

        cs_data.append(jl.load(fname))

        output_year.append(year_list[i])
        output_date.append(day_idx)

    output_dict = {
        "year": output_year,
        "date": output_date,
    }

    jl.dump( output_dict, "Existed_events.joblib")

if __name__ == "__main__":
    main()

