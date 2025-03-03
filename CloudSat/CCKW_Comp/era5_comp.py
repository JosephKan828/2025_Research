# This program is to composite CCKW with ERA5 data
# Import packages
import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load selected date
    with nc.Dataset("/home/b11209013/2025_Research/CloudSat/CloudSat_Itp/CCKW_select.nc", "r") as f:
        sel_time = f.variables["time"][:]

    # Load ERA5
    with nc.Dataset("/work/b11209013/2024_Research/nstcCCKW/Q1/Q1Flt.nc", "r") as f:
        dims = {
            key: f.variables[key][:]
            for key in f.dimensions.keys()
        }

        str_idx = np.argmin(np.abs(dims["time"] - 933072))
        trm_idx = np.argmin(np.abs(dims["time"] - 1033728))

        lon_lim = np.where((dims["lon"] >= 160) & (dims["lon"] <= 260))[0]
        lat_lim = np.where((dims["lat"] >= -15) & (dims["lat"] <= 15))[0]

        q1 = f.variables["Q1"][str_idx:trm_idx+1, :, lat_lim, lon_lim][sel_time].mean(axis=(0, 2))

    c = plt.contourf(dims["lon"][lon_lim], dims["lev"], q1)
    plt.gca().invert_yaxis()
    plt.colorbar(c)
    plt.show()

if __name__ == '__main__':
    main()
