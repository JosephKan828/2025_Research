# This program is to composite CCKW with ERA5 data
# Import packages
import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load selected date
    with nc.Dataset("/home/b11209013/2025_Research/CloudSat/ERA5_select/CCKW_select.nc", "r") as f:
        sel_time = f.variables["time"][:]

    # Load ERA5
    with nc.Dataset("/work/b11209013/2024_Research/nstcCCKW/Q1/Q1Flt.nc", "r") as f:
        dims = {
            key: f.variables[key][:]
            for key in f.dimensions.keys()
        }

        time_idx = np.array([np.searchsorted(dims["time"], t) for t in sel_time])

        lat_lim = np.where((dims["lat"] >= 0) & (dims["lat"] <= 5))[0]
        lon_lim = np.where((dims["lon"] >= 150) & (dims["lon"] <= 260))[0]

        q1 = f.variables["Q1"][time_idx, :, lat_lim, lon_lim] * 86400 / 1004.5

    with nc.Dataset("/work/b11209013/2024_Research/nstcCCKW/t/tFlt.nc", "r") as f:
        t = f.variables["t"][time_idx, :, lat_lim, lon_lim]
    
        t -= t.mean(axis=(0, 3), keepdims=True)
    
    c = plt.contourf(dims["lon"][lon_lim], dims["lev"], np.nanmean(q1, axis=(0, 2)), cmap="RdBu_r")
    ct = plt.contour(dims["lon"][lon_lim], dims["lev"], np.nanmean(t, axis=(0, 2)), colors="k", levels=20)
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.clabel(ct, inline=True,)
    plt.colorbar(c)
    plt.show()

if __name__ == '__main__':
    main()
