# Import packages
import sys
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def main():
    # Load CCKWs events (time selections)
    with nc.Dataset("/home/b11209013/2025_Research/CloudSat/CloudSat_Itp/CCKW_select.nc", "r") as f:
        sel_time = f["time"][:]

    print("Finish loading sel_time")

    # Load CloudSat data with Dask for parallel processing
    with nc.Dataset("/work/b11209013/2024_Research/CloudSat/CloudSat_merged.nc", "r") as f:
        dims = {
            key: f[key][:]
            for key in f.dimensions.keys()
        }

        lon_lim = np.where((dims["lon"] >= 160) & (dims["lon"] <= 260))[0]
        lat_lim = np.where((dims["lat"] >= -15) & (dims["lat"] <= 15))[0]
    
        time_idx = np.array([find_nearest(dims["time"], t) for t in sel_time])


        qlw = f.variables["qlw"][time_idx, :, lat_lim, lon_lim]
        qsw = f.variables["qsw"][time_idx, :, lat_lim, lon_lim]

    

    print("Finish loading CloudSat data")

    print("qlw shape:", qlw.shape)

    # Load ERA5 data with Dask
    with nc.Dataset("/work/b11209013/2024_Research/nstcCCKW/t/tFlt.nc", "r") as f:
        lev_era5 = f.variables["lev"][:]/100.
        temp = f.variables["t"][time_idx, :, :, lon_lim]

    print(temp.shape)
    #print("Finish loading ERA5 data")

    sel_qlw_mean  = np.nanmean(qlw, axis=(0, 2))
    sel_qsw_mean  = np.nanmean(qsw, axis=(0, 2))
    sel_temp_mean = np.nanmean(temp, axis=(0, 2))

    print("Finish Processing data")

    print("shape of qlw mean:", sel_qlw_mean.shape)

    # Plotting
    plt.figure(figsize=(8, 6))
    cf = plt.contourf(dims["lon"][lon_lim], dims["lev"], sel_qlw_mean, cmap="RdYlBu_r")
    c = plt.contour(dims["lon"][lon_lim], lev_era5, sel_temp_mean, colors="k")
    plt.gca().invert_yaxis()
    plt.coloribar(cf)
    plt.xlabel("Longitude")
    plt.ylabel("Level")
    
    plt.clabel(c, inline=1, fontsize=10)
    plt.title("CloudSat Composite")
    plt.show()

if __name__ == "__main__":
    main()
