# This program is to select the Kelvin wave significant events from the ERA5 data
# Import packages
import sys
import numpy as np
import xarray as xr
from pandas import date_range
import matplotlib.pyplot as plt

sys.path.append("/home/b11209013/Package")
import Theory as th # type: ignore

def load_data():
    path = "/work/b11209013/2024_Research/nstcCCKW/Q1/Q1Flt.nc"
    
    date = date_range("1979-01-01", "2021-12-31", freq="D")

    str_idx = np.where(date.year == 2006)[0][162]
    trm_idx = np.where(date.year == 2017)[0][330]

    time_slice = slice(str_idx, trm_idx)

    with xr.open_dataset(path, engine="netcdf4") as f:
        dims = dict(f.coords)
        
        dims["time"] = dims["time"].isel(time=slice(str_idx, trm_idx+1))

        q1 = f.Q1.isel(time=slice(str_idx, trm_idx+1))
    return dims, q1

def vert_int(dims, data) -> np.ndarray:
    data = data.chunk({"time": "auto"})
    
    data_ave = ((data.shift(lev=-1) + data) / 2).isel(lev=slice(0, -1))

    data_vint = -(data_ave * (dims.shift(lev=-1) - dims).isel(lev=slice(0, -1))).sum("lev")
    
    return data_vint * 86400/9.81/2.5e6

def fft2(data):
    fft = np.fft.fft(data, axis=0)
    fft = np.fft.ifft(fft, axis=2) * data.shape[2]

    return fft

def ifft2(data):
    ifft = np.fft.ifft(data, axis=0)
    ifft = np.fft.fft(ifft, axis=2) / data.shape[2]
    return ifft

def main():
    dims, q1 = load_data()

    q1_vint = vert_int(dims["lev"], q1)

    q1_fft = fft2(q1_vint)

    # define kelvin waves dispersion curve
    kel_curve = lambda wn, ed: wn * np.sqrt(9.81 * ed) * (86400 / (2*np.pi*6.371e6))

    # setup coordinate
    wn = np.fft.fftfreq(len(dims["lon"]), d=1/len(dims["lon"])).astype(int)
    fr = np.fft.fftfreq(len(dims["time"]))

    wnm, frm = np.meshgrid(wn, fr)

    wnm, frm = wnm[:, None, :], frm[:, None, :]

    q1_sel = np.where(
        ((wnm >= 1) & (wnm <=14) &
        (frm >= 1/20) & (frm <= 1/2.5) &
        (frm >= kel_curve(wnm, 8)) & (frm <= kel_curve(wnm, 90))) |
        ((wnm <= -1) & (wnm >=-14) &
        (frm <= -1/20) & (frm >= -1/2.5) &
        (frm >= kel_curve(wnm, 8)) & (frm <= kel_curve(wnm, 90)))
        , q1_fft, 0
        )

    q1_recon = ifft2(q1_sel).real

    lon_m, lat_m = np.meshgrid(dims["lon"], dims["lat"])
    lon_m, lat_m = lon_m[None, :, :], lat_m[None, :, :] 

    q1_selected = np.where((lon_m >=180) & (lon_m <= 240) & (lat_m >= 0) & (lat_m <= 10), q1_recon, np.nan)
    
    q1_selected = np.nanmean(q1_selected, axis=(1, 2))
    
    # Selecting time indices where q1_selected is greater than mean + 2*std
    time_sel = dims["time"][np.where(q1_selected > q1_selected.mean() + 2 * q1_selected.std())].values

    # Creating an xarray Dataset with time as a coordinate
    ds = xr.Dataset(
        #data_vars={},  # No data variables, just coordinates
        coords={"time": ("time", time_sel)}  # Define 'time' as a coordinate
    )

    # Save to NetCDF
    ds.to_netcdf("CCKW_select.nc")



if __name__ == "__main__":
    main()