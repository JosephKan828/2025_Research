# This program is to select the Kelvin wave significant events from the ERA5 data
# Import packages
import sys
import numpy as np
import xarray as xr
from pandas import date_range
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

    print(np.where(q1_recon.var(axis=0) == q1_recon.var(axis=0).max()))
    
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)})
    
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", linestyle="--")
    m = ax.contourf(dims["lon"], dims["lat"], q1_recon.var(axis=0), cmap="Blues", transform=ccrs.PlateCarree(), levels=20)
    ax.plot(dims["lon"][335], dims["lat"][43], "ro", transform=ccrs.PlateCarree())
    plt.colorbar(m, ax=ax, orientation="horizontal", shrink=0.8, aspect=40)
    plt.show()
    
if __name__ == "__main__":
    main()