# This program is to compute the variance map of Q1

# # import packages
import numpy as np;
import xarray as xr;
import netCDF4 as nc;
from pandas import date_range;
from matplotlib import pyplot as plt;
from scipy.fft import fft, ifft, fftfreq, fftshift;

from cartopy import crs as ccrs;
from cartopy import feature as cfeature;

# functions

def vert_int(data: np.ndarray, lev: np.ndarray) -> np.ndarray:
    # Compute layer thickness (can be precomputed if lev is constant)
    lev_diff = np.diff(lev)
    
    # Perform vertical integration directly without intermediate arrays
    data_vint = -np.sum((data[:, 1:] + data[:, :-1]) / 2 * lev_diff[None, :, None, None], axis=1)
    
    return data_vint

def main():
    # Load data 
    # # setting path
    fname : str = "/work/b11209013/2024_Research/nstcCCKW/Q1/Q1Flt.nc";

    # # set date range
    date = date_range("1979-01-01", "2021-12-31", freq="D");

    str_idx = np.where(date.year == 2006)[0][162];
    trm_idx = np.where(date.year == 2017)[0][330];

    # # load q1 data
    with nc.Dataset(fname, "r") as f:
        dims: dict[str, xr.dataarray] = {
            key: f[key][:]
            for key in f.dimensions.keys()
        };

        dims["time"] = dims["time"][str_idx:trm_idx+1];

        q1: np.ndarray = f["Q1"][str_idx:trm_idx+1];

    q1_vint: np.ndarray = vert_int(q1, dims["lev"]) * 86400 / 9.8 / 2.5e6;

    del q1;    # Compute symmetry along specified dimension
    
    q1_sym = (q1_vint + q1_vint[:, ::-1]) / 2

    del q1_vint;

    # Apply FFT along specified dimensions while keeping xarray metadata

    q1_fft: np.ndarray = fft(q1_sym, axis=0); # fft on time
    q1_fft: np.ndarray = ifft(q1_fft, axis=2) * dims["lon"].shape[0]; # fft on lon

    # Apply bandpass filter on q1_fft
    fr: np.ndarray = fftfreq(dims["time"].shape[0], d=1);
    wn: np.ndarray = fftfreq(dims["lon"].shape[0], d=1/dims["lon"].shape[0]).astype(int);

    wnm, frm = np.meshgrid(wn, fr);

    ## define kelvin waves curve
    kel_curve = lambda wn, ed: wn * np.sqrt(9.81 * ed) * (86400 / (2*np.pi*6.371e6));

    q1_sel: np.ndarray = np.where(
        ((wnm >= 1) & (wnm <=14) &
        (frm >= 1/20) & (frm <= 1/2.5) &
        (frm >= kel_curve(wnm, 8)) & (frm <= kel_curve(wnm, 90))
         ) |
        ((wnm <= -1) & (wnm >=-14) &
        (frm <= -1/20) & (frm >= -1/2.5) &
        (frm <= kel_curve(wnm, 8)) & (frm >= kel_curve(wnm, 90))
        ), 1, 0
            );

    kel_sel = q1_fft * q1_sel[:, None, :];

    # reconstruct q1
    q1_recon = fft(kel_sel, axis=2) / dims["lon"].shape[0];
    q1_recon = ifft(q1_recon, axis=0);

    # variance map of q1
    q1_var = np.var(q1_recon, axis=0);

    plt.figure(figsize=(12, 6))    
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

    # Plot the variable
    c = plt.contourf(dims["lon"], dims["lat"], q1_var, 
                    cmap="Blues", levels=20, transform=ccrs.PlateCarree())

    # Coastlines and optional extent
    ax.coastlines()

    # Gridlines with labels
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Axis labels
    plt.title("Q1 Variable (Vertically Integrated)", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # Colorbar
    cb = plt.colorbar(c, orientation="horizontal", aspect=30, shrink=0.8, pad=0.05)
    cb.set_label("Q1 variance")  # Replace "units" with actual units, e.g., "K/day"
    
    plt.savefig("q1_var_vint.png", dpi=500)
    plt.show() 

if __name__ == "__main__":
    main();

