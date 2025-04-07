# This program is to compute the variance map of Q1

# # import packages
import numpy as np;
import netCDF4 as nc;
from pandas import date_range;
from matplotlib import pyplot as plt;
from scipy.fft import fft, ifft, fftfreq, fftshift;
from joblib import Parallel, delayed;

from cartopy import crs as ccrs;
from cartopy import feature as cfeature;

def fft2(data, axes=(None, None)):
    fft_temp = fft(data, axis=axes[0]);
    fft_temp = ifft(fft_temp, axis=axes[1]) * data.shape[axes[1]];
    return fft_temp

def ifft2(data, axes=(None, None)):
    ifft_temp = ifft(data, axis=axes[0]);
    ifft_temp = fft(ifft_temp, axis=axes[1]) / data.shape[axes[1]];
    return ifft_temp

def main():
    # Load data 
    # # setting path
    fname : str = "/work/b11209013/2024_Research/nstcCCKW/t/tFlt.nc";

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

        q1: np.ndarray = f["t"][str_idx:trm_idx+1];

    print("Finish loading q1");
    q1_sym = (q1 + q1[:, :, ::-1]) / 2

    del q1;

    # Apply FFT along specified dimensions while keeping xarray metadata
    q1_fft = np.stack(
        Parallel(n_jobs=8)(delayed(fft2)(
            q1_sym[:, i, :, :], (0, 2))
            for i in range(q1_sym.shape[1])
        ), axis=1
    );
    print(q1_fft.shape)

    print("Finish FFT");

    del q1_sym;

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

    kel_sel = q1_fft * q1_sel[:, None, None, :];
    
    del q1_fft;

    # reconstruct q1
    q1_recon = np.stack(
        Parallel(n_jobs=8)(delayed(ifft2)(
            kel_sel[:, i, :, :], (0, 2))
            for i in range(kel_sel.shape[1])
        ), axis=1
    ).real;
    print("Finish reconstruct");

    # variance map of q1
    q1_var = np.var(q1_recon, axis=0);

    del q1_recon;

    q1_var_vint  = -np.trapz(q1_var, dims["lev"], axis=0) / np.trapz(dims["lev"]);
    del q1_var
    print("Finish variance");

    # plot variance map
    plt.figure(figsize=(12, 6))    
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

    # Plot the variable
    c = plt.contourf(dims["lon"], dims["lat"], q1_var_vint, 
                    cmap="Blues", levels=20, transform=ccrs.PlateCarree())

    # Coastlines and optional extent
    ax.coastlines()

    # Gridlines with labels
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Axis labels
    plt.title("T Variance (Vertically Integrated)", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # Colorbar
    cb = plt.colorbar(c, orientation="horizontal", aspect=30, shrink=0.8, pad=0.05)
    cb.set_label("T variance")  # Replace "units" with actual units, e.g., "K/day"
    
    plt.savefig("t_var_vint.png", dpi=500)
    plt.show()  

if __name__ == "__main__":
    main();

