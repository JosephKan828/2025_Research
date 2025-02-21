# This program is to interpolate the z into denser coordinate
# Import package
import numpy as np
import xarray as xr
import joblib as jl

# Load dataset with selected events.
def main():
    # Load existed events idx
    existed = jl.load("/home/b11209013/2025_Research/CloudSat/CloudSat_Itp/Existed_events.joblib")
    year = existed["year"]
    date = existed["date"]

    # Load z data
    z_sel = []
    
    lev = np.array([1000, 925, 850, 700, 500, 250, 200 ,100])
    
    lat_range = [14.75, 14.75]
    lon_range = [160.3125, 259.6875]
    
    for i in range(len(year)):
        fname = f"/work/b11209013/2024_Research/nstcCCKW/z/z_{year[i]}.nc"
        with xr.open_dataset(fname, decode_times=False) as f:
            
            dims = f.coords
            if i==0:
                lat_idx = dims["lat"].to_index().get_indexer([-14.75, 14.75])
                lon_idx = dims["lon"].to_index().get_indexer([160.3125, 259.6875])
            z_selected = f.z.isel(
                time=date[i],
                lat=slice(lat_idx[0], lat_idx[1]+1),
                lon=slice(lon_idx[0], lon_idx[1]+1)
            )
            z_sel.append(z_selected)

    z_sel = xr.concat(z_sel, dim="time")
    z_sel = z_sel.assign_coords(sfc=lev).rename({"sfc":"lev"})
    
    
    # interpolate
    lev_new = np.linspace(1000, 100, 37)
    
    z_interp = z_sel.interp(lev=lev_new)
    print(z_interp)
    z_interp.to_netcdf("/work/b11209013/2024_Research/CloudSat/z_interp.nc")
    
if __name__ == "__main__":
    main()