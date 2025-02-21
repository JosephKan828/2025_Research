# This program is to interpolate the CLoudSat to ERA5 pressure coord
# Import package
import numpy as np
import xarray as xr
import joblib as jl
from itertools import product
from scipy.interpolate import interp1d

def main():
    
    # load existed date 
    existed = jl.load("/home/b11209013/2025_Research/CloudSat/CloudSat_Itp/Existed_events.joblib")
    
    year = existed["year"]
    date = existed["date"]
    
    # load interpolated era5 geopotential
    with xr.open_dataset("/work/b11209013/2024_Research/CloudSat/z_interp.nc") as f:
        dims = f.coords
        
        z = f.z / 9.81
        
    # Load cloudsat data
    new_year, new_date, cs_data, new_z = [], [], [], []

    i = 0

    for y, d in zip(year, date):
        
        try:
            path = f"/work/b11209013/2024_Research/CloudSat/CS_sel/CS_{y}_{(d+1):03d}.joblib"
        
            temp_data = jl.load(path)
        
            daily = [entry for entry in temp_data if entry["lon"].shape[0] != 0]
            for d in temp_data:
                if not d["lon"].shape[0] == 0:
                    daily.append(d)
        
            if daily:
                new_year.append(y)
                new_date.append(d)
                cs_data.append(daily)
                new_z.append(z[i])

        except (IndexError, FileNotFoundError):
            continue 

        i += 1

    year[:] = new_year 
    date[:] = new_date
    z = np.array(new_z)

    print(z.shape)

    # interpolate the CloudSat data with geopotential height 

    cs_interp = np.emtpty((451, 37, 60, 160), dtype=object)

    for i in range(len(year)):
        for f in cs_data[i]:
            lon_cs = f["lon"]
            lat_cs = f["lat"]
            hgt_cs = f["hgt"]
            qlw_cs = f["qlw"]
            qsw_cs = f["qsw"]

            for la, lo in product(dims["lat"], dims["lon"]):
                lo_cond = np.where((lon_cs < lo+0.3125) & (lon_cs >= lo-0.3125))
                la_cond = np.where((lat_cs < la+0.25) & (lat_cs >= la-0.25))

                if (len(lon_cs[lo_cond]) == 0) | (len(lat_cs[la_cond]) == 0):
                    cs_interp[i, :, la, lo] = np.nan
                else:
                    qlw_interp = interp1d(hgt_cs, )
                    qsw_interp = interp1d(hgt_cs, )
                
    
if __name__ == "__main__":
    main()