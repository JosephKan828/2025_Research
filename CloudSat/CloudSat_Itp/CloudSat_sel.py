# This program is to interpolate the CloudSat data to a pressure coordinate
# import packages
import os 
import sys
import glob
import numpy as np
import xarray as xr

from pyhdf.HC  import HC
from pyhdf.SD  import SD, SDC
from pyhdf.VS  import VS
from pyhdf.HDF import HDF
import joblib
from joblib import Parallel, delayed

year = int(sys.argv[1])
date = int(sys.argv[2])

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File path '{filepath}' does not exist.")
        sys.exit(1)  # Shut down the script with a non-zero exit code

def load_data(fname):
    from pyhdf.VS  import VS
    hdf = HDF(fname, SDC.READ)
    
    vs = hdf.vstart()
    lat = np.array(vs.attach("Latitude")[:])
    lon = np.array(vs.attach("Longitude")[:])
    vs.end()
    
    lon[lon <= 0] += 360
    
    cond = np.where((lat >= -15) & (lat <= 15) & (lon >= 160) & (lon <= 260))

    lon = lon[cond]; lat = lat[cond]

    file_sd = SD(fname, SDC.READ)

    hgt_all = np.array(file_sd.select("Height")[:])
    qr_all = np.array(file_sd.select("QR")[:])

    file_sd.end()

    hgt = hgt_all[cond]
    qlw = qr_all[0, cond] / 100.
    qsw = qr_all[1, cond] / 100.

    return {
        "lon": lon, "lat": lat, "hgt": hgt,
        "qlw": qlw, "qsw": qsw
    }



def main():
    path = f"/work/DATA/Satellite/CloudSat/{year}/{date:03d}/"

    check_file_exists(path)
    
    file_list = glob.glob(path + "*.hdf")
    
    limit_data = Parallel(n_jobs = -1)(delayed(load_data)(fname) for fname in file_list)

    new_data = []

    for f in limit_data:
        if not len(f["lon"]) == 0:
            new_data.append(f)
        else:
            break

    joblib.dump(limit_data, f"/work/b11209013/2024_Research/CloudSat/CS_sel/CS_{year}_{date:03d}.joblib", compress=("zlib", 1))

    
if __name__ == "__main__":
    main()
