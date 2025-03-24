#!/bin/sh

cd /work/b11209013/2024_Research/CloudSat/CloudSat_Interp

cdo -P 8 mergetime CloudSat_Interpolated_*.nc /work/b11209013/2024_Research/CloudSat/CloudSat_merged.nc