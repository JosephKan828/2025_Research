## Selecting Kelvi waves events in ERA5
# using package
using FFTW
using Dates
using JSON3
using Plots; pythonplot()
using PlotUtils
using NCDatasets
using Statistics
using LazyGrids: ndgrid
using GeoMakie, CairoMakie
FFTW.set_num_threads(12)

# defin fft2 functions
function fft2(data::Array{Float32, 4})
  data_comp :: Array{ComplexF32, 4} = complex.(data);

  fft!(data_comp, (1,));
  ifft!(data_comp, (4, ));

  data_comp .*= size(data, 4);

  return data_comp;
end

function ifft2(data::Array{ComplexF32, 4})
  ifft!(data, (1,));
  fft!(data, (4, ));

  data ./= size(data, 4);

  return data;
end

function vert_int(data::Array{Float32, 3}, lev::Array{Float32, 1})
  data_ave  :: Array{Float32, 3} = (data[:, :, 2:end] .+ data[:, :, 1:end-1]) ./ 2;

  data_vint :: Array{Float32, 3} = sum(data_ave .* reshape(lev[2:end] .- lev[1:end-1], 1, 1, size(lev, 1)-1), dims=(3,)) / sum(lev[2:end] .- lev[1:end-1]);

  return data_vint;
end

# define functions for Kelvin waves
function kel_curves(wn, ed)
  return wn * sqrt(9.81 .* ed) .* (86400 / (2 * pi * 6.371e6));
end


# Load data
## file name
fname :: String = "/work/b11209013/2024_Research/nstcCCKW/t/tFlt.nc";

## Seting dates and time index
init_date :: Date = Date(1979, 1, 1)  ;
term_date :: Date = Date(2021, 12, 31);

data_array :: Array{Date, 1} = collect(init_date:Day(1):term_date);

## Set targeting range
str_date :: Date = Date(2006, 6, 11);
end_date :: Date = Date(2017, 11, 26);

### Find index of start and end date
str_idx :: Int32 = findfirst(isequal(str_date), data_array);
end_idx :: Int32 = findfirst(isequal(end_date), data_array);

## Load data
ds :: NCDataset  = NCDataset(fname, "r");

lev :: Array{Float32, 1} = ds["lev"][:];
lon :: Array{Float32, 1} = ds["lon"][:];
lat :: Array{Float32, 1} = ds["lat"][:];
q1 :: Array{Float32, 4} = ds["t"][:, :, :, str_idx:end_idx];

close(ds);

### Compute anomaly
q1_mean :: Array{Float32, 4} = mean(q1, dims = (1, 4)); # remove climatology and zonal-mean
q1_ano  :: Array{Float32, 4} = q1 .- q1_mean;

## Compute FFT
@time q1_fft2 :: Array{ComplexF32, 4} = fft2(q1_ano);

## setting axis
fr :: Array{Float32, 1} = fftfreq(size(q1, 4), 1);
wn :: Array{Float32, 1} = fftfreq(size(q1, 1), size(q1, 1));

wnm, frm = ndgrid(wn, fr);

cond_pos :: Array{Bool, 2} = (
    (wnm .>= 1) .& (wnm .<= 14) .&
    (frm .>= 1/20) .& (frm .<= 1/2.5) .&
    (frm .>= kel_curves(wnm, 8)) .& (frm .<= kel_curves(wnm, 90))
);

cond_neg :: Array{Bool, 2} = (
    (wnm .<= -1) .& (wnm .>= -14) .&
    (frm .<= -1/20) .& (frm .>= -1/2.5) .&
    (frm .>= kel_curves(wnm, 8)) .& (frm .<= kel_curves(wnm, 90))
);

kel_cond :: Array{Bool, 2} = (cond_pos .|| cond_neg);

println("Shape of kel_cond: ", size(kel_cond));

## masking q1
q1_masked :: Array{ComplexF32, 4} = q1_fft2 .* reshape(kel_cond, size(q1_fft2, 1), 1, 1, size(q1_fft2, 4));

## Reconstruct
@time q1_recon :: Array{Float32, 4} = real.(ifft2(q1_masked));

## Compute time variance of data
variance = dropdims(var(q1_recon, dims=(4,)), dims=(4,));

## Vertically integral
var_vint :: Array{Float32, 2} = dropdims(vert_int(variance, lev), dims=3);

# Plot contour map with coastlines
fig = Figure(; size=(1200, 500))
ga = GeoAxis(fig[1, 1];
             aspect=AxisAspect(7.5), dest="+proj=merc +lon_0=180", title="t variance map", xgridcolor=:black,
             xgridwidth=0.2, ygridcolor=:black, ygridwidth=0.2, limits=((90, 300), (-15, 15)),
            )

# Contour plot
cont = GeoMakie.contourf!(ga, lon, lat, var_vint;
                          colormap=:Blues, levels=20)

# Plot coastlines
coast = GeoMakie.coastlines()
lines!(ga, coast; color=:black, linewidth=1)

# Set labels
ga.xlabel = "Longitude"
ga.ylabel = "Latitude"

Colorbar(fig[2, 1], cont; label="Temp variance (m2/s2)", vertical=false, width=Relative(1.0))
# Save the figure
save("t_var_vint.png", fig)
