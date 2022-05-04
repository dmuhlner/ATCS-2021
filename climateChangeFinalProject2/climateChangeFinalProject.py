import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import geopy.distance
import dask
import cartopy as cr

ccrs = cr.crs



#data = pd.read_csv("crop yield data/Attainable yields (Mueller et al. 2012).csv")
# ds = xr.open_dataset("crop yield data/maize/yield_1981.nc4")
# dataframe = ds.to_dataframe()
# print(dataframe.shape)

def findTime(filename):
    year = filename[-8:-4]
    print(filename)
    return int(year)


def preProcess(ds):
    filename = ds.encoding["source"]
    time = findTime(filename)
    data = ds.assign_coords({"year": time})
    print(data)
    return data

def combineMFFolder(folder):
    return xr.open_mfdataset((folder + "*.nc4"), preprocess=preProcess, combine="nested", concat_dim = "year")

def viewPrecipitation():
    precip = xr.open_mfdataset(["climateData/climatology-pr-annual-mean_cru_annual_cru-ts4.05-climatology_mean_1991-2020.nc", "climateData/climatology-pr-annual-mean_cru_annual_cru-ts4.05-climatology_mean_1961-1990.nc"], combine = "nested", concat_dim= "time")
    time = pd.date_range("2000-01-01", freq="H", periods=365 * 24)
    # precip = precip.sel(time="1991-01-16")
    prec = precip.variables["climatology-pr-annual-mean"]
    prec = prec[0, :, :]
    print(precip)
    print(nc.Dataset("climateData/climatology-pr-annual-mean_cru_annual_cru-ts4.05-climatology_mean_1991-2020.nc").variables)
    lats = precip.variables["lat"][:]
    lons = precip.variables["lon"][:]
    map = ccrs.PlateCarree()

    ax = plt.axes(projection=map)
    plt.contourf(lons, lats, prec, 60,transform=map)
    # ax.coastlines()

    plt.show()


climate = xr.open_dataset("climateData/annualTimeseries.nc")

def parseYear(year):
    return pd.Timestamp(str(year) + "-01-06")


#climate = xr.open_mfdataset(["climateData/aggregatedPrecip.nc4", "climateData/annualTimeseries.nc"], combine = "nested", concat_dim= "time")
print(climate)
# precip.to_netcdf("climateData/aggregatedPrecip.nc4")

# p = precip.isel(time=0).plot(
#         subplot_kws=dict(projection=ccrs.Orthographic(-80, 35), facecolor="gray"),
#         transform=ccrs.PlateCarree(),
#     )
#
# p.axes.set_global()
# p.axes.coastlines()

# xr.Dataset.to_netcdf(combineMFFolder("gdhy_v1.2_v1.3_20190128/wheat/"), "gdhy_v1.2_v1.3_20190128/aggregatedWheat.nc4")
#
# print(xr.open_dataset("gdhy_v1.2_v1.3_20190128/aggregatedWheat.nc4"))

#def fillArrays():

#
# data = xr.Dataset(coords = {"year": range(1981, 2016)})

#
# yields = data["var"][:]

#
# print(otherVals)


# h = nc.variables[vname]
# times = nc.variables['time']
# jd = netCDF4.num2date(times[:],times.units)
# hs = pd.Series(h[:,station],index=jd)
#
# fig = plt.figure(figsize=(12,4))
# ax = fig.add_subplot(111)
# hs.plot(ax=ax,title='%s at %s' % (h.long_name,nc.id))
# ax.set_ylabel(h.units)