import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import geopy.distance
import dask

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