import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import geopy.distance

#data = pd.read_csv("crop yield data/Attainable yields (Mueller et al. 2012).csv")
# ds = xr.open_dataset("crop yield data/maize/yield_1981.nc4")
# dataframe = ds.to_dataframe()
# print(dataframe.shape)


data = xr.open_dataarray("gdhy_v1.2_v1.3_20190128/maize/yield_1993.nc4")

print(data.values)


#def fillArrays():


data = xr.Dataset(coords = {"year": range(1981, 2016)})

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