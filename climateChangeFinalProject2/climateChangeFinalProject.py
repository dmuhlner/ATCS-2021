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

# def precipToYear():


def viewPrecipitation(year):
    precip = xr.open_dataset("climateData/precip1900-2020.nc")
    #time = pd.date_range("2000-01-01", freq="H", periods=365 * 24)
    # precip = precip.sel(time="1991-01-16")
    prec = precip.variables["timeseries-pr-annual-mean"]
    prec = prec[year-1901, :, :]
    print(precip)
    print(nc.Dataset("climateData/climatology-pr-annual-mean_cru_annual_cru-ts4.05-climatology_mean_1991-2020.nc").variables)
    lats = precip.variables["lat"][:]
    lons = precip.variables["lon"][:]
    map = ccrs.PlateCarree()

    ax = plt.axes(projection=map)
    plt.contourf(lons, lats, prec, 60,transform=map)
    # ax.coastlines()

    plt.show()

def checkSame():
    precip = xr.open_dataset("climateData/precip1900-2020.nc")

    gen = np.random.default_rng()
    counter = 0
    data = precip.data_vars["timeseries-pr-annual-mean"]
    for i in range(1000):
        index = [int(gen.random() * 360), int(gen.random() * 720)]
        if data[0][index[0]][index[1]] == data[119][index[0]][index[1]]:
            counter+= 1

    print(counter)
    return

def aggregateClimate():
    files = ["climateData/annualTimeseries.nc", "climateData/precip1900-2020.nc", "climateData/maxTemps.nc", "climateData/minTemps.nc"]
    opened =[]
    for file in files:
        opened.append(xr.open_dataset(files[files.index(file)]))
    climate = xr.merge(opened)
    return climate.isel(year=range(80, 116))


def aggregateCrops():
    files = ["gdhy_v1.2_v1.3_20190128/aggregatedMaize.nc4", "gdhy_v1.2_v1.3_20190128/aggregatedSoy.nc4", "gdhy_v1.2_v1.3_20190128/aggregatedRice.nc4", "gdhy_v1.2_v1.3_20190128/aggregatedWheat.nc4"]
    opened =[]
    for file in files:
        index = files.index(file)
        opened.append(xr.open_dataset(files[index]).rename({"var" : file[-9:-4]}))
    crops = xr.merge(opened)
    return crops

def visualizeSlice(data, independent, year):
    ind = data.variables[independent]
    ind = ind[year - 1981, :, :]
    lats = data.variables["lat"][:]
    lons = data.variables["lon"][:]
    map = ccrs.PlateCarree()

    ax = plt.axes(projection=map)
    plt.contourf(lons, lats, ind, 60, transform=map)
    plt.show()


#visualizeSlice(aggregateCrops(), "Maize", 2015)
#xr.Dataset.to_netcdf(xr.merge([xr.open_dataset("aggregatedClimateData.nc4"), aggregateCrops()]), "combinedDataset.nc4")
# climate = xr.open_dataset("climateData/annualTimeseries.nc")
# #precip = xr.open_dataset("climateData/aggregatedPrecip.nc4")
#
# def yearToDatetime(year):
#     return pd.to_datetime(str(year), )
#
# climate.assign_coords({"year":})
#
# print(climate)
#
# print(nc.Dataset("climateData/annualTimeseries.nc").variables)

def parseYear(yearIn):
    return pd.Timestamp(year = yearIn, month = 1, day=16, hour=0)


# precipitation1 = nc.Dataset("climateData/climatology-pr-annual-mean_cru_annual_cru-ts4.05-climatology_mean_1961-1990.nc")
# print(precipitation1.variables["climatology-pr-annual-mean"].shape)

# precip = xr.open_dataset("climateData/precip1900-2020.nc")
# print(precip.data_vars["timeseries-pr-annual-mean"])

# print(precipitation.time)

# precipitation.assign_coords({"year": precipitation.time.asType(object).year})
# print(precipitation)

#climate = xr.open_mfdataset(["climateData/aggregatedPrecip.nc4", "climateData/annualTimeseries.nc"], combine = "nested", concat_dim= "time")

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

#convert float32 time in days since 1900-1-1 to int year