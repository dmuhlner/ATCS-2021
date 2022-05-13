import itertools
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import geopy.distance
import dask
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import sklearn.neural_network as nn
# from sklearn.model_selection import GridSearchCV
from joblib import dump, load

#start work timer
start = time.time()
# ccrs = cr.crs


#commented code generally is code that was used previously but does not need to be compiled each time it is run
#data = pd.read_csv("crop yield data/Attainable yields (Mueller et al. 2012).csv")
# ds = xr.open_dataset("crop yield data/maize/yield_1981.nc4")
# dataframe = ds.to_dataframe()
# print(dataframe.shape)

def findTime(filename):
    year = filename[-8:-4]
    print(filename)
    return int(year)

#function for compiling food datasets
def preProcess(ds):
    filename = ds.encoding["source"]
    time = findTime(filename)
    data = ds.assign_coords({"year": time})
    print(data)
    return data


def combineMFFolder(folder):
    return xr.open_mfdataset((folder + "*.nc4"), preprocess=preProcess, combine="nested", concat_dim = "year")

# def precipToYear():


# def viewPrecipitation(year):
#     precip = xr.open_dataset("climateData/precip1900-2020.nc")
#     #time = pd.date_range("2000-01-01", freq="H", periods=365 * 24)
#     # precip = precip.sel(time="1991-01-16")
#     prec = precip.variables["timeseries-pr-annual-mean"]
#     prec = prec[year-1901, :, :]
#     print(precip)
#     print(nc.Dataset("climateData/climatology-pr-annual-mean_cru_annual_cru-ts4.05-climatology_mean_1991-2020.nc").variables)
#     lats = precip.variables["lat"][:]
#     lons = precip.variables["lon"][:]
#     map = ccrs.PlateCarree()
#
#     ax = plt.axes(projection=map)
#     plt.contourf(lons, lats, prec, 60,transform=map)
#     # ax.coastlines()
#
#     plt.show()

#function to search data and validate that it is not identical to itself
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

# def visualizeSlice(data, independent, year):
#     ind = data.variables[independent]
#     ind = ind[year - 1981, :, :]
#     lats = data.variables["lat"][:]
#     lons = data.variables["lon"][:]
#     map = ccrs.PlateCarree()
#
#     ax = plt.axes(projection=map)
#     plt.contourf(lons, lats, ind, 60, transform=map)
#     plt.show()

# def makePreProcess(X):
#     wrapper = skx.wrap(StandardScaler())
#     wrapper.fit(X)
#     wrapper2 = skx.wrap(train_test_split())
#     wrapper2.fit(X)
#
# data = xr.open_dataset("combinedDataset.nc4")

#function made to be used with xarray ML models (not used)
def modelFit(coords, rain, tempmax, tempmin, ave, xc, yc, rainc, maxc, minc, avec, xex, yex, rex, maex, miex, avex,scalar):
    x, y = coords
    val= scalar + xc * np.power(x, xex)\
         + yc * np.power(y, yex) + \
         rainc * np.power(rain, rex) + \
         np.power(tempmax, maex) * maxc + \
         np.power(tempmin, miex) * minc + \
         np.power(ave, avex) * avec
    return val

# variable = data._variables["timeseries-pr-annual-mean"]
# print(variable)
#
# data[["Maize", "edSoy", "dRice", "Wheat"]].curvefit(
#     coords=["lat", "lon"],
#     reduce_dims=["timeseries-pr-annual-mean", "lon", "timeseries-tasmax-annual-mean", "lat", "timeseries-tasmin-annual-mean", "timeseries-tas-annual-mean"],
#     func=modelFit)
#
# print(data)

# sanitizer = skx.preprocessing.Sanitizer()
# data = sanitizer.fit_transform(data["Maize", "edSoy", "dRice", "Wheat","timeseries-pr-annual-mean", "timeseries-tas-annual-mean", "timeseries-tasmax-annual-mean", "timeseries-tasmin-annual-mean"])
#
# data=pd.read_csv("dataframe.csv")
# data = xr.open_dataset("combinedDataset.nc4")
# data = data.to_dataframe(dim_order=["year", "lon", "lat", "bnds"])
# data = data.reset_index(level="year")
# print(data)
# print(data)
# print(data.loc[[-179.75]].shape)

#function building linear model (replaced by neural network)
def linearModel(data):
    data = data.dropna()
    print(data)
    x = data[["year", "lon_bnds", "lat_bnds", "timeseries-tas-annual-mean", "timeseries-pr-annual-mean", "timeseries-tasmax-annual-mean", "timeseries-tasmin-annual-mean"]]
    y = data[["Maize", "edSoy", "dRice", "Wheat"]]
    print(x)
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    model = LinearRegression().fit(X=xtrain, y=ytrain)

#function building and training a neural network
def neuralModel(data):
    data = data.dropna()
    print(data["year"])
    x = data[["year", "timeseries-tas-annual-mean", "timeseries-pr-annual-mean", "timeseries-tasmax-annual-mean",
              "timeseries-tasmin-annual-mean"]]
    y = data[["Maize", "edSoy", "dRice", "Wheat"]]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    scaler = StandardScaler().fit(xtrain)
    xtrain = scaler.transform(xtrain)
    xtest = scaler.transform(xtest)
    dump(scaler, "foodScaler.joblib")
    neural = nn.MLPRegressor(verbose=True, max_iter=6000, batch_size=20000, activation="tanh", solver="lbfgs", hidden_layer_sizes=(100, 12, 80, 50))
    # tuneModel(neural, x, y)
    neural = neural.fit(xtrain, ytrain)

    # predict = model.predict(xtest)
    print("Train score:", neural.score(xtrain, ytrain))
    print("Test score:", neural.score(xtest, ytest))
    print(neural.get_params())
    return neural

#function to try many hyperparameter options concurrently to determine best parameters for neural network
def tuneModel(model, X, Y):
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    solver = ["sgd", "adam", "lbfgs"]
    first = [100, 200, 500]
    second = [8, 12, 20]
    third = [50, 80, 100]
    fourth = [3, 5, 7]

    structures = list(itertools.product(first, second, third, fourth))
    parameterGrid = dict(solver=solver, hidden_layer_sizes=structures)
    searchGrid = GridSearchCV(estimator=model, param_grid=parameterGrid, n_jobs=-1, cv=3)
    gridResult = searchGrid.fit(X, Y)
    print(gridResult)
    print("Best: %f using %s" % (gridResult.best_score_, gridResult.best_params_))


# dump(neuralModel(data), "foodModel.joblib")
# model = load("foodModel.joblib")
# scaler = load("foodScaler.joblib")
#
# data = data.dropna()
# x = data[["year", "timeseries-tas-annual-mean", "timeseries-pr-annual-mean", "timeseries-tasmax-annual-mean",
#           "timeseries-tasmin-annual-mean"]]
# y = data[["Maize", "edSoy", "dRice", "Wheat"]]
# x = scaler.transform(x)
# print(model.score(x, y))

#function to make second neural network (focused on weather data)
def makeModel2(data, filepath):
    data = data[["year", "lon_bnds", "lat_bnds", "timeseries-tas-annual-mean", "timeseries-pr-annual-mean", "timeseries-tasmax-annual-mean",
              "timeseries-tasmin-annual-mean"]]
    data = data.dropna()
    y = data[["timeseries-tas-annual-mean", "timeseries-pr-annual-mean", "timeseries-tasmax-annual-mean",
              "timeseries-tasmin-annual-mean"]]
    x = data[["year", "lon_bnds", "lat_bnds"]]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    scaler = StandardScaler().fit(xtrain)
    xtrain = scaler.transform(xtrain)
    xtest = scaler.transform(xtest)
    dump(scaler, "weatherScaler.joblib")
    neural = nn.MLPRegressor(verbose=True, max_iter=10)
    # tuneModel(neural, x, y)
    neural = neural.fit(xtrain, ytrain)

    # predict = model.predict(xtest)
    print("Train score:", neural.score(xtrain, ytrain))
    print("Test score:", neural.score(xtest, ytest))
    print(neural.get_params())
    dump(neural, filepath)

# makeModel2(data, "weatherModel.joblib")
#function to verify functionality of weather model
def checkWeather(data):
    model = load("weatherModel.joblib")
    scaler = load("weatherScaler.joblib")
    data = data[["year", "lon_bnds", "lat_bnds", "timeseries-tas-annual-mean", "timeseries-pr-annual-mean", "timeseries-tasmax-annual-mean",
              "timeseries-tasmin-annual-mean"]]
    data = data.dropna()
    y = data[["timeseries-tas-annual-mean", "timeseries-pr-annual-mean", "timeseries-tasmax-annual-mean",
          "timeseries-tasmin-annual-mean"]]
    x = data[["year", "lon_bnds", "lat_bnds"]]
    x = scaler.transform(x)
    print(model.score(x, y))

#function for UX
def getYearAndLocation():
    year = input("Year for prediction:")
    lon = input("Longitude for prediction:")
    lat = input("Latitude for prediction:")
    return year, lon, lat

#main UX function
def makePrediction():
    year, lon, lat = getYearAndLocation()
    weatherData = predictWeather(year, lon, lat)
    foodData = predictFood(weatherData, year)
    dataSet = xr.open_dataset("combinedDataset.nc4")
    compareSet = dataSet.sel(year=[2016], lon=[lon], lat=[lat], method="nearest")
    compare = {"Maize" : compareSet["Maize"].values, "Soy" : compareSet["edSoy"].values, "Rice" : compareSet["dRice"].values, "Wheat" : compareSet["Wheat"].values}
    for key in compare:
        if compare[key]:
            compare[key] = 0
    print("In 2016:", compare)
    return foodData, compareSet

#loads weather model and predicts based on given parameters
def predictWeather(year, lon, lat):
    model = load("weatherModel.joblib")
    scaler = load("weatherScaler.joblib")
    data = [[year, lon, lat]]
    data = scaler.transform(data)
    return model.predict(data)

#loads food model and predicts based on given parameters
def predictFood(data, year):
    model = load("foodModel.joblib")
    scaler = load("foodScaler.joblib")
    data = np.insert(data, [0], year)
    data = data.reshape(1, -1)
    data = scaler.transform(data)
    return model.predict(data)


predicted, comparison = makePrediction()
print("             Maize:       Soy:         Rice:        Wheat:")
print("Predicted:", predicted)
print("Predictions are predicting possibility, not actuality.")


# dataSet = xr.open_dataset("combinedDataset.nc4")
# dataSet = dataSet.dropna(dim="Maize")
# print(dataSet.sel(year=[2016]))

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

#prints total time spent on process
print("Time spent:", time.time() - start, "seconds")