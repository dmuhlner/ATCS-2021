"""
@author: Ms. Namasivayam (replace with your name)
@version: 02/23/2022
@source: CodeHS
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

''' Load Data '''
data = pd.read_csv("data/blood_pressure.csv")
x = data["Age"]
y = data["Blood Pressure"]

''' TODO: Create Linear Regression '''
# Get the values from x and y
# Use reshape to turn the x values into 2D arrays:

# Create the model

# Find the slope and intercept
# Each should be a float and rounded to two decimal places.


# Print out the linear equation

# Predict the the blood pressure of someone who is 43 years old.

# Print out the prediction

''' Visualize Data '''
# set the size of the graph
plt.figure(figsize=(5, 4))

# label axes and create a scatterplot
plt.xlabel("Age")
plt.ylabel("Systolic Blood Pressure")
plt.title("Systolic Blood Pressure by Age")
plt.scatter(x, y)
plt.show()

print("Pearson's Correlation: r = :", x.corr(y))
