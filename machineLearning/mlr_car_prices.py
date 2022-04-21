"""
A Machine Learning algorithm to predict car prices

@author: Ms. Namasivayam (replace with your name)
@version: 02/23/2022
@source: CodeHS
"""

import pandas as pd
import matplotlib.pyplot as plt

''' Load Data '''
data = pd.read_csv("car.csv")
x_1 = data["miles"]
x_2 = data["age"]
y = data["Price"]

''' Visualize Data '''
fig, graph = plt.subplots(2)
graph[0].scatter(x_1, y)
graph[0].set_xlabel("Total Miles")
graph[0].set_ylabel("Price")

graph[1].scatter(x_2, y)
graph[1].set_ylabel("Price")
graph[1].set_xlabel("Car Age")

print("Correlation between Total Miles and Car Price:", x_1.corr(y))
print("Correlation between Age and Car Price:", x_2.corr(y))

plt.tight_layout()
plt.show()

''' TODO: Create Linear Regression '''
# Reload and/or reformat the data to get the values from x and y

# Separate data into training and test sets

# Create multivariable linear regression model

# Find and print the coefficients, intercept, and r squared values.
# Each rounded to two decimal places.

# Test the model

# Print out the actual vs the predicted values
