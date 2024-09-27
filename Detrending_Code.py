from matplotlib import pyplot as plt
import numpy as np

###

# functions to help with the average part
def calculate_percentile(sorted_data, percentile):
    index = (len(sorted_data) - 1) * percentile
    lower = int(np.floor(index))
    upper = int(np.ceil(index))
    if lower == upper:
        return sorted_data[int(index)]
    return sorted_data[lower] * (upper - index) + sorted_data[upper] * (index - lower)

def remove_outliers(x, y):
    Q1 = calculate_percentile(y, 0.25)
    Q3 = calculate_percentile(y, 0.75)
    IQR = Q3 - Q1
    outliers = [x for x in y if x < (Q1 - 1.5 * IQR) or x > (Q3 + 1.5 * IQR)]
    good_y = []
    good_x = []
    for index, value in enumerate(y):
        if value not in outliers:
            good_x.append(x[index])
            good_y.append(value)
    return (good_x, good_y)

###

def detrend(x, y):

    # Fits a 5th order polynomial to the data
    coefficients = np.polyfit(x, y, 5)

    # Creates a polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(x)

    # Plots the original data and the polynomial fit (for visual)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', s=0.5)
    plt.scatter(x, y_fit, s=0.5)
    plt.show()

    # getting the average
    # code is messy because sometimes there are no outliers
    new_x, new_y = remove_outliers(x, y)
    try:
        average = sum(new_y) / len(new_y)
    except:
        average = sum(y) / len(y)
    #

    # I get the weights by subtracting the average from values in the fitted polynomial
    # Then use these to subtract the trend from the data
    weights = [i - average for i in y_fit]
    y_new = [y[i] - weights[i] for i in range(len(weights))]
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_new, s=0.5)
    average = [average, average]
    plt.plot(np.linspace(min(x), max(x), 2), average)
    plt.show()

    return x, y_new, polynomial

# y is detrended
# polynomial is the fitted polynomial
def retrend(x, y, polynomial):
    polynomial = polynomial(x)
    average = sum(y) / len(y)
    weights = [i - average for i in polynomial]
    y_new = [y[i] + weights[i] for i in range(len(weights))]
    return y_new
