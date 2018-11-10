#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import *
from implementations import *

#### Loading of the data
ydata, xdata, idsdata = load_csv_data("../data/train.csv", False)
yguess, xguess, idsguess = load_csv_data("../data/test.csv", False)

# Printing their shapes
#print("shape of xtrain {}".format(xdata.shape))
#print("shape of ytrain {}".format(ydata.shape))
#print("shape of idstrain {}".format(idsdata.shape))

### Standardization of the data set by characteristic

x_data=np.copy(xdata)
cleaning_999_mean(x_data)
x_data,o=standarize_by_characteristic(x_data)
replace_outliers_by_mean(x_data, o)
x_data[abs(x_data)<0.000001]=0

#### Uploading: once we know the best hyperparameters
#Selected hyperparamerts
degree_upload=11
lambda_upload=0.0000373

# Obtaining the model
x_data=np.copy(xdata)
cleaning_999_mean(x_data)
x_data,o=standarize_by_characteristic(x_data)
replace_outliers_by_mean(x_data, o)
x_data[abs(x_data)<0.000001]=0

phix_data = polynomial_singular(x_data, degree_upload)  
losses, w_star=ridge_regression(ydata, phix_data, lambda_upload)

# Applying the model to the guess data
x_guess=np.copy(xguess)
cleaning_999_mean(x_guess)
x_guess,o=standarize_by_characteristic(x_guess)
replace_outliers_by_mean(x_guess, o)

phix_guess = polynomial_singular(x_guess, degree_upload)  
y_upload=np.dot(phix_guess, w_star)
y_upload=np.sign(y_upload)

# Creating submission
create_csv_submission(idsguess, y_upload, "../results/Subida 4")

accuracy_data=correctas_ls(ydata, phix_data, w_star)

