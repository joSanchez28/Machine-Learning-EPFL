#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *

#### Loading of the data

ydata, xdata, idsdata = load_csv_data("../data/train.csv", False)
yguess, xguess, idsguess = load_csv_data("../data/test.csv", False)

### Standardization of the data set by characteristic

x_data=np.copy(xdata)
cleaning_999_mean(x_data)
x_data,o=standarize_by_characteristic(x_data)
replace_outliers_by_mean(x_data, o)
x_data[abs(x_data)<0.000001]=0

# Functions for plotting
        
def plot_degrees(degrees, degrees_plot, lambdas, split_ratios, split_ratios_plot, acc_rmse_test, acc_rmse_train, acc_rms_str):
    
    plt.figure()
    h=0
    while(h<len(degrees_plot)):   
        plt.title("Degrees-Lambdas")
        plt.xlabel("Lambdas")
        plt.ylabel(acc_rms_str)
        plt.semilogx(lambdas, acc_rmse_test[(np.where(degrees==degrees_plot[h]))[0][0], :, np.where(split_ratios==split_ratios_plot)[0][0]], label='Degree' + "%d" %(degrees_plot[h]))
        plt.legend(loc=1)
    
        h=h+1
        
def plot_degree_train_test(degrees, degree_plot, lambdas, split_ratios, split_ratios_plot, acc_rmse_test, acc_rmse_train, acc_rms_str):
    
    plt.figure()
    plt.title("Degree " + "%d" %(degree_plot))
    plt.xlabel("Lambdas")
    plt.ylabel(acc_rms_str)
    plt.semilogx(lambdas, acc_rmse_test[(np.where(degrees==degree_plot))[0][0], :, np.where(split_ratios==split_ratios_plot)[0][0]], color='b', label='Test')
    plt.semilogx(lambdas, acc_rmse_train[(np.where(degrees==degree_plot))[0][0], :, np.where(split_ratios==split_ratios_plot)[0][0]], color='g', label='Train')
    plt.legend(loc=1)

def plot_lambdas(degrees, lambdas, lambdas_plot, split_ratios, split_ratios_plot, acc_rmse_test, acc_rmse_train, acc_rms_str):
    
    h=0
    while(h<len(lambdas_plot)):   
        plt.figure()
        plt.title("Lambdas-Degrees")
        plt.xlabel("Degrees")
        plt.ylabel(acc_rms_str)
        plt.semilogx(degrees, acc_rmse_test[:, np.where(lambdas==lambdas_plot[h])[0][0], np.where(split_ratios==split_ratios_plot)[0][0]], label='Degree' + "%.9f" %(lambdas_plot[h]))
        plt.legend(loc=1)
        plt.pause(0.5)
        
        h=h+1
        
def plot_lambda_train_test(degrees, lambdas, lambda_plot, split_ratios, split_ratios_plot, acc_rmse_test, acc_rmse_train, acc_rms_str):
      
    plt.figure()
    plt.title("Lambdas " + "%.9f" %(lambda_plot))
    plt.xlabel("Degrees")
    plt.ylabel(acc_rms_str)
    plt.semilogx(degrees, acc_rmse_test[:, np.where(lambdas==lambda_plot)[0][0], np.where(split_ratios==split_ratios_plot)[0][0]], color='b', label='Test')
    plt.semilogx(degrees, acc_rmse_train[:, np.where(lambdas==lambda_plot)[0][0], np.where(split_ratios==split_ratios_plot)[0][0]], color='g', label='Train')
    plt.legend(loc=1)

# Demo-Functions

"""
Function name: 
    finding_the_best_model

Arguments:
    x: input data points, length 250000x30
    y: outputs, 250000x1 {-1,1}
    degree: polynomy's degree
    lambda_: regularization parameter
    ratio: percentaje of trainning data over all the known input data
    seed: for generating randomness
    
Functionality:
    This function is going to be used to compute the different configurations
    for the models using a polinomy based in exponentation of single 
    characteristics. It first divide the data in two different sets: one 
    for which is going to be used to obtain the model and another for test. 
    Then it computes the ridge_regression for the traning set, taking the 
    indicated lambda. This way we obtain the weights most appropiate for
    this polinomy and degree, and we compute de RMSE for the traning set and
    the RMSE for the test set. Then we calculate the success percentaje
    over both sets, what is going to be the main value to decide which model
    is better than the others.
    
Outputs:
    accuracy_success_tr: percentaje of success in the traning set
    accuracy_success_te: percentaje of success in the test set
    
"""

def demo_finding_the_best_model():
    
    #### Principal method to find the best model without cross-validation, and using
    # ridge_regression
    
    def finding_the_best_model(phix, y, degree, lambda_,  split_ratio, seed):
        """polynomial regression with different split ratios and different degrees."""
     
        phix_train, y_train, phix_test, y_test = split_data(phix, y, split_ratio, seed)
        
        losses, w_star=ridge_regression(y_train, phix_train, lambda_)
        
        accuracy_test=correctas_ls(y_test, phix_test, w_star)
        accuracy_train=correctas_ls(y_train, phix_train, w_star)
        rmse_train=(2.0*losses)**(1.0/2)
        rmse_test=(2.0*compute_loss(y_test, phix_test, w_star))**(1.0/2)
        
        print("split_ratio={p}, degree={d}, lambda_={l:.9f}, Test RMSE={rte:.3f}, Train RMSE={rtr:.3f}, %test={a:.5f}, %train={b:.5f}".format(
              p=split_ratio, d=degree, l=lambda_, rte=rmse_test, rtr=rmse_train, a=accuracy_test, b=accuracy_train))
    
        
        return(rmse_test, rmse_train, accuracy_test, accuracy_train)
        
    def plots():
        ### Plotting desired configurations. Will plot using the nearest values asignned 
        # in the previous 'Main loop'
        
        ## Accuracy
            
        # For concrete values of degree and ratio, using vector 'lambdas'
        degrees_plot=np.copy(degrees) #np.array([1,2,3])
        split_ratios_plot=[0.85]
            
        plot_degrees(degrees, degrees_plot, lambdas, split_ratios, split_ratios_plot, accuracy_test, accuracy_train, "Accuracy %")   
        
        # For concrete values of lambda and ratio, usgin vector 'degrees'
        lambdas_plot=np.concatenate((np.array([0]), np.logspace(-9, -1, 22)), axis=0)
        split_ratios_plot=0.85
        
        plot_lambdas(degrees, lambdas, lambdas_plot, split_ratios, split_ratios_plot, accuracy_test, accuracy_train, "Accuracy %")   
        
        ## RMSE
        degrees_plot=np.copy(degrees) #np.array([1,2,3])
        split_ratios_plot=0.85
        
        plot_degrees(degrees, degrees_plot, lambdas, split_ratios, split_ratios_plot, rmse_test, rmse_train, "RMSE")   
        
        # For concrete values of lambda and ratio, usgin vector 'degrees'
        lambdas_plot=np.concatenate((np.array([0]), np.logspace(-9, -1, 22)), axis=0)
        split_ratios_plot=0.85
        
        plot_lambdas(degrees, lambdas, lambdas_plot, split_ratios, split_ratios_plot, rmse_test, rmse_train, "RMSE")   
        
        
    ## Main loop using the aforementioned function
    # Parameters
    seed = 6
    degrees = np.array([3, 6, 7, 10, 11]) #np.arange(1,3)
    split_ratios = np.array([0.85])
    lambdas = np.concatenate((np.array([0]), np.logspace(-9, -1, 22)), axis=0)
    
    # Matrixes for sabing the results
    accuracy_test=np.zeros((len(degrees), len(lambdas), len(split_ratios)))
    accuracy_train=np.zeros((len(degrees), len(lambdas), len(split_ratios)))
    rmse_test=np.zeros((len(degrees), len(lambdas), len(split_ratios)))
    rmse_train=np.zeros((len(degrees), len(lambdas), len(split_ratios)))
    
    # Loop
    h=0
    while(h<len(split_ratios)):
        
        w=0
        while(w<len(degrees)):
            phix_data=polynomial_singular(x_data, degrees[w])
            
            m=0
            while(m<len(lambdas)):
                rmse_test[w,m,h], rmse_train[w,m,h], accuracy_test[w,m,h], accuracy_train[w,m,h]=finding_the_best_model(phix_data, ydata, degrees[w], lambdas[m], split_ratios[h], seed)
                
                m=m+1
            w=w+1
        h=h+1
    
    plots()

demo_finding_the_best_model()

"""
Function name: 
    finding_the_best_model_cross_validation

Arguments:
    x: input data points, length 250000x30 (in this case)
    y: outputs, 250000x1 {-1,1} (in this case)
    degree: polynomy's degree
    lambda_: regularization parameter
    k_part: number of divisions in which divide all the available data
    seed: for generating randomness
    
Functionality:
    This function is going to be used to compute the different configurations
    for the models using a polinomy based in exponentation of single 
    characteristics. The first thing it does is the ridge_regression_cross_validation
    which is a function that applys ridge regression over the cross validation
    technique, allowing us to use all the data for training and for testing. This
    will lead us to the best model, and as the number of divisions incresases
    the precision in the model obtained will be bigger. We compute de RMSE for 
    the means of training and test, and the  accuracy. The one with larger accuracy
    will be the model which tell us which degree and lambda we should use for
    obatining the correct weights to be applied over unknown-result data. Once
    we know the degree and lambda, we should use the simple ridge regression
    to get our w_star.
    
Outputs:
    rmse_test: rmse of the mean of the the test divisions
    rmse_train: rmse of the mean of the train divisions
    accuracy_success_tr: percentaje of success in the traning set
    accuracy_success_te: percentaje of success in the test set
    
"""

def demo_finding_the_best_model_cross_validation():
    
    #### Principal method to find the best model with cross-validation, and using
    # ridge_regression
    
    def finding_the_best_model_cross_validation(phix, y, degree, lambda_,  k_part, seed):
        """polynomial regression with different split ratios and different degrees."""
     
        rmse_test_v, rmse_train_v, accuracy_test_v, accuracy_train_v, w_star=ridge_regression_cross_validation(phix, y, lambda_, k_part, seed)
        
        accuracy_test=np.mean(accuracy_test_v)
        accuracy_train=np.mean(accuracy_train_v)
        rmse_test=np.mean(rmse_test_v)
        rmse_train=np.mean(rmse_train_v)
        
        print("k_part={p}, degree={d}, lambda_={l:.9f}, Test RMSE={rte:.3f}, Train RMSE={rtr:.3f}, %test={a:.5f}, %train={b:.5f}".format(
              p=k_part, d=degree, l=lambda_, rte=rmse_test, rtr=rmse_train, a=accuracy_test, b=accuracy_train))
    
        
        return(rmse_test, rmse_train, accuracy_test, accuracy_train)
    
    def plots():
        ### For plots we should use the same functions as previously
        ## Accuracy
            
        # For concrete values of degree and ratio, using vector 'lambdas'
        degrees_plot=np.copy(degrees) #np.array([1,2,3])
        k_parts_plot=6
            
        plot_degrees(degrees, degrees_plot, lambdas, k_parts, k_parts_plot, accuracy_test_cv, accuracy_train_cv, "Accuracy %")   
        
        # For concrete values of lambda and ratio, usgin vector 'degrees'
        lambdas_plot=np.concatenate((np.array([0]), np.logspace(-9, -1, 22)), axis=0)
        k_parts_plot=6
        
        plot_lambdas(degrees, lambdas, lambdas_plot, k_parts, k_parts_plot, accuracy_test_cv, accuracy_train_cv, "Accuracy %")
        
        ## RMSE
        degrees_plot=np.copy(degrees) #np.array([1,2,3])
        split_ratios_plot=0.85
        
        plot_degrees(degrees, degrees_plot, lambdas, k_parts, k_parts_plot, rmse_test_cv, rmse_train_cv, "RMSE")   
         
        # For concrete values of lambda and ratio, usgin vector 'degrees'
        lambdas_plot=np.concatenate((np.array([0]), np.logspace(-9, -1, 22)), axis=0)
        split_ratios_plot=0.85
        
        plot_lambdas(degrees, lambdas, lambdas_plot, k_parts, k_parts_plot, rmse_test_cv, rmse_train_cv, "RMSE")
    
    ## Main loop using the aforementioned function
    # Parameters
    seed = 6
    degrees = np.arange(1,14)
    k_parts = np.array([6])
    lambdas = np.concatenate((np.array([0]), np.logspace(-9, -1, 22)), axis=0)
    
    # Matrixes for sabing the results
    accuracy_test_cv=np.zeros((len(degrees), len(lambdas), len(k_parts)))
    accuracy_train_cv=np.zeros((len(degrees), len(lambdas), len(k_parts)))
    rmse_test_cv=np.zeros((len(degrees), len(lambdas), len(k_parts)))
    rmse_train_cv=np.zeros((len(degrees), len(lambdas), len(k_parts)))
    
    # Loop
    h=0
    while(h<len(k_parts)):
        
        w=0
        while(w<len(degrees)):
            phix_data=polynomial_singular(x_data, degrees[w])
            
            m=0
            while(m<len(lambdas)):
                rmse_test_cv[w,m,h], rmse_train_cv[w,m,h], accuracy_test_cv[w,m,h], accuracy_train_cv[w,m,h]=finding_the_best_model_cross_validation(phix_data, ydata, degrees[w], lambdas[m], k_parts[h], seed)
                
                m=m+1
            w=w+1
        h=h+1
    
    plots()

demo_finding_the_best_model_cross_validation()