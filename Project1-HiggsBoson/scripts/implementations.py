#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *

#Functions for computing the weights

# LEAST SQUARES

def least_squares(y, tx):
    """Calculate the least squares solution"""
    
    #w_opt=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(tx), tx)), np.transpose(tx)),y)
    a_w = np.dot(np.transpose(tx), tx)
    b_y = np.dot(np.transpose(tx), y)
    w_opt=np.linalg.solve(a_w, b_y)

    return (compute_loss(y, tx, w_opt), w_opt)

# LEAST SQUARES GD
    
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Calculate the least squares solution using Gradient Descent"""
    
    def compute_gradient(y, tx, w):
        """Computes the gradient"""
        
        g=np.zeros(len(w))
        e=y-np.dot(tx, w)
        g=-(1.0/len(y))*np.dot(np.transpose(tx), e)

        return (g)
    
    def gradient_descent(y, tx, w, max_iters, gamma):
        """Gradient descent algorithm"""
       
        for n_iter in range(max_iters):
            w=w-gamma*compute_gradient(y,tx,w)

        return (compute_loss(y, tx, w), w)
    
    return (gradient_descent(y, tx, initial_w, max_iters, gamma))

# LEAST SQUARES SGD
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Calculate the least squares solution using Stochastic Gradient Descent"""
    
    def compute_stoch_gradient(y, tx, w):
        """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
        
        g=np.zeros(len(w))
        e=y-np.dot(tx, w)
        g=-(1.0/len(y))*np.dot(np.transpose(tx), e)

        return (g)
    
    def stochastic_gradient_descent(y, tx, w, batch_size, max_iters, gamma):
        """Stochastic gradient descent algorithm."""

        for n_iter in range(max_iters):
            for y_batch, tx_batch in batch_iter(y, tx, batch_size):
                w=w-gamma*compute_stoch_gradient(y_batch, tx_batch, w)

        return (compute_loss(y, tx, w), w)

    return (stochastic_gradient_descent(y, tx, initial_w, 1, max_iters, gamma))

# RIDGE REGRESSION

def ridge_regression(y, tx, lambda_):
    """Calculate ridge regression using normal equations"""

    #w_opt=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(tx), tx)+(lambda_*np.identity(len(tx[0,:])))), np.transpose(tx)),y)
    a_wl = np.dot(np.transpose(tx), tx) + 2*lambda_*np.identity(len(tx[0,:]))*len(tx[:,0])
    b_y = np.dot(np.transpose(tx), y)
    w_opt=np.linalg.solve(a_wl, b_y)
    
    return (compute_loss(y, tx, w_opt), w_opt)

def split_data(x, y, ratio, seed=1):
    """Split the dataset based on the split ratio is for training"""
    
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    numtraindata=np.round(len(x)*ratio).astype(int)
    numtestdata=len(x)-numtraindata
    xtrain=np.zeros(numtraindata).astype(int)
    xtest=np.zeros(numtestdata)
    ytrain=np.zeros(numtraindata).astype(int)
    ytest=np.zeros(numtestdata)
    
    #xtrain=np.unique(x[np.random.randint(0, len(x), numtraindata)])
    xtrainpos=np.sort(np.random.choice(np.arange(len(x)), numtraindata, replace=0))
    xtrain=x[xtrainpos]
    ytrain=y[xtrainpos]
    
    xtestpos=np.sort(np.setdiff1d(np.arange(len(x)), xtrainpos))
    
    xtest=x[xtestpos]
    ytest=y[xtestpos]

    return(xtrain, ytrain, xtest, ytest)
    
def split_data_cross_validation(tx, ty, k_trozos, seed=1):
    """Split the dataset based cross validation, separating randomlyin equal 
    parts all the available data and saving it in a vectorof matrix (for x and y)"""
    
    np.random.seed(seed)
    
    numtraindata=len(tx)/k_trozos
    numtraindatares=len(tx)%k_trozos #50%8 #(len(x)-numtraindata*k_trozos) [0, 7]
    #Definición matrices almacenar matrices 'k' de training y test
    k_xdata=np.zeros((k_trozos, numtraindata+1, len(tx[0,:])))
    k_ydata=np.zeros((k_trozos, numtraindata+1))
    
    #Vector aleatorio de todas las posiciones de x
    xtrainpos=np.random.choice(np.arange(len(tx)), len(tx), replace=0)
    
    c=0
    h=0
    while(h<k_trozos):
        
        if(numtraindatares>=1 and h==0):
            c=c+1
            k_xdata[h, :, :]=tx[xtrainpos[h*numtraindata:(h+1)*numtraindata+c]]
            k_ydata[h, :]=ty[xtrainpos[h*numtraindata:(h+1)*numtraindata+c]]
            
        elif(h<numtraindatares):
            k_xdata[h, :, :]=tx[xtrainpos[h*numtraindata+c:(h+1)*numtraindata+c+1]]
            k_ydata[h, :]=ty[xtrainpos[h*numtraindata+c:(h+1)*numtraindata+c+1]]
            c=c+1

        else:
            k_xdata[h, 0:numtraindata, :]=tx[xtrainpos[h*numtraindata+c:(h+1)*numtraindata+c]]
            k_ydata[h, 0:numtraindata]=ty[xtrainpos[h*numtraindata+c:(h+1)*numtraindata+c]]
        
        h=h+1
    
    return(k_xdata, k_ydata)

def ridge_regression_cross_validation(phix, y, lambda_, k_trozos, seed):
    """Applying ridge regression over cross validation technique"""
    
    k_phixdata, k_ydata=split_data_cross_validation(phix, y, k_trozos, seed)
    
    numtraindata=len(phix)/k_trozos
    numtraindatares=len(phix)%k_trozos
    
    accuracy_test=np.zeros(k_trozos)
    accuracy_train=np.zeros(k_trozos)
    rmse_test=np.zeros(k_trozos)
    rmse_train=np.zeros(k_trozos)
    
    w_star=np.zeros((k_trozos, len(phix[0,:])))
    
    h=0
    while(h<k_trozos):

        phixte=k_phixdata[h]
        yte=k_ydata[h]
        phixtr=0
        ytr=0
        w=0
        
        if(h!=0 and 0<numtraindatares):
            phixtr=k_phixdata[0, :, :]
            ytr=k_ydata[0, :]
            w=1
        elif(h!=0 and 0>=numtraindatares):
            phixtr=k_phixdata[0, 0:numtraindata, :]
            ytr=k_ydata[0, 0:numtraindata]
            w=1  
        elif(h==0 and 1<numtraindatares):
            phixtr=k_phixdata[1, :, :]
            ytr=k_ydata[1, :]
            w=2
        elif(h==0 and 1>=numtraindatares):
            phixtr=k_phixdata[1, 0:numtraindata, :]
            ytr=k_ydata[1, 0:numtraindata]
            w=2
            
        while(w<k_trozos):
            if(w!=h and h>=numtraindatares):
                phixtr=np.concatenate((phixtr, k_phixdata[w, 0:numtraindata, :]), axis=0)
                ytr=np.concatenate((ytr, k_ydata[w, 0:numtraindata]), axis=0)
            if(w!=h and h<numtraindatares):
                phixtr=np.concatenate((phixtr, k_phixdata[w, :, :]), axis=0)
                ytr=np.concatenate((ytr, k_ydata[w, :]), axis=0)
            w=w+1
        
        loss, w_star[h, :]=ridge_regression(ytr, phixtr, lambda_)
        
        accuracy_test[h]=correctas_ls(yte, phixte, w_star[h])
        accuracy_train[h]=correctas_ls(ytr, phixtr, w_star[h])
        
        rmse_test[h]=(2.0*compute_loss(yte, phixte, w_star[h]))**(1.0/2)
        rmse_train[h]=(2.0*loss)**(1.0/2)
        
        h=h+1
    
    w_opt=sum(w_star)/(1.0*len(w_star[:,0]))
    
    return(rmse_test, rmse_train, accuracy_test, accuracy_train, w_opt)

# LOGISTIC REGRESSION IMPLEMENTATION

def sigmoid(t):
    """Applies sigmoid function on t"""
    
    return(1.0/(1.0+(np.exp(-t))))

def calculate_gradient(y, tx, w):
    """Computes the gradient of loss"""
    ypre=np.dot(tx, w)
    ypre[np.where(ypre <= 0.5)] = 0
    ypre[np.where(ypre > 0.5)] = 1
    
    k=1.0/len(y)
    
    return (k*np.dot(np.transpose(tx),(sigmoid(ypre)-y)))

def calculate_loss(y, tx, w):
    """Computes the cost by negative log likelihood"""
    
    ypre=np.dot(tx, w)
    ypre[np.where(ypre <= 0.5)] = 0
    ypre[np.where(ypre > 0.5)] = 1
    
    sg=sigmoid(ypre)
    multi=(np.ones(len(tx[:,0]))-np.transpose(y))*np.log(np.ones(len(tx[:,0]))-np.transpose(sg))
    
    return (-np.sum(np.log(sg)*y+np.transpose(multi)))/len(tx[0,:])

def logistic_regression(y, tx, w, max_iters, gamma):
    """Computes the technique of logistic regression"""
    
    losses = []
    tresshold=1e-7
    
    h=0
    while(h<max_iters):
        gradient = calculate_gradient(y, tx, w)
        loss = calculate_loss(y, tx, w)
        w=w-gamma*gradient
        losses.append(loss)
        if (len(losses)>1 and np.abs(losses[len(losses)]-losses[-2])<tresshold):
            gamma=gamma/10
            if (gamma<1e-10):
                h=max_iters
        if (len(losses)>100 and losses[len(losses)]>losses[-100]):
            gamma=gamma/10
        
        h=h+1
        
    return (w, losses[len(losses)])

# RIDGE LOGISTIC REGRESSION IMPLEMENTATION

def calculate_loss_reg(y, tx, w, lambda_):
    """Computes the cost by negative log likelihood with regularization"""
    return calculate_loss(y, tx, w)+(lambda_ /(2*len(y)))*np.power(np.linalg.norm(w), 2)


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
     """Do one step of gradient descen using logistic regression. Return the loss and the updated w"""
     loss = calculate_loss_reg(y, tx, w, lambda_)
     gradient = calculate_gradient(y, tx, w) + (1.0/len(tx[:,0]))*(lambda_/2)*w
     w=w-gamma*gradient
    
     return (w, loss, np.linalg.norm(gradient))

def reg_logistic_regression(y, tx, lambda_, w, max_iters, gamma):
    """Computes the technique of logistic regression with regularization"""
    
    losses = []
    tresshold=1e-7
    
    h=0
    while(h<max_iters):
        w, loss, grad_norm = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        losses.append(loss)
        if(len(losses)>100 and np.abs(losses[len(losses)]-losses[-100])<tresshold):
            gamma=gamma/10
            if (gamma<1e-10):
                h=max_iters
        if (len(losses)>100 and losses[len(losses)]>losses[-100]):
            gamma=gamma/10
        
        h=h+1
            
    return (w, losses[len(losses)])

# Functions for polynomial expansions
    
def polynomial_singular(x, degree):
    """Increases the number of weights using a polynomial expansion of each
    individual characteristic without mixing"""

    phi=np.ones((len(x[:,0]), len(x[0,:])*degree+1))
    
    h=0
    w=0
    while(h<len(x[:,0])):
        m=0
        while(m<len(x[0,:])):
            w=1
                
            while(w<=degree):
                phi[h, (m*degree+w)]=(x[h, m]**w)
                w=w+1
              
            m=m+1
        h=h+1
    
    return(phi)
             
def polynomial_mixed(x, degree):
    """Increases the number of weights using a polynomial expansion of each
    characteristic mixing one with degree-each other"""

    phi=np.ones((len(x[:,0]), len(x[0,:])*degree+1))

    h=0
    w=0
    while(h<len(x[:,0])):
        m=0
        while(m<len(x[0,:])):
            w=1
                
            while(w<=degree):
                k=0
                while(k<w):
                    if(m+k<len(x[0,:])):
                        phi[h, (m*degree+w)]=phi[h, (m*degree+w)]*x[h, m+k]
                    else:
                        phi[h, (m*degree+w)]=phi[h, (m*degree+w)]*x[h, (m+k)-len(x[0,:])]
                    
                    k=k+1
                
                w=w+1
              
            m=m+1
        h=h+1
    
    return(phi)
    
def polynomial_mixed_exp(x, degree):
    """Increases the number of weights using a polynomial expansion of each
    characteristic mixing one with degree-each other and with exponent"""
    
    phi=np.ones((len(x[:,0]), len(x[0,:])*degree+1))

    h=0
    w=0
    while(h<len(x[:,0])):
        m=0
        while(m<len(x[0,:])):
            w=1
                
            while(w<=degree):
                k=0
                while(k<w):
                    if(m+k<len(x[0,:])):
                        phi[h, (m*degree+w)]=phi[h, (m*degree+w)]*(x[h, m+k]**(k+1))
                    else:
                        phi[h, (m*degree+w)]=phi[h, (m*degree+w)]*(x[h, (m+k)-len(x[0,:])]**(k+1))
                    
                    k=k+1
                
                w=w+1
              
            m=m+1
        h=h+1
    
    return(phi)
                
# Functions for costs and accurracy

def compute_loss(y, tx, w):
    """Calculate the loss"""
    
    def compute_MSE(y,tx,w):
        """MSE"""
        
        #1 Form
        #MSE=0
        #h=0
        #while(h<len(y)):
        #    MSE=MSE+(y[h]-np.dot(tx[h,:], w))**2
        #    h=h+1
        #
        #MSE=MSE/(2-0*len(y))

        #2 Form
        MSE=(1/(2.0*len(y)))*np.sum((y-np.dot(tx, w))**2)

        return (MSE)
    
    def compute_MAE(y,tx,w):
        """MAE"""
        
        #1 Form
        #MAE=0
        #h=0
        #while(h<len(y)):
        #    MAE=MAE+np.abs((y[h]-np.dot(tx[h,:], w)))
        #    h=h+1
        #
        #MAE=MAE/(1.0*len(y))

        #2 Form
        MAE=(1.0/(len(y)))*np.sum(np.abs((y-np.dot(tx, w))))

        return (MAE)
    
    return (compute_MSE(y, tx, w))

def correctas_ls(y, tx, w):
    """Function that computes the percentaje of success of the model with a 
    x,y known set"""
    
    correctas=0
    h=0
    while(h<len(y)):
        
        if(np.sign(np.dot(tx[h,:], w))==y[h]):
            correctas=correctas+1
        h=h+1
    
    return(correctas*1.0/len(y))
    
# Functions for pre-processing

def cleaning_999_mean(x):
    """Function that changes the positions of the -999 values for the mean
    of each characteristic (columns)"""
    h=0

    while(h<len(x[0,:])):
        suma=0
        contribuyentes=0
        
        w=0
        while(w<len(x[:,0])):
            if(x[w,h]!=-999):
                suma=suma+x[w,h]
                contribuyentes=contribuyentes+1
            w=w+1
        
        mean=suma/(1.0*contribuyentes)
        
        m=0
        while(m<len(x[:,0])):
            if(x[m,h]==-999):
                x[m,h]=mean
            m=m+1
            
        h=h+1
        
def cleaning_999_zero(x):
    """Function that changes the positions of the -999 values for 0 (columns)"""
    h=0

    while(h<len(x[0,:])):
        
        m=0
        while(m<len(x[:,0])):
            if(x[m,h]==-999):
                x[m,h]=0
            m=m+1
            
        h=h+1 
        
def cleaning_999_remove_rows(x):
    """Function that deletes the rows in which there is any -999"""
    
    h=0
    
    while(h<len(x[:,0])):
        
        m=0
        while(m<len(x[0,:])):
            if(x[h,m]==-999):
                x=np.delete(x, (h), axis=0)
                y=np.delete(x, (h), axis=0)
                break
            m=m+1
            
        h=h+1  
        
def cleaning_999_remove_columns(x):
    """Function that deletes the columns in which there is any -999"""

    h=0
    
    while(h<len(x[:,0])):
        
        m=0
        while(m<len(x[0,:])):
            if(x[h,m]==-999):
                x=np.delete(x, (m), axis=1)
                m=m-1
            m=m+1
            
        h=h+1  
        
def cleaning_999_remove_rows_and_columns(x, y):
    """Function that deletes the rows and the columns in which there is any -999"""

    h=0
    
    while(h<len(x[:,0])):
        
        m=0
        while(m<len(x[0,:])):
            if(x[h,m]==-999):
                x=np.delete(x, (m), axis=1)
                x=np.delete(x, (h), axis=0)
                y=np.delete(x, (h), axis=0)
                break
            m=m+1
            
        h=h+1  


# Functions for standarization of the data

def standardize(x):
    """Standardize the original data set"""
    
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = np.true_divide(x,std_x)
    
    return (x, mean_x, std_x)

def standarize_by_characteristic(x):
    """Standardize the original data set by characteristic"""

    i=np.zeros((len(x[:,0]), len(x[0,:])))
    o=np.zeros(len(x[0,:]))
    h=0
    while(h<len(x[0,:])):
        i[:,h],p,o[h]=standardize(x[:,h])
        
        h=h+1

    return(i,o)
    
def replace_outliers_by_mean(x, o):
    """Replaces the 'considered' outliers by the mean (standardiced => 0)"""
    
    h=0
    
    while(h<len(x[:,0])):
        w=0
        while(w<len(x[0,:])):
            if(x[h,w]>5 or x[h,w]<-5):
                x[h,w]=0
            w=w+1
        h=h+1