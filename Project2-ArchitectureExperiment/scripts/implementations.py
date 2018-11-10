# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import *

#Functions for computing the weights

def least_squares(y, tx):
    """Calculate the least squares solution"""
    
    #w_opt=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(tx), tx)), np.transpose(tx)),y)
    a_w = np.dot(np.transpose(tx), tx)
    b_y = np.dot(np.transpose(tx), y)
    w_opt=np.linalg.solve(a_w, b_y)

    return (compute_loss(y, tx, w_opt), w_opt)

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

def ridge_regression(y, tx, lambda_):
    """Calculate ridge regression using normal equations"""

    #w_opt=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(tx), tx)+(lambda_*np.identity(len(tx[0,:])))), np.transpose(tx)),y)
    a_wl = np.dot(np.transpose(tx), tx) + 2*lambda_*np.identity(len(tx[0,:]))*len(tx[:,0])
    b_y = np.dot(np.transpose(tx), y)
    w_opt=np.linalg.solve(a_wl, b_y)
    
    return (compute_loss(y, tx, w_opt), w_opt)

def split_data(x, y, ratio, seed=1):
    """Split the dataset based on the split ratio. If ratio is 0.8 you will have 80% of your data set dedicated to training 
       and the rest dedicated to testing"""
    
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
    
    k_phixdata, k_ydata=split_data_cross_validation(phix, y, k_trozos, seed)
    
    numtraindata=len(phix)/k_trozos
    numtraindatares=len(phix)%k_trozos
    
    porcentajeaciertoste=np.zeros(k_trozos)
    porcentajeaciertostr=np.zeros(k_trozos)
    
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
        
        porcentajeaciertoste[h]=correctas_ls(yte, phixte, w_star[h])
        porcentajeaciertostr[h]=correctas_ls(ytr, phixtr, w_star[h])
        
        h=h+1
    
    w_opt=sum(w_star)/(1.0*len(w_star[:,0]))
    
    return(porcentajeaciertoste, porcentajeaciertostr, w_opt)

    
def polynomial_ones(x,degree):
    
    phi=np.ones((len(x[:,0]), len(x[0,:])*degree+1))

    
    return(phi) 
    
    
    

def polynomial_singular(x, degree):
    
    phi=np.ones((len(x[:,0]), len(x[0,:])*degree+1))
    
    h=0
    w=0
    while(h<len(x[:,0])):
        m=0
        while(m<len(x[0,:])):
            w=1
                
            while(w<=degree):
                phi[h, (m*degree+w)]=(x[h, m]**w/2.0)
                w=w+1
              
            m=m+1
        h=h+1
    
    return(phi)
             
def polynomial_mixed(x, degree):
    
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
                
#Functions for costs

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
    
    correctas=0
    h=0
    while(h<len(y)):
        
        if(np.sign(np.dot(tx[h,:], w))==y[h]):
            correctas=correctas+1
        h=h+1
    
    return(correctas*1.0/len(y))


def cleaning_999_mean(x):
    
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
    
    h=0

    while(h<len(x[0,:])):
        
        m=0
        while(m<len(x[:,0])):
            if(x[m,h]==-999):
                x[m,h]=0
            m=m+1
            
        h=h+1 
        
def cleaning_999_remove_rows(x,y):
    x2=np.zeros( (len(x[:,0]), len(x[0,:])) )
    y2=np.zeros(len(y))
    h=0
    i=0
    
    while(h<len(x[:,0])):
        m=0
        while(m<len(x[0,:]) and x[h,m]!=-999):
            if(m==len(x[0,:])-1):
                x2[i,:]=x[h,:]
                y2[i]=y[h]
                i=i+1
            
            m=m+1
            
        h=h+1  
        
    return(x2[0:i,:], y2[0:i])
        
def cleaning_999_remove_columns(x):
    x2=np.zeros( (len(x[:,0]), len(x[0,:])) )
    
    m=0
    i=0
    while(m<len(x[0,:])):
        h=0
        while(h<len(x[:,0]) and x[h,m]!=-999):
            if(h==len(x[0,:])-1):
                x2[:,i]=x[:,m]
                i=i+1
            
            h=h+1
            
        m=m+1  
    return(x2[:,0:1])
        


#Functions for standarization of the data

def standardize(x):
    """Standardize the original data set"""
    
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    
    return (x, mean_x, std_x)

def standardize_by_characteristic(x):
    
    i=np.zeros((len(x[:,0]), len(x[0,:])))
    h=0
    while(h<len(x[0,:])):
        i[:,h],p,o=standardize(x[:,h])
        
        h=h+1

    return(i)


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form"""
    
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    
    return (y, tx)

