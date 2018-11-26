#!/usr/bin/env python
# coding: utf-8

# # Architecture experiment: Subjective and physiological responses to interiors
# 
# 
# ### Libraries
# 
# - [scikit-learn](http://scikit-learn.org/stable/)
# - pandas
# - matplotlib
# - seaborn

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


DATA_FOLDER = './data/' #The data is in the .gitignore in order to not upload it to the GitHub repository


# In[37]:


get_ipython().system('jupyter nbconvert --to script Project2-Notebook.ipynb #to export from the jupyter notebook(.ipynb) to a .py file.')


# ## Exploratory Data Analysis

# First we load the data into a Pandas data frame (with the Pandas library; to install this package with conda run:
# `conda install -c anaconda pandas`):

# In[19]:


df_room_features = pd.read_csv(DATA_FOLDER + 'T_features.csv')
df_labels_and_metrics = pd.read_csv(DATA_FOLDER + 'table_dataset_GreeceSwitzerland_N265_metrics_mSC5_JPEGtoBMP_Michelson_RMS.csv')


# Let's see what is inside `T_features.csv`:

# In[25]:


df_room_features


# In[26]:


df_room_features.shape


# In[23]:


df_room_features.columns


# We have obtained a data frame with 36 rows × 31 columns. Each column is explained below:
# * 'filename': File where we have the cube map projections associated to every room. (See in `./Cubemap_Projections/BMP/<filename>` or `./Cubemap_Projections/JPEG/<filename>`). The filenames have the structure "p(pattern_id)_(context)_(SkyType)_sg_largewin_simu".
# * 'Pattern': 6 x patterns (determine how are the blinds, and so how are the shadows)
# * 'Context': 2 x context scenarios (social scenario, work scenario)
# * 'SkyType': 2 x countries (Greece –138 participants, Switzerland –127 participants)
# * Means and medians of the outputs (of what people have answer in their respective survey): 'mean_pleasant', 'median_pleasant', 'mean_interesting', 'median_interesting', 'mean_calming', 'median_calming', mean_exciting', 'median_exciting'
# * Room metrics (for describing the room quantitatively): 'mean_complex', 'median_complex', 'mean_bright', 'median_bright', 'mean_view', 'median_view', 'mean_spacious', 'median_spacious', 'contrast_mean_mSC5', 'contrast_max_mSC5', 'contrast_mean_Michelson', 'contrast_mean_Michelson_cube123', 'contrast_mean_Michelson_cube3', 'contrast_mean_RMS', 'contrast_mean_RMS_cube123','contrast_mean_RMS_cube3', 'complexity_mean_JPEGtoBMP', 'complexity_cube3_JPEGtoBMP', 'complexity_cube123_JPEGtoBMP'

# In[27]:


df_labels_and_metrics.head()


# In[28]:


df_labels_and_metrics.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# SEE THE APPLIED ML TUTORIAL BELOW:

# # Applied Machine Learning with Scikit Learn - Regressions
# 
# *Adapted from https://github.com/justmarkham*
# 
# ### Libraries
# 
# - [scikit-learn](http://scikit-learn.org/stable/)
# - pandas
# - matplotlib
# 
# In this tutorial we will see some basic example of Linear Regression for prediction and Logistic Regression for classification.
# 

# # Prediction with Linear Regression
# 
# |     *            | continuous     | categorical    |
# | ---------------- | -------------- | -------------- |
# | **supervised**   | **regression** | classification |
# | **unsupervised** | dim. reduction | clustering     |
# 
# ### Motivation
# 
# Why are we learning linear regression?
# - widely used
# - runs fast
# - easy to use (not a lot of tuning required)
# - highly interpretable
# - basis for many other methods
# 

# Let's import the dataset:

# In[2]:


data = pd.read_csv('Advertising.csv', index_col=0)
data.head()


# What are the **features**?
# - TV: advertising dollars spent on TV for a single product in a given market (in thousands of dollars)
# - Radio: advertising dollars spent on Radio
# - Newspaper: advertising dollars spent on Newspaper
# 
# What is the **response**?
# - Sales: sales of a single product in a given market (in thousands of widgets)

# In[3]:


# print the shape of the DataFrame
data.shape


# In[4]:


# visualize the relationship between the features and the response using scatterplots
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='sales', ax=axs[0], figsize=(16, 8), grid=True)
data.plot(kind='scatter', x='radio', y='sales', ax=axs[1], grid=True)
data.plot(kind='scatter', x='newspaper', y='sales', ax=axs[2], grid=True)


# ## Estimating ("Learning") Model Coefficients
# 
# Generally speaking, coefficients are estimated using the **least squares criterion**, which means we find the line (mathematically) which minimizes the **sum of squared residuals** (or "sum of squared errors"):

# <img src="08_estimating_coefficients.png">

# What elements are present in the diagram?
# - The black dots are the **observed values** of x and y.
# - The blue line is our **least squares line**.
# - The red lines are the **residuals**, which are the distances between the observed values and the least squares line.
# 
# How do the model coefficients relate to the least squares line?
# - $\beta_0$ is the **intercept** (the value of $y$ when $x$=0)
# - $\beta_1$ is the **slope** (the change in $y$ divided by change in $x$)
# 
# Here is a graphical depiction of those calculations:

# <img src="08_slope_intercept.png">

# ## Hands on!
# Let's create the features and class vectors (X and y)

# In[5]:


feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data.sales

X.describe()


# **Scikit-learn** provides a easy way to tran the model:

# In[6]:


logistic = LinearRegression()  # create the model
logistic.fit(X, y)  # train it


# Back to the theory! Let's see how the formula looks:

# In[7]:


for f in range(len(feature_cols)):
    print("{0} * {1} + ".format(logistic.coef_[f], feature_cols[f]))
print(logistic.intercept_)


# 
# 
# $$y = \beta_0 + \beta_1  \times TV + \beta_1  \times radio + \beta_1  \times newspaper$$
# $$y = 2.938 + 0.045 \times TV + 0.18  \times radio + -0.001  \times newspaper$$

# Let's plot the predictions and the original values:

# In[8]:


lr = LinearRegression()

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, X, y, cv=5)

# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=4)
ax.set_xlabel('Original')
ax.set_ylabel('Predicted')
plt.show()


# # Classification with Logistic Regression
# 
# |*|continuous|categorical|
# |---|---|---|
# |**supervised**|regression|**classification**|
# |**unsupervised**|dim. reduction|clustering|

# Let's go back to the Titanic dataset. We are interessed in predicting the 'survived' variable given the feature of the passenger. For the sake of simplicity, we consider only 4 features:
# 
# - pclass
# - sex
# - age
# - fare

# In[9]:


titanic_raw = pd.read_excel('titanic.xls')
titanic = titanic_raw[['pclass', 'sex', 'age', 'fare', 'survived']].dropna(axis=0, how='any')
titanic.head()


# In[10]:


dead = titanic[titanic['survived']==0]
survived = titanic[titanic['survived']==1]

print("Survived {0}, Dead {1}".format(len(dead), len(survived)))


# Specify the columns to use as features and the labels for the traning:

# In[11]:


titanic_features = ['pclass', 'sex', 'age', 'fare']
titanic_class = 'survived'


# #### Q: How is the age distribution between the two groups?

# In[12]:


fig, axes = plt.subplots(1, 2, figsize=(10, 5));

dead_age = dead[['age']]
survived_age = survived[['age']]

dead_age.plot.hist(ax=axes[0], ylim=(0, 150), title='Dead - Age')
survived_age.plot.hist(ax=axes[1], ylim=(0, 150), title='Survived - Age')


# Visible difference for young children.

# ### Let's prepare the feature vector for the training
# 
# The dataset contains categorical variable: sex (male|female)
# 
# We need to convert it in vector format. Pandas offers the method *get_dummies* that takes care of this expansion

# In[13]:


# The features vector
X = pd.get_dummies(titanic[titanic_features])
X.head()
# titanic['pclass'] = titanic['pclass'].astype('category')


# The labels used for the traning:

# In[14]:


y = titanic['survived']


# Let's create a new model...

# In[15]:


logistic = LogisticRegression(solver='lbfgs')

# for f in range(len(feature_cols)):
#     print("{0} * {1} + ".format(logistic.coef_[f], feature_cols[f]))
print(logistic)


# ... and evaluate the precison/recall with a cross validation (10 splits).
# 
# **Scikit-Learn** offers this convenient menthod to split the dataset and evaluate the performance.

# In[16]:


precision = cross_val_score(logistic, X, y, cv=10, scoring="precision")
recall = cross_val_score(logistic, X, y, cv=10, scoring="recall")

# Precision: avoid false positives
print("Precision: %0.2f (+/- %0.2f)" % (precision.mean(), precision.std() * 2))
# Recall: avoid false negatives
print("Recall: %0.2f (+/- %0.2f)" % (recall.mean(), recall.std() * 2))


# ### Explore the model output
# 
# Let's train on the full dataset

# In[17]:


logistic = LogisticRegression(solver='lbfgs')
logistic.fit(X, y)


# Given one sample, logistic regression generates the probability of belonging to the positive class. With **Scikit-Learn** we can access to this value thanks to the method *predict_proba*

# In[18]:


pred = logistic.predict_proba(X)
pred


# Of course, since we trained the whole dataset, we don't have new samples to predict, but we can predict the outcome and the relative probability for some artificial samples. Would you survive?

# In[19]:


X.columns


# In[20]:


logistic.predict([[3, 25, 200, 0, 1]])


# In[21]:


logistic.predict_proba([[3, 25, 200, 0, 1]])


# In[22]:


logistic.predict([[3, 25, 200, 1, 0]])


# In[23]:


logistic.predict_proba([[3, 25, 200, 1, 0]])


# In[ ]:




