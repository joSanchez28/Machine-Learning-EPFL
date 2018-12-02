
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
# - keras

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[4]:


DATA_FOLDER = './data/' #The data is in the .gitignore in order to not upload it to the GitHub repository


# Execute the following line to export from the jupyter notebook(.ipynb) to a .py file (ignore the warnings):

# In[5]:


#!jupyter nbconvert --to script Project2-Notebook.ipynb 


# ## Exploratory Data Analysis

# ### Loading phase

# First we load the data into a Pandas data frame (with the Pandas library; to install this package with conda run:
# `conda install -c anaconda pandas`):

# In[6]:


df_interior_features = pd.read_csv(DATA_FOLDER + 'T_features.csv')
df_labels_and_features = pd.read_csv(DATA_FOLDER + 'table_dataset_GreeceSwitzerland_N265_metrics_mSC5_JPEGtoBMP_Michelson_RMS.csv')


# Let's see what is inside `table_dataset_GreeceSwitzerland_N265_metrics_mSC5_JPEGtoBMP_Michelson_RMS.csv`:

# In[7]:


df_labels_and_features.head(7)


# In[8]:


df_labels_and_features.shape


# In[9]:


df_labels_and_features.columns


# The data frame has 1590 rows × 18 columns where we can see the original sample of the experiment (both inputs and outputs (xn,yn)). Each column is explained below:
# * 'ID': Identifier of the subject of the experiment (of one person).
# * 'Country': Country where the person was living.  2 x countries (Greece –138 participants, Switzerland –127 participants)
# * 'Stimulus_SkyType': 3 x sky types (clear sky with high sun angle, clear sky with low sun angle, overcast sky)
# * 'Stimulus_Context':  2 x context scenarios (social scenario, work scenario)
# * 'Gender': Male or Female
# * 'Pattern': 6 x patterns (determine how are the blinds, and so how are the shadows)
# * Outputs (yn): 'pleasant', 'interesting', 'exciting', 'calming', 'complex', 'bright', 'view', 'spacious'.
# * interior metrics (for describing the interior (the kind of room, light, ...) quantitatively): 'contrast_mean_mSC5', 'contrast_mean_Michelson', 'contrast_mean_RMS', 'complexity_mean_JPEGtoBMP'.
# 
# Note: The metrics in this data set are applied to the whole virtual reallity image (in every interior).

# Now let's see what is inside `T_features.csv`:

# In[10]:


df_interior_features.head()


# In[11]:


df_interior_features.shape


# In[12]:


df_interior_features.columns


# We have obtained a data frame with 36 rows × 31 columns. This time we see the features of each kind of interior (the kind of room, light, ...)  and several metrics for describing quantitatively the interior situation Each column is explained below:
# * 'filename': File where we have the cube map projections associated to every interior. (See in `./Cubemap_Projections/BMP/<filename>` or `./Cubemap_Projections/JPEG/<filename>`). The filenames have the structure "p(pattern_id)_(context)_(SkyType)_sg_largewin_simu".
# * 'Pattern': 6 x patterns (determine how are the blinds, and so how are the shadows)
# * 'Context': 2 x context scenarios (social scenario, work scenario)
# * 'SkyType': 3 x sky types (clear sky with high sun angle, clear sky with low sun angle, overcast sky)
# * Means and medians of the outputs (of what people have answer in their respective survey): 'mean_pleasant', 'median_pleasant', 'mean_interesting', 'median_interesting', 'mean_calming', 'median_calming', mean_exciting', 'median_exciting', 'mean_complex', 'median_complex', 'mean_bright', 'median_bright', 'mean_view', 'median_view', 'mean_spacious', 'median_spacious',
# * Interior metrics (for describing the interior quantitatively): 'contrast_mean_mSC5', 'contrast_max_mSC5', 'contrast_mean_Michelson', 'contrast_mean_Michelson_cube123', 'contrast_mean_Michelson_cube3', 'contrast_mean_RMS', 'contrast_mean_RMS_cube123','contrast_mean_RMS_cube3', 'complexity_mean_JPEGtoBMP', 'complexity_cube3_JPEGtoBMP', 'complexity_cube123_JPEGtoBMP'
# 
# Note: The metrics in this data set are applied not just to the whole virtual reallity image (we have already this data in the other data frame), but to the different parts of the whole image. Cube1 refers to what you see on your left (when you are doing the experiment with VR), cube2 refers to the front and cube3 refers to the right. Cube123 refers to the metric applied to the three subimages. For more details see`./Cubemap_Projections/JPEG/<filename>`.

# Having done this previous load and brief analysis, we are going to create the data frame we are interested in work with. Firstly we find that our inputs (or features) could be classified in two types and we are taking both of them: 
# * 1- Features of the people (country and gender) who have taken part in the experiment.
# * 2- Features of the interior: Now we are taking both categorical and metrics data.
# 
# Secondly, the labels (outputs) of our new data frame are the data that comes from every survey (which has been done by each person): 'pleasant', 'interesting', 'exciting', 'calming', 'complex', 'bright', 'view', 'spacious'. We are asked to study first the 'exciting' and the 'calming' labels but we will split the data frame afterwards.

# In[13]:


df_labels_and_features.head(1)


# In[14]:


df_interior_features.head(1)


# For doing the join, we check that one of the metrics determine totally the interior (there are not two interiors with the same metric value):

# In[15]:


df_interior_features["to_join"]=(1000*df_interior_features["contrast_mean_RMS"]).astype(int) #We cannot do the join on the float
df_labels_and_features["to_join"]=(1000*df_labels_and_features["contrast_mean_RMS"]).astype(int) #so we add another column

df_interior_features.set_index("to_join").index.is_unique


# In[16]:


df_ml = df_interior_features.merge(df_labels_and_features, on = "to_join", how = 'inner', suffixes = ("_a",""))
df_ml.head(1)


# Now we drop the duplicates columns and order the columns in order to have a data frame with the structure X|Y, where X is the matrix of features (each column is a feature) and Y is the matrix with the labels.

# In[17]:


df_ml = df_ml[['Country', 'Gender', 'Pattern', 'Context', 'SkyType', 'contrast_mean_mSC5', 'contrast_max_mSC5', 
               'contrast_mean_Michelson', 'contrast_mean_Michelson_cube123', 'contrast_mean_Michelson_cube3', 
               'contrast_mean_RMS', 'contrast_mean_RMS_cube123', 'contrast_mean_RMS_cube3', 'complexity_mean_JPEGtoBMP',
               'complexity_cube3_JPEGtoBMP', 'complexity_cube123_JPEGtoBMP', 'pleasant', 'interesting', 'exciting', 'calming', 
               'complex', 'bright', 'view', 'spacious']]


# In[18]:


df_ml.head(2)


# We can also check and see that there are no missing values. (all instances are non-null)

# In[19]:


df_ml.info()


# and see that all ratings are between 0 and 10:

# In[20]:


df_ml[['pleasant', 'interesting', 'exciting', 'calming', 'complex', 'bright',
       'view', 'spacious']].describe().loc[['min','max']]


# Finally, from this data frame is easy to get X and the labels desired (y) separately. So we get the data frame which contains the matrix of features X and, separately,the labels corresponding to 'exciting' and 'calming' (we were asked to analize this labels firstly):

# In[21]:


x_fdata = df_ml.iloc[:,0:16] #fdata -> first data without preprocessing
x_fdata.head(5)


# In[22]:


y_data = df_ml["exciting"]
y_data.to_frame().head() #Note that y_data is a Pandas serie


# ### Preprocesing Phase
# 
# From the dataframe we can see that there are 5 catgorical features:
# 1. __*Country*__ 
# 2. __*Gender*__ 
# 3. __*Pattern*__ 
# 4. __*Context*__ 
# 5. __*SkyType*__ 

# We can confirm this by checking the type of data for each feature:

# In[23]:


x_fdata.dtypes


# As we can see the five features mentionned above are the only categorical features (type object).
# 
# Let's inspect these features closely:

# In[24]:


categorical_features = ['Country','Gender','Pattern','Context',                         'SkyType']
for feat in categorical_features:
    print(feat + ':')
    print(x_fdata[feat].value_counts())


# In order to perform dummy variable encoding on the categorical features, we can use the pandas method `pd.get_dummies()`. Also since we need k-1 dummy variables to represent k categories, we can drop the first column for each encoding (`drop_first = True`). We'll store this as a new dataframe `dummy_df`.

# In[25]:


dummy_df = pd.get_dummies(x_fdata, columns=categorical_features, drop_first=False)
dummy_df.head(3)


# We standardize the features in preparation for the training (output excluded, "exciting" column).

# In[26]:


x_without_std = dummy_df.copy() #x_without_std is the final standardized data (Just in case you want to use it)
for feat in dummy_df.columns.tolist():
    mean = dummy_df[feat].mean()
    std = dummy_df[feat].std()
    dummy_df[feat] = (dummy_df[feat] - mean)/std


# In[27]:


x_data = dummy_df #x_data will be the final standardized data


# ### Data visualization

# In this section we try to visualize the possible relationship between the features and the respose variable

# First let's plot the distribution of exciting

# In[28]:


print(y_data.describe())
plt.figure(figsize=(9, 8))
sns.distplot(y_data, color='g');


# It almost looks symetric

# Now we check the correlation between the **numeric** features and 'exciting'

# In[29]:


numeric_data = pd.concat([y_data, x_data], axis=1, sort=False).drop( [ 'Country_Greece', 'Country_Switzerland',
       'Gender_Female', 'Gender_Male', 'Pattern_P1EL', 'Pattern_P2EL',
       'Pattern_P3EL', 'Pattern_P4EL', 'Pattern_P5EL', 'Pattern_P6EL',
       'Context_social', 'Context_work', 'SkyType_clearhigh',
       'SkyType_clearlow', 'SkyType_overcast'], axis = 1)



numeric_data.corr()['exciting']


# unfortunately there are no strongly correlated features (with respect to 'exciting').
# But correlation only measures linear relationship. we might get more info by plotting the data.

# 
# 

# #### plots

# We visualize the relations between "exciting"(response variable) and the features

# In[28]:


for feature in numeric_data.columns:
    sns.jointplot( numeric_data[feature], numeric_data['exciting'], kind ="scatter")


# The scatter plot obviously fails to give a good visualization since the points are overlapping.
# So we turn to density plots. 

# #### hex plot

# In[29]:


for feature in numeric_data.columns:
    sns.jointplot( numeric_data[feature], numeric_data['exciting'], kind = "hex", cmap="Reds")


# #### kernel density plot

# In[30]:


for feature in numeric_data.columns:
    sns.jointplot( numeric_data[feature], numeric_data.exciting, kind = "kdeplot", cmap="Reds", shade=True)


# As the spread of data is mostly vertical there is still no obvious relationship bewteen the numeric 
# variables and 'exciting'. 
# 
# we can also demonstrate the linear regression line estimate for each feature.

# In[114]:


fig, ax = plt.subplots(round(len(numeric_data.columns) / 3), 3, figsize = (18, 12))

for i, ax in enumerate(fig.axes):
        sns.regplot(x=numeric_data.columns[i],y='exciting', data=numeric_data, ax=ax)


# Now lets plot box plots for categorical features to see the relashionship between them and exciting

# In[115]:


# we need the categorical variables in their originaL(non dummy) form
categ_data = x_fdata[categorical_features]



# In[116]:


fig, ax = plt.subplots(round(len(categ_data.columns) / 3), 3, figsize = (18, 12))

for i, ax in enumerate(fig.axes):
        if (i < len(categ_data.columns)):
            sns.boxplot(x=categ_data.iloc[:,i], y=y_data, ax=ax)


# we can also look at the distributions of categorical variabels

# In[117]:


fig, axes = plt.subplots(round(len(categ_data.columns) / 3), 3, figsize=(12, 5))

for i, ax in enumerate(fig.axes):
    if i < len(categ_data.columns):
        sns.countplot(x=categ_data.columns[i], data=categ_data, ax=ax)

fig.tight_layout()


# we see that all categories almost have a unifrom distribution. so all of them have the possibility of helping us predict the outcome.

# Now lets take a look at the correlation between the **numeric** features

# In[118]:


# remove the output variable
corr = numeric_data.drop('exciting', axis=1).corr() 

plt.figure(figsize=(11, 11))

sns.heatmap(corr[(corr >= 0.7) | (corr <= -0.7)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# This tells us that a lot of our measures are highly correlated. 
# So if one of them is not useful there might be no reason to use the other one either. 
# And maybe we can also reduce our features.

# #### Splitting the data

# In order to split the data into the two sets (training and test), first we need to rejoin the y_data and the x_data. Then we can use the sample method (`pd.sample()`) which  randomly samples from the dataframe according to a given ratio (0.8 for train in this case). The remaining part of the data will be the test set.

# In[119]:


#In order to split the data, we join then again:
data = pd.concat([y_data, x_data], axis=1, sort=False)
data.head(1)


# In[120]:


train_df = data.sample(frac=0.8,replace=False)
test_df = data.drop(train_df.index.tolist(),axis=0)


# In[121]:


train_df.head(3)


# In[122]:


test_df.head(3)


# Quick check of the validity of the split by making sure that the size of the train plus the size of the test equals the size of the dummy:

# In[123]:


print(train_df.shape[0] + test_df.shape[0] == dummy_df.shape[0])


# Double check the mean of the features

# In[124]:


#dummy_df.mean()


# The means aren't exactly 0 but they are small enough for us to approximate them to 0.

# Now for the standart deviation

# In[125]:


#dummy_df.std()


# Hence our standardized features and our labels are ready to go.

# In[ ]:


x_train = train_df.iloc[1:-1]
y_train = train_df.iloc[0:1]
x_test = train_df.iloc[1:-1]
y_test = train_df.iloc[0:1]
#x_data and y_data are the preprocessed data without splitting. Maybe we can use them considering that we have very few data


# ## Exploring different approachs

# Having done that, we can start trying differents methods for obtaining predictions about the excitation/calm of a person who do the experiment.

# ## Linear Regression 

# In[32]:


from sklearn import linear_model


# In[33]:


reg = LinearRegression()
reg.fit(x_data, y_data)


# In[34]:


reg.coef_


# In[35]:


reg.score(x_data, y_data)


# according to the docs score is close to 0 when the model predicts a constant close to the expected value.
# 
# Lets also calculate accuracy since our problem can also be seen as classification

# In[36]:


def compute_accuracy(model, x_data, y_true):
    
    # predict and round to closest digit
    preds = model.predict(x_data)
    preds = np.rint(preds)
    # round less than 0 and more than 10 to themselves
    preds[preds<0] = 0
    preds[preds>10] = 10
    
    accurate = len(preds[preds == y_true])
    return accurate/len(preds)

print(compute_accuracy(reg, x_data, y_data))


# This is a very low accuracy. Let's try polynomial features as well.

# In[37]:


from sklearn.preprocessing import PolynomialFeatures


# In[38]:


for i in range(1,5):
    poly = PolynomialFeatures(degree=i)
    poly_x = poly.fit_transform(x_data)

    reg = LinearRegression()
    reg.fit(poly_x, y_data)
    
    print("for degree=%d"%i)

    accr = compute_accuracy(reg, poly_x, y_data)
    print("accuracy = %f"%(accr))
    
    print("score = %f \n" %(reg.score(poly_x, y_data)) )


# ## Random Forest

# In[39]:


from sklearn.ensemble import RandomForestClassifier


# RandomForestClassifier(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)

# Firstly, we do grid search on the number of estimators "n_estimators" and the max depth of the trees "max_depth". We decide to use the accuracy as a criteria for comparing the quality of the diferents models (created by differents "n_estimators" and "max_depth"). So, we compute a data frame with the accuracy for every pair (n_estimators, max_depth) explorated.

# In[46]:


n_estimators_range=range(1,100, 5)
max_depth_range= range(1,100, 5)


# In[ ]:


performance_results = pd.DataFrame(columns=['n_estimators', 'max_depth','accuracy', 'mean_error_scorer'])

index = 0
for i in n_estimators_range:
    for j in max_depth_range:
        forestC = RandomForestClassifier(n_estimators=i, max_depth=j)
        accuracy = cross_val_score(forestC, x_data, y_data, cv = 5, scoring='accuracy').mean()
        mean_error_scorer = cross_val_score(forestC, x_data, y_data, cv = 5, scoring='neg_mean_absolute_error').mean()
        
        performance_results.loc[index] = [i, j, accuracy, mean_error_scorer]  
        index = index + 1

performance_results.head()


# In[ ]:


performance_results.sort_values(by = "accuracy", ascending=False).head(2)


# In[ ]:


performance_results.sort_values(by = "mean_error_scorer", ascending=False).head(3)


# ## Decision Trees

# In[100]:


from sklearn import tree

depths_DTR = np.arange(1,15,1); # deepness of the tree
y_DTR = np.zeros((len(depths_DTR), len(y_data)));
mse_DTR = np.zeros(len(depths_DTR));
mse_DTR_cross = np.zeros(len(depths_DTR));

h=0
while(h<len(depths_DTR)):
    DTR = tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=depths_DTR[h]) # min_weight_fraction_leaf?
    DTR = DTR.fit(x_data, y_data)

    #Using everything
    y_DTR[h, :] = DTR.predict(x_data)
    mse_DTR[h] = sum((y_DTR[h, :]-y_data.values)**2)/(len(y_data))
    
    #Using everything with cross validation
    mse_DTR_cross[h] = cross_val_score(DTR, x_data, y_data, cv = 10, scoring='neg_mean_squared_error').mean()
    
    h=h+1

print(mse_DTR)
print(mse_DTR_cross)


# ## DENSE NEURAL NETWORKS

# In[62]:


# Use conda install -c conda-forge keras; conda install -c anaconda keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
np.random.seed(7)

embed_dim = 128
lstm_out = 196
max_features = len(x_data.values[1, :])

def rnn_model():
    #Create Model: interpreted as a sequence of layers https://keras.io/models/model/
    model = Sequential() 
    model.add(Embedding(max_features, embed_dim, input_length=len(x_data.values[0, :])))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)) # randomnly discard 0.2 for each node?
    model.add(Dense(1, activation='softmax'))
    
    #Compile Model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy']) # loss: categorical_crossentropy (possible?? have to vectorize "exiciting"?)  ;adam: efficient gradient descent
    return(model)

model = rnn_model()
model.summary
print(model.summary())
model.fit(x_data, y_data, epochs=15, batch_size=32)
    
    

