#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


# reading the input file

housing = pd.read_csv("train.csv")

housing.head()


# In[3]:


housing.shape


# In[4]:


housing.info()


# In[5]:


# count the number of null values present in the data

round(housing.isnull().sum()/len(housing.index)*100,2)

### Certain columns have more than 85% missing values. Hence, it would be better to remove them from our consideration. These include Alley, PoolQC and Miscfeature  
# 

# In[6]:


housing.drop(['Alley','PoolQC','MiscFeature'],axis=1,inplace = True)

housing.head()


# In[7]:


# count the number of null values present in the data

round(housing.isnull().sum()/len(housing.index)*100,2)


# In[8]:


# We shall now create a Correlation Map to see how features are correlated with SalePrice
corrmat = housing.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# ### From the pairplot given above we can see that some of the numerical colums are highly related with the sales price
# 

# #### - LotFrontage
# #### - Overall Quality
# #### - Year Built
# #### - Year removeadd
# #### - MasVnrArea
# #### - TotalBsmn SF
# #### - 1st Foor SF
# #### - Gr ving Area
# #### - Fullbath
# #### - Fireplaces
# #### - Garage Area

# ### We shall be keeping them in our consideration while building our model. 

# ### Let us also check the most corelated values present in the data from the nearly 80 variables, beginning with the Top 10.

# In[9]:


k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(housing[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# ## We can see from the above chart that the top 10 variables are similar to what we had initially deciphered

# ##### OverallQual - It is obvious that having a good quality house would fetch bigger price
# ##### Grlivarea - More the area of the floor above the ground floor, greater is the price
# ##### Garage cars, garage area, total basement SF, 1st floor SF -Seems to make sense

# ## Let us now look for patterns in the data by visualising the relationships between these variables
# 

# In[10]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(housing[cols],height = 2)
plt.show()


# ### Now let us see what % of of data is missing (apart from the 3 we have already removed) and how can we treat them

# In[11]:


total = housing.isnull().sum().sort_values(ascending=False)
percent = (housing.isnull().sum()/housing.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# ### Since Fence' has nearly  ~81% of the data missing, we would recommend dropping it.

# In[13]:


housing['Fence'].value_counts()


# In[14]:


ousing.drop(['Fence'],axis=1,inplace = True)


# In[15]:


housing.shape


# In[17]:


housing['FireplaceQu'].value_counts()/len(housing['FireplaceQu'])*100


# In[18]:


housing['LotFrontage'].value_counts()/len(housing['LotFrontage'])*100


# In[19]:


housing['GarageCond'].value_counts()/len(housing['GarageCond'])*100


# In[20]:


housing['GarageType'].value_counts()/len(housing['GarageType'])*100


# In[21]:


housing['GarageYrBlt'].value_counts()/len(housing['GarageYrBlt'])*100


# In[22]:


housing['GarageFinish'].value_counts()/len(housing['GarageFinish'])*100


# In[23]:


housing['GarageQual'].value_counts()/len(housing['GarageQual'])*100


# In[24]:


housing['BsmtExposure'].value_counts()/len(housing['BsmtExposure'])*100


# In[25]:


housing['BsmtFinType2'].value_counts()/len(housing['BsmtFinType2'])*100


# In[26]:


housing['BsmtFinType1'].value_counts()/len(housing['BsmtFinType1'])*100


# In[27]:


housing['BsmtCond'].value_counts()/len(housing['BsmtCond'])*100


# In[28]:


housing['BsmtQual'].value_counts()/len(housing['BsmtQual'])*100


# In[29]:


housing['MasVnrArea'].value_counts()/len(housing['MasVnrArea'])*100


# In[30]:


housing['MasVnrType'].value_counts()/len(housing['MasVnrType'])*100


# ### From the calculations above we can see that some of the columns can be removed directly as they are not adding any variance to our data and hence are not at all useful.

#     #### - BsmtCond
#     #### - BsmtFinType2
#     #### - BsmtExposure
#     #### - GarageQual
#     #### - GarageYrBlt
#     #### - GarageCond
#     #### - MasVnrArea

# In[31]:


housing.drop(['BsmtCond','BsmtFinType2','BsmtExposure','GarageQual','GarageYrBlt','GarageCond','MasVnrArea'],axis=1,inplace = True)


# In[32]:


housing.shape


# In[33]:


qualitative = [f for f in housing.columns if housing.dtypes[f] == 'object']


# In[34]:


for c in qualitative:
    housing[c] = housing[c].astype('category')
    if housing[c].isnull().any():
        housing[c] = housing[c].cat.add_categories(['MISSING'])
        housing[c] = housing[c].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(housing, id_vars=['SalePrice'], value_vars=qualitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, height=5)
g = g.map(boxplot, "value", "SalePrice")


# ### - Most of these variables have a diverse relationship with the 'Sales' and we will try to define some of them below -

# In[35]:


def anova(frame):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

a = anova(housing)
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)


# ### - From the above chart we can see that some of the majorly influencing variables are

# ##### - Neighbourhood
# ##### - ExterQuality
# ##### - Basement Quality
# ##### - KitchenQuality

# ### Now we will create dummy variables for the variables present in the current data set

# In[36]:


housing.shape


# In[37]:


housing.info()


# In[38]:


housing['LotFrontage'].fillna('0', inplace = True)

housing.head()


# ### Preparing the data for modelling

# In[39]:


housing_1 = housing

housing_1.head()


# In[40]:


y = housing_1['SalePrice']


# In[41]:


housing_1.drop(['SalePrice'],axis=1,inplace=True)


# In[42]:


housing_1.head()


# In[43]:


housing_1.info()


# In[44]:


housing_categorical = housing_1.select_dtypes(include=['category'])
housing_categorical.head()


# In[45]:


housing_dummies = pd.get_dummies(housing_categorical)
housing_dummies.head()


# In[46]:


# Dropping all categorical variables

housing_1 = housing_1.drop(list(housing_categorical.columns),axis=1)


# In[47]:


housing_1.head()


# In[48]:


# concat dummy variables with housing dataset
housing_1 = pd.concat([housing_1, housing_dummies], axis=1)


# In[49]:


housing_1.head()


# In[50]:


# Scaling the features

from sklearn.preprocessing import scale

cols = housing_1.columns
housing_1 = pd.DataFrame(scale(housing_1))
housing_1.columns = cols
housing_1.columns


# In[51]:


# Split into test and train data set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(housing_1, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)


# In[52]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[53]:


# list of alphas to tune
params = {'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.003,0,20]}

ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train) 


# In[54]:


cv_results = pd.DataFrame(model_cv.cv_results_)
#cv_results = cv_results[cv_results['param_alpha']<=200]
cv_results.head()


# In[55]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


# In[56]:


alpha = 20
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)


# In[57]:


from sklearn.metrics import r2_score


# In[58]:


y_train_pred=ridge.predict(X_train)
r2_score(y_train,y_train_pred)


# In[59]:


alpha = 20
ridge = Ridge(alpha=alpha)

ridge.fit(X_test, y_test)

y_test_pred=ridge.predict(X_test)
r2_score(y_test,y_test_pred)


# ### Checking values for different values of alpha (10, 5, 1, 0.01, 0.03, 0.05)

# In[60]:


alpha = 10
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)

y_train_pred=ridge.predict(X_train)
r2_score(y_train,y_train_pred)


# In[61]:


alpha = 10
ridge = Ridge(alpha=alpha)

ridge.fit(X_test, y_test)

y_test_pred=ridge.predict(X_test)
r2_score(y_test,y_test_pred)


# In[62]:


alpha = 5
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)

y_train_pred=ridge.predict(X_train)
r2_score(y_train,y_train_pred)


# In[63]:


alpha = 5
ridge = Ridge(alpha=alpha)

ridge.fit(X_test, y_test)

y_test_pred=ridge.predict(X_test)
r2_score(y_test,y_test_pred)


# In[64]:


alpha = 1
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)

y_train_pred=ridge.predict(X_train)
r2_score(y_train,y_train_pred)


# In[65]:


alpha = 1
ridge = Ridge(alpha=alpha)
ridge.fit(X_test, y_test)

y_test_pred=ridge.predict(X_test)
r2_score(y_test,y_test_pred)


# ### So we observe that with Ridge regression at different levels of alpha, the R2 results are good

# ### Now we will run the model using Lasso regression model

# In[66]:


lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = 5, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train)


# In[67]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()


# In[68]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


# In[69]:


alpha =20

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train, y_train)


# In[70]:


alpha =20

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train, y_train) 


# In[71]:


y_train_pred=lasso.predict(X_train)
r2_score(y_train,y_train_pred)


# ### The Lasso Regression too throws up a good R2 score

# #### Hence we see that with both lasso and Ridge regression we are able to get test and train score. We can use either for the final model and would be going ahead with the Lasso model

# In[ ]:




