import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from scipy.stats import spearmanr
from sklearn import metrics
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.linear_model import LinearRegression # OLS algorithm
from sklearn.linear_model import Ridge # Ridge algorithm
from sklearn.linear_model import Lasso # Lasso algorithm
from sklearn.linear_model import BayesianRidge # Bayesian algorithm
from sklearn.linear_model import ElasticNet # ElasticNet algorithm
from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2

#dataset

data = pd.read_csv(r"D:\MobilePhoneDataset2.csv")
test_data = pd.read_csv(r"D:\test_data2.csv")


y  = data['Price']
x = data.drop('Price', axis = 1)


#Shows the price histogram 
sns.histplot(data['Price'],kde=True)         


#Shows the all unique values

print('\nUNIQUE VALUES\n')
for col in data.columns:
    print(f'{col}: {len(data[col].unique())}\n')


#This heatmap plot denotes the highly correlated features.
#Higher number in the block has higher correlation.

corr = data.corr()                                                  
highly_corr_features = corr.index[abs(corr["Price"])>-1]           
plt.figure(figsize=(10,10))
map = sns.heatmap(data[highly_corr_features].corr(),annot=True,cmap="RdYlGn")


fig = plt.figure(figsize=(14,10))

#ROM
plt.subplot(321)
sns.scatterplot(data=data, x='ROM', y="Price")

#Colour
plt.subplot(322)
sns.scatterplot(data=data, x='Colour', y="Price")

#Guarantee
plt.subplot(323)
sns.scatterplot(data=data, x='Guarantee', y="Price")

#BatteryHealth
plt.subplot(324)
sns.scatterplot(data=data, x='BatteryHealth', y="Price")

#ScreenScratchCondition
plt.subplot(325)
sns.scatterplot(data=data, x='ScreenScratchCondition', y="Price")

#BodyScratchCondition
plt.subplot(326)
sns.scatterplot(data=data, x='BodyScratchCondition', y="Price")


#MODELING



x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 101)
    
print(x_train.shape)
print(x_valid.shape)


    

## Random Forest regression
    

rf = RandomForestRegressor(n_estimators = 100, random_state=101, oob_score = True) 
    
    
model_rf = rf.fit(x_train, y_train)
    
    
y_pred_rf = rf.predict(x_valid) 


# to see model’s score " rf.score(x_valid,y_valid) " -- 0.8249993874800947
rf.score(x_valid,y_valid)

#RMSE VALUE -- 487.8088195650574
error = sqrt(mean_squared_error(y_valid,y_pred_rf))


pd.crosstab(y_valid, y_pred_rf, rownames=['Actual Class'], colnames=['Predicted Class'])    



predicted_Price = rf.predict(test_data)
test_data['Price'] = predicted_Price



# 1. OLS

ols = LinearRegression()
ols.fit(x_train, y_train)
ols_yhat = ols.predict(x_valid)
# to see model’s score "  --0.6688804537292994
ols.score(x_valid,y_valid)
#RMSE VALUE -- 670.9996178634376
error2 = sqrt(mean_squared_error(y_valid,ols_yhat))

# 2. Ridge

ridge = Ridge(alpha = 0.5)
ridge.fit(x_train, y_train)
ridge_yhat = ridge.predict(x_valid)
# to see model’s score "  -- 0.6743673986070058
ridge.score(x_valid,y_valid)
#RMSE VALUE -- 665.4168637318677
error3 = sqrt(mean_squared_error(y_valid,ridge_yhat))


# 3. Lasso

lasso = Lasso(alpha = 0.01)
lasso.fit(x_train, y_train)
lasso_yhat = lasso.predict(x_valid)
#RMSE VALUE -- 670.9880331057335
error4 = sqrt(mean_squared_error(y_valid,lasso_yhat))
# to see model’s score "  -- 0.6688918871384962
lasso.score(x_valid,y_valid)


# 4. Bayesian

bayesian = BayesianRidge()
bayesian.fit(x_train, y_train)
bayesian_yhat = bayesian.predict(x_valid)
#RMSE VALUE -- 473.3750839666289
error5 = sqrt(mean_squared_error(y_valid,bayesian_yhat))
# to see model’s score "  -- 0.8352023315834805
bayesian.score(x_valid,y_valid)

# 5. ElasticNet

en = ElasticNet(alpha = 0.01)
en.fit(x_train, y_train)
en_yhat = en.predict(x_valid)
#RMSE VALUE -- 668.659260694884
error6 = sqrt(mean_squared_error(y_valid,en_yhat))
# to see model’s score "  -- 0.671186227214025
en.score(x_valid,y_valid)

## KNN Regression

model_knn=KNeighborsRegressor(n_neighbors=9)
model_knn.fit(x_train, y_train)
y_pred_knn = model_knn.predict(x_valid)  
#RMSE VALUE -- 654.0359265866009
error7 = sqrt(mean_squared_error(y_valid,y_pred_knn))
# to see model’s score "  -- 0.685411036375219
model_knn.score(x_valid,y_valid)


print("RMSE SCORES:")

print("-------------------------------------------------------------------------------")
print("RMSE of RANDOM FOREST model is {}".format(sqrt(mean_squared_error(y_valid,y_pred_rf))))
print("-------------------------------------------------------------------------------")
print("RMSE of OLS model is {}".format(sqrt(mean_squared_error(y_valid,ols_yhat))))
print("-------------------------------------------------------------------------------")
print("RMSE of Ridge model is {}".format(sqrt(mean_squared_error(y_valid,ridge_yhat))))
print("-------------------------------------------------------------------------------")
print("RMSE of Lasso model is {}".format(sqrt(mean_squared_error(y_valid,lasso_yhat))))
print("-------------------------------------------------------------------------------")
print("RMSE of Bayesian model is {}".format(sqrt(mean_squared_error(y_valid,bayesian_yhat))))
print("-------------------------------------------------------------------------------")
print("RMSE of ElasticNet is {}".format(sqrt(mean_squared_error(y_valid,en_yhat))))
print("-------------------------------------------------------------------------------")
print("RMSE of KNN is {}".format(sqrt(mean_squared_error(y_valid,y_pred_knn))))
print("-------------------------------------------------------------------------------")


print("\n\nMODEL'S SCORES:")

print("-------------------------------------------------------------------------------")
print("Score of RANDOM FOREST model is {}".format(rf.score(x_valid,y_valid)))
print("-------------------------------------------------------------------------------")
print("Score of OLS model is {}".format(ols.score(x_valid,y_valid)))
print("-------------------------------------------------------------------------------")
print("Score of Ridge model is {}".format(ridge.score(x_valid,y_valid)))
print("-------------------------------------------------------------------------------")
print("Score of Lasso model is {}".format(lasso.score(x_valid,y_valid)))
print("-------------------------------------------------------------------------------")
print("Score of Bayesian model is {}".format(bayesian.score(x_valid,y_valid)))
print("-------------------------------------------------------------------------------")
print("Score of ElasticNet is {}".format(en.score(x_valid,y_valid)))
print("-------------------------------------------------------------------------------")
print("Score of KNN is {}".format(model_knn.score(x_valid,y_valid)))
print("-------------------------------------------------------------------------------")