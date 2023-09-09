import inline as inline
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

matplotlib, inline

from sklearn.datasets import fetch_california_housing

california_df = fetch_california_housing()
X = pd.DataFrame(california_df.data, columns=california_df.feature_names)
y = california_df.target
print(X.head())

from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)
print(y_pred)

from sklearn.metrics import r2_score
score=r2_score(y_pred,y_test)
print(score)

parameter={
    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
    'splitter':['best','random'],
    'max-depth':[1,2,3,4,5,6,7,8,10,11,12],
    'max_features':['auto','sqrt','log2']
}
regressor=DecisionTreeRegressor()
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
regressorcv=GridSearchCV(regressor,param_grid=parameter,cv=5,scoring='neg_mean_squared_error')
regressorcv.fit(X_train,y_train)
regressorcv.best_params_
y_pred=regressorcv.predict(X_test)
r2_score(y_pred,y_test)



