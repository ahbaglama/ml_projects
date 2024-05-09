

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
from sklearn.preprocessing import StandardScaler



#model_selection
from sklearn.model_selection import train_test_split,StratifiedKFold,RepeatedStratifiedKFold,cross_val_score




#metrics
from sklearn.metrics import r2_score,explained_variance_score,median_absolute_error,mean_absolute_error





#models
from sklearn import svm, tree, linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor



#Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel


import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("dataset.csv", index_col=0)
X=df.values[:,0:-1]
Y=df.values[:,-1]
df.pop("price")


#Feature Selection
mut_info = SelectKBest(mutual_info_regression, k=3).fit_transform(X, Y)
f_reg = SelectKBest(f_regression, k=3).fit_transform(X, Y)
var_threshold = VarianceThreshold(threshold=(.7 * (1 - .7))).fit_transform(X)
sel_model = SelectFromModel(estimator=tree.DecisionTreeRegressor(max_depth=4)).fit(X, Y)
sel_model=sel_model.transform(X)



#Feature Extraction
scaler=StandardScaler()
scaled_data=scaler.fit_transform(df)
pca=PCA(n_components=2)
x_pca=pca.fit_transform(scaled_data)


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=15)


def get_score(metric,model,X_train,X_test,Y_train,Y_test):
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    Y_pred2=model.predict(X_train)
    if metric=="r2":return [r2_score(Y_test,Y_pred),r2_score(Y_train,Y_pred2)]
    elif metric=="explained_variance":return [explained_variance_score(Y_test,Y_pred),explained_variance_score(Y_train,Y_pred2)]
    elif metric=="neg_mean_absolute_error":return [mean_absolute_error(Y_test,Y_pred),mean_absolute_error(Y_train,Y_pred2)]
    elif metric=="neg_median_absolute_error":return [median_absolute_error(Y_test,Y_pred),median_absolute_error(Y_train,Y_pred2)]



n=3
def run_model(model):
    #for m_selection in [KFold(n_splits=n),StratifiedKFold(n_splits=n),TimeSeriesSplit(n_splits=n),RepeatedStratifiedKFold(n_splits=n, n_repeats=4),ShuffleSplit(n_splits=3, test_size=0.33)]:
    print("\n","="*100,"\n\nRepeated Stratified KFold\n")
    for metric in ["r2","neg_mean_absolute_error","neg_median_absolute_error","explained_variance"]:
        print(metric,":",cross_val_score(model,X_train,Y_train,cv=RepeatedStratifiedKFold(n_splits=n, n_repeats=10),scoring=metric),"// Average :",cross_val_score(model,X_train,Y_train,cv=RepeatedStratifiedKFold(n_splits=n, n_repeats=4),scoring=metric).mean())
        print("Test Score: ",get_score(metric,model,X_train,X_test,Y_train,Y_test)[0],"// Train Score",get_score(metric,model,X_train,X_test,Y_train,Y_test)[1],"\n")



model_list=[0,linear_model.LinearRegression(),svm.SVR(C=65000),tree.DecisionTreeRegressor(max_depth=4),KNeighborsRegressor(n_neighbors=4),PLSRegression(n_components=2),RandomForestRegressor()]
model_index=1
compare_list=[]
while True:
    try:
        model_index=int(input("Select a model :\n \nLinear Regression=1\nSupport Vector Regression=2\nDecision Tree Regression=3\nKNeighbors Regression=4\nPLS Regression=5\nRandom Forest Regression=6\nExit=0\n"))
        model=model_list[model_index]
        if model==0:break
        run_model(model)
        time.sleep(0.3)
    except:
        print("\nPlease enter a valid number\n")
        continue
print(compare_list)


#experiments

from sklearn.model_selection import GridSearchCV

clf=GridSearchCV(svm.SVR(),{
    "C":[x for x in range(0,100000,5000)],
},cv=RepeatedStratifiedKFold(n_splits=3,n_repeats=10),return_train_score=False,scoring="neg_mean_absolute_error")

clf.fit(X_train,Y_train)

df3=pd.DataFrame(clf.cv_results_)

df3[["params","mean_test_score"]]



