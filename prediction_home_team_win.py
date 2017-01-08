import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing,model_selection,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

df= pd.read_csv('modified_final_analysis.csv')

df= df[['st_kpp_home','st_kpp_away','goals_kpp_home','goals_kpp_away','corner_kpp_home','corner_kpp_away','modified_result']]


X= np.array(df[['st_kpp_home','st_kpp_away','goals_kpp_home','goals_kpp_away','corner_kpp_home','corner_kpp_away']])
y= np.array(df['modified_result'])

models=[]

models.append(('svm',svm.SVC()))
models.append(('LogReg',LogisticRegression()))
models.append(('DTC',DecisionTreeClassifier()))

names=[]
results=[]

for name, model in models:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results_accuracy = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
	cv_results_roc_auc = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
	msg1 = "%s: mean_accuracy:%f (%f)" % (name, cv_results_accuracy.mean(), cv_results_accuracy.std())
	msg2 = "%s: mean_roc_auc:%f (%f)" % (name, cv_results_roc_auc.mean(), cv_results_roc_auc.std())
	print(msg1)
	print(msg2)


ensembles=[]

ensembles.append(('AB',AdaBoostClassifier()))
ensembles.append(('GBM',GradientBoostingClassifier()))
ensembles.append(('RF',RandomForestClassifier()))
ensembles.append(('ET',ExtraTreesClassifier()))

for name, model in ensembles:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results_accuracy = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
	cv_results_roc_auc = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
	msg1 = "%s: mean_accuracy:%f (%f)" % (name, cv_results_accuracy.mean(), cv_results_accuracy.std())
	msg2 = "%s: mean_roc_auc:%f (%f)" % (name, cv_results_roc_auc.mean(), cv_results_roc_auc.std())
	print(msg1)
	print(msg2)

#Tuning svm

C_range = [0.001,0.01,0.1,1,10]
gamma_range = [i for i in range(0,11)]
kernels= ['rbf','linear','rbf']

param_grid = dict(gamma=gamma_range, C=C_range, kernel=kernels)

kfold= KFold(n_splits=10,random_state=7)

grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=kfold)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#The best parameters are {'kernel': 'rbf', 'gamma': 2, 'C': 1} with a score of 0.62


#After rescaling

scale= StandardScaler()

X_rescaled= scale.fit_transform(X)


C_range = [0.001,0.01,0.1,1,10]
gamma_range = [i for i in range(0,11)]
kernels= ['rbf','linear','rbf']

param_grid = dict(gamma=gamma_range, C=C_range, kernel=kernels)

kfold= KFold(n_splits=10,random_state=7)

grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=kfold)
grid.fit(X_rescaled, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#The best parameters are {'kernel': 'linear', 'gamma': 0, 'C': 0.1} with a score of 0.60


