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

df= df[['st_kpp_home','st_kpp_away','goals_kpp_home','goals_kpp_away','corner_kpp_home','corner_kpp_away','final_result']]

#taking final result (includes draws)

X= np.array(df[['st_kpp_home','st_kpp_away','goals_kpp_home','goals_kpp_away','corner_kpp_home','corner_kpp_away']])
y= np.array(df['final_result'])


models=[]

models.append(('svm',svm.SVC()))
models.append(('LogReg',LogisticRegression()))
models.append(('DTC',DecisionTreeClassifier()))

names=[]
results=[]

for name, model in models:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results_accuracy = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
	#cv_results_roc_auc = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
	msg1 = "%s: mean_accuracy:%f (%f)" % (name, cv_results_accuracy.mean(), cv_results_accuracy.std())
	#msg2 = "%s: mean_roc_auc:%f (%f)" % (name, cv_results_roc_auc.mean(), cv_results_roc_auc.std())
	print(msg1)
	#print(msg2)


ensembles=[]

ensembles.append(('AB',AdaBoostClassifier()))
ensembles.append(('GBM',GradientBoostingClassifier()))
ensembles.append(('RF',RandomForestClassifier()))
ensembles.append(('ET',ExtraTreesClassifier()))

for name, model in ensembles:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results_accuracy = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
	#cv_results_roc_auc = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
	msg1 = "%s: mean_accuracy:%f (%f)" % (name, cv_results_accuracy.mean(), cv_results_accuracy.std())
	#msg2 = "%s: mean_roc_auc:%f (%f)" % (name, cv_results_roc_auc.mean(), cv_results_roc_auc.std())
	print(msg1)
	#print(msg2)

"""
accuracies when draws are included:

svm: mean_accuracy:0.452489 (0.071713)
LogReg: mean_accuracy:0.470910 (0.028119)
DTC: mean_accuracy:0.426245 (0.057103)
AB: mean_accuracy:0.365292 (0.063810)
GBM: mean_accuracy:0.418137 (0.050752)
RF: mean_accuracy:0.386415 (0.080198)
ET: mean_accuracy:0.407397 (0.067838)

"""

