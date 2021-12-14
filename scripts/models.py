#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
#%%
X = pd.read_pickle('features.pkl')
y = pd.read_pickle('target.pkl')
#%%
y = y['outcome']
# %%
#splitting into test and train data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=444)
X_train
#%%
# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter = 5000, 
                            solver = 'saga', 
                            penalty = 'l2',
                            class_weight = {1:40})
logreg.fit(X_train, y_train)

test_log = logreg.predict_proba(X_test)
auc = metrics.roc_auc_score(y_test, test_log[:,1])
y_pred = logreg.predict(X_test)
# %%
print("Recall Score:", metrics.recall_score(y_test,y_pred))
print("Logistic Regression AUC:", auc)
print('\n')
print('Train Accuracy:', logreg.score(X_train, y_train))
print('Test Accuracy:', logreg.score(X_test, y_test))
# %%





#%%
# Support Vector Machine
m = min(y_train)
M = max(y_train)
y_train2 = 2*(y_train-(m+M)/2)/(M-m)

m = min(y_test)
M = max(y_test)
y_test2 = 2*(y_test-(m+M)/2)/(M-m)
# %%
from sklearn import svm
model = svm.SVC(kernel = 'linear', class_weight = {1:27})
model.fit(X_train, y_train2)
y_pred5 = model.predict(X_test)

test_svm = model.decision_function(X_test)
aucsvm = metrics.roc_auc_score(y_test2, test_svm)
# %%
print("Recall:",metrics.recall_score(y_test2, y_pred5))
print("Support Vector Machine AUC:", aucsvm)
print('\n')
print('SVM Train Accuracy: ', model.score(X_train, y_train2))
print('SVM Test Accuracy: ', model.score(X_test, y_test2))
# %%




#%%
# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state = 4444,
                                 learning_rate = 1,
                                 n_estimators = 500,
                                 subsample = .25,
                                 max_features = 'auto')
gbc.fit(X_train,y_train)

test_logbc = gbc.predict_proba(X_test)
aucgbc = metrics.roc_auc_score(y_test, test_logbc[:,1])
y_predgbc = gbc.predict(X_test)
# %%
print("Recall:",metrics.recall_score(y_test, y_predgbc))
print("AUC:", aucgbc)
print('\n')
print('Train Accuracy: ', gbc.score(X_train, y_train))
print('Test Accuracy: ', gbc.score(X_test, y_test))
# %%
