__author__ = 'hujie'

from sklearn.svm import SVC
import pandas as pd
import numpy as np
import data as data
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from test import test
train_x,train_y,valid_x,valid_y,test_x=data.loadData()
train_x,train_y=shuffle(train_x,train_y)

param_grid = [
 {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
 {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
svr = SVC()
gs = GridSearchCV(svr,param_grid,n_jobs=8,verbose=2)
gs.fit(train_x, train_y)

valid_predictions = gs.predict_proba(valid_x)
test_predictions= gs.predict_proba(test_x)

test(valid_y,valid_predictions)


data.saveData(valid_predictions,"../valid_results/valid_215.csv")
data.saveData(test_predictions,"../results/results_215.csv")
