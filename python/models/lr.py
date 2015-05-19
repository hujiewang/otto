__author__ = 'hujie'


import pandas as pd
import numpy as np
from XGBoostClassifier import XGBoostClassifier
import data as data
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import random
from sklearn.calibration import CalibratedClassifierCV
from test import test
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
random.seed()
train_x,train_y,valid_x,valid_y,test_x=data.loadData()

def train(model_id,train_x,train_y,valid_x,valid_y,test_x):
    train_x,train_y=shuffle(train_x,train_y)

    # normalization
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    valid_x = scaler.transform(valid_x)
    test_x = scaler.transform(test_x)

    random_state=random.randint(0, 1000000)
    print('random state: {state}'.format(state=random_state))

    clf = LogisticRegression(penalty='l2',
                             dual=False,
                             tol=0.0001,
                             C=1.0,
                             fit_intercept=True,
                             intercept_scaling=1,
                             class_weight=None,
                             random_state=None,
                             solver='lbfgs',
                             max_iter=1000,
                             multi_class='ovr',
                             verbose=True
                             )

    clf.fit(train_x, train_y)

    valid_predictions = clf.predict_proba(valid_x)
    test(valid_y,valid_predictions)

    ccv = CalibratedClassifierCV(base_estimator=clf,method="sigmoid",cv='prefit')
    ccv.fit(train_x,train_y)

    valid_predictions = ccv.predict_proba(valid_x)
    test(valid_y,valid_predictions)

    test_predictions= ccv.predict_proba(test_x)

    data.saveData(valid_predictions,"../valid_results/valid_"+str(model_id)+".csv")
    data.saveData(test_predictions,"../results/results_"+str(model_id)+".csv")

for model_id in range(216,217):
    train(model_id,train_x,train_y,valid_x,valid_y,test_x)