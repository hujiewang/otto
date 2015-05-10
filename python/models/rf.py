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

    clf = RandomForestClassifier(n_estimators=500,
                                 criterion='gini',
                                 max_depth=None,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.0,
                                 max_features='auto',
                                 max_leaf_nodes=None,
                                 bootstrap=True,
                                 oob_score=False,
                                 n_jobs=16,
                                 random_state=None,
                                 verbose=0,
                                 warm_start=True,
                                 class_weight=None
                )

    clf.fit(train_x, train_y)

    valid_predictions = clf.predict_proba(valid_x)
    test(valid_y,valid_predictions)

    ccv = CalibratedClassifierCV(base_estimator=clf,method="sigmoid",cv="prefit")
    ccv.fit(valid_x,valid_y)

    valid_predictions = ccv.predict_proba(valid_x)
    test(valid_y,valid_predictions)

    test_predictions= ccv.predict_proba(test_x)

    data.saveData(valid_predictions,"../valid_results/valid_"+str(model_id)+".csv")
    data.saveData(test_predictions,"../results/results_"+str(model_id)+".csv")

for model_id in range(215,220):
    train(model_id,train_x,train_y,valid_x,valid_y,test_x)