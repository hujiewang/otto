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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from test import test

random.seed()
train_x,train_y,valid_x,valid_y,test_x=data.loadData()
# normalization
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
valid_x = scaler.transform(valid_x)
test_x = scaler.transform(test_x)

def trainrf(model_id,train_x,train_y,valid_x,valid_y,test_x):
    train_x,train_y=shuffle(train_x,train_y)


    random_state=random.randint(0, 1000000)
    print('random state: {state}'.format(state=random_state))

    clf = RandomForestClassifier(n_estimators=random.randint(50,5000),
                                 criterion='gini',
                                 max_depth=random.randint(10,1000),
                                 min_samples_split=random.randint(2,50),
                                 min_samples_leaf=random.randint(1,10),
                                 min_weight_fraction_leaf=random.uniform(0.0,0.5),
                                 max_features=random.uniform(0.1,1.0),
                                 max_leaf_nodes=random.randint(1,10),
                                 bootstrap=False,
                                 oob_score=False,
                                 n_jobs=30,
                                 random_state=random_state,
                                 verbose=0,
                                 warm_start=True,
                                 class_weight=None
                )

    clf.fit(train_x, train_y)

    valid_predictions1 = clf.predict_proba(valid_x)
    test_predictions1= clf.predict_proba(test_x)

    t1 = test(valid_y,valid_predictions1)

    ccv = CalibratedClassifierCV(base_estimator=clf,method="sigmoid",cv='prefit')
    ccv.fit(valid_x,valid_y)

    valid_predictions2 = ccv.predict_proba(valid_x)
    test_predictions2= ccv.predict_proba(test_x)

    t2 = test(valid_y,valid_predictions2)

    if t2<t1:
        valid_predictions=valid_predictions2
        test_predictions=test_predictions2
        t=t2
    else:
        valid_predictions=valid_predictions1
        test_predictions=test_predictions1
        t=t1

    if t < 0.450:
        data.saveData(valid_predictions,"../valid_results/valid_"+str(model_id)+".csv")
        data.saveData(test_predictions,"../results/results_"+str(model_id)+".csv")


def trainxgb(model_id,train_x,train_y,valid_x,valid_y,test_x):
    train_x,train_y=shuffle(train_x,train_y)

    random_state=random.randint(0, 1000000)
    print('random state: {state}'.format(state=random_state))

    xgb = XGBoostClassifier(base_estimator='gbtree',
                 objective='multi:softprob',
                 metric='mlogloss',
                 num_classes=9,
                 learning_rate=random.uniform(0.01,0.05),
                 max_depth=random.randint(10,20),
                 max_samples=random.uniform(0.0,1.0),
                 max_features=random.uniform(0.0,1.0),
                 max_delta_step=random.randint(1,10),
                 min_child_weight=random.randint(1,10),
                 min_loss_reduction=1,
                 l1_weight=0.0,
                 l2_weight=0.0,
                 l2_on_bias=False,
                 gamma=0.02,
                 inital_bias=random.uniform(0.0,1.0),
                 random_state=random_state,
                 watchlist=[[valid_x,valid_y]],
                 n_jobs=30,
                 n_iter=3000,
                )

    xgb.fit(train_x, train_y)

    valid_predictions = xgb.predict_proba(valid_x)

    if test(valid_y,valid_predictions) <0.450:
        test_predictions= xgb.predict_proba(test_x)
        data.saveData(valid_predictions,"../valid_results/valid_"+str(model_id)+".csv")
        data.saveData(test_predictions,"../results/results_"+str(model_id)+".csv")

for model_id in range(215,1000000):
    if model_id % 2 == 0:
        trainxgb(model_id,train_x,train_y,valid_x,valid_y,test_x)
    else:
        trainrf(model_id,train_x,train_y,valid_x,valid_y,test_x)