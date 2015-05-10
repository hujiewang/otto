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

    xgb = XGBoostClassifier(base_estimator='gbtree',
                 objective='multi:softprob',
                 metric='mlogloss',
                 num_classes=9,
                 learning_rate=0.05,
                 max_depth=16,
                 max_samples=0.7,
                 max_features=0.7,
                 max_delta_step=0,
                 min_child_weight=4,
                 min_loss_reduction=1,
                 l1_weight=0.0,
                 l2_weight=0.0,
                 l2_on_bias=False,
                 gamma=0.02,
                 inital_bias=0.5,
                 random_state=random_state,
                 watchlist=[[valid_x,valid_y]],
                 n_jobs=8,
                 n_iter=2000,
                )

    xgb.fit(train_x, train_y)

    valid_predictions = xgb.predict_proba(valid_x)
    test_predictions= xgb.predict_proba(test_x)

    if test(valid_y,valid_predictions) <0.450:
        data.saveData(valid_predictions,"../valid_results/valid_"+str(model_id)+".csv")
        data.saveData(test_predictions,"../results/results_"+str(model_id)+".csv")

for model_id in range(500,600):
    train(model_id,train_x,train_y,valid_x,valid_y,test_x)