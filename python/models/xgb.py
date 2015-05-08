__author__ = 'hujie'

import pandas as pd
import numpy as np
from XGBoostClassifier import XGBoostClassifier
import data as data
import xgboost as xgb

train_x,train_y,valid_x,valid_y=data.loadData()


xgb = XGBoostClassifier(base_estimator='gbtree',
                 objective='multi:softprob',
                 metric='mlogloss',
                 num_classes=9,
                 learning_rate=0.25,
                 max_depth=10,
                 max_samples=1.0,
                 max_features=1.0,
                 max_delta_step=0,
                 min_child_weight=4,
                 min_loss_reduction=1,
                 l1_weight=0.0,
                 l2_weight=0.0,
                 l2_on_bias=False,
                 gamma=0.02,
                 inital_bias=0.5,
                 random_state=None,
                 watchlist=None,
                 n_jobs=4,
                 n_iter=150)

xgb.fit(train_x, train_y, valid_x, valid_y)
