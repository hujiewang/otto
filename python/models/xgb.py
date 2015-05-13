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
import getopt
import sys

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
    config={
		         'base_estimator':'gbtree',
                 'objective':'multi:softprob',
                 'metric':'mlogloss',
                 'num_classes':9,
                'learning_rate':random.uniform(0.01,0.15),
                 'max_depth':13+random.randint(0,7),
                 'max_samples':random.uniform(0.5,1.0),
                 'max_features':random.uniform(0.5,1.0),
                 'max_delta_step':random.randint(1,10),
                 'min_child_weight':random.randint(1,8),
                 'min_loss_reduction':1,
                 'l1_weight':0.0,
                 'l2_weight':0.0,
                 'l2_on_bias':False,
                 'gamma':0.02,
                 'inital_bias':0.5,
                 'random_state':random_state,
                 'n_jobs':8,
                 'n_iter':20000000,

	}
    xgb = XGBoostClassifier(
                 config['base_estimator'],
                 config['objective'],
                 config['metric'],
                 config['num_classes'],
                 config['learning_rate'],
                 config['max_depth'],
                 config['max_samples'],
                 config['max_features'],
                 config['max_delta_step'],
                 config['min_child_weight'],
                 config['min_loss_reduction'],
                 config['l1_weight'],
                 config['l2_weight'],
                 config['l2_on_bias'],
                 config['gamma'],
                 config['inital_bias'],
                 config['random_state'],
                 watchlist=[[valid_x,valid_y]],
                 n_jobs=8,
                 n_iter=20000000,
                )
    print(config)
    xgb.fit(train_x, train_y)

    valid_predictions = xgb.predict_proba(valid_x)
    test_predictions= xgb.predict_proba(test_x)

    if test(valid_y,valid_predictions) <0.438:
        data.saveData(valid_predictions,"../valid_results/valid_"+str(model_id)+".csv")
        data.saveData(test_predictions,"../results/results_"+str(model_id)+".csv")
        print(config)

def main(argv):
	opts,args=getopt.getopt(argv,"n:")
	for opt,arg in opts:
		if opt=='-n':
			n=int(arg)
	print('working on '+str(n)+' to '+str(n+1000))
	for model_id in range(n,n+1000):
   		 train(model_id,train_x,train_y,valid_x,valid_y,test_x)
if __name__ == "__main__":
   main(sys.argv[1:])