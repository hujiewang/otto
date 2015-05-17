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
import getopt
import sys

random.seed()
train_x,train_y,valid_x,valid_y,test_x=data.loadData()
# normalization
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
valid_x = scaler.transform(valid_x)
test_x = scaler.transform(test_x)

def train(model_id,train_x,train_y,valid_x,valid_y,test_x):
    train_x,train_y=shuffle(train_x,train_y)


    random_state=random.randint(0, 1000000)
    print('random state: {state}'.format(state=random_state))

    clf = RandomForestClassifier(bootstrap=False, class_weight=None,
            criterion='entropy', max_depth=29008, max_features=36,
            max_leaf_nodes=None, min_samples_leaf=5, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=4494, n_jobs=8,
            oob_score=False, random_state=979271, verbose=0,
            warm_start=False)

    clf.fit(train_x, train_y)

    ccv = CalibratedClassifierCV(base_estimator=clf,method="sigmoid",cv="prefit")
    ccv.fit(valid_x,valid_y)

    valid_predictions = ccv.predict_proba(valid_x)
    test_predictions= ccv.predict_proba(test_x)

    loss = test(valid_y,valid_predictions,True)
    if  loss<0.52:
        data.saveData(valid_predictions,"../valid_results/valid_"+str(model_id)+".csv")
        data.saveData(test_predictions,"../results/results_"+str(model_id)+".csv")

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
