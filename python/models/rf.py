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
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
import getopt
import sys
import numpy as np
from operator import itemgetter
from scipy.stats import randint as sp_randint

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

    # build a classifier
    clf = RandomForestClassifier(n_jobs=8)

   # specify parameters and distributions to sample from

    param_dist = {
            "n_estimators":sp_randint(20,40),
            "criterion": ["gini", "entropy"],
            "max_depth": sp_randint(3, 10000),
            "min_samples_split": sp_randint(1, 30),
            "min_samples_leaf": sp_randint(1, 30),
            "max_features": sp_randint(1, 93),
            "bootstrap": [True, False],
            'random_state':sp_randint(1, 1000000),
            }


    # run randomized search
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=2,cv=9,n_jobs=3)
    random_search.fit(train_x,train_y)
    valid_predictions = random_search.predict_proba(valid_x)
    test_predictions= random_search.predict_proba(test_x)
    loss = test(valid_y,valid_predictions,True)
    if  loss<10.438:
        output=[loss,random_search.best_estimator_]
        print("model[\""+str(model_id)+"\"]="),
        print(output)

        data.saveData(valid_predictions,"../valid_results/valid_"+str(model_id)+".csv")
        data.saveData(test_predictions,"../results/results_"+str(model_id)+".csv")

def main(argv):
	opts,args=getopt.getopt(argv,"n:")
	for opt,arg in opts:
		if opt=='-n':
			n=int(arg)

	#print('working on '+str(n)+' to '+str(n+1000))
	#for model_id in range(n,n+1000):
   	train(999999,train_x,train_y,valid_x,valid_y,test_x)
if __name__ == "__main__":
   main(sys.argv[1:])