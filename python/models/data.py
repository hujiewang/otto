__author__ = 'hujie'

import pandas as pd
import numpy as np

# Returns train_x,valid_x in 2d, train_y, valid_y in 1d  and start with '0'
def loadData():
    # Preprocess
    class2num={"Class_1":1,"Class_2":2,"Class_3":3,"Class_4":4,"Class_5":5,"Class_6":6,"Class_7":7,"Class_8":8,"Class_9":9}

    features=[]
    for i in range(1,94):
        features.append("feat_"+str(i))

    # Load data
    train = pd.read_csv("../data/train_data.csv")
    valid = pd.read_csv("../data/valid_data.csv")
    test = pd.read_csv("../data/test_data.csv")

    train_labels = train['target']
    valid_labels = valid['target']

    train_x=train[features].values
    valid_x=valid[features].values
    test_x=test[features].values

    train_y=[]
    valid_y=[]

    for i in range(0,len(train_labels.values)):
        train_y.append(class2num[train_labels.values[i]]-1)
    train_y=np.array(train_y)

    for i in range(0,len(valid_labels.values)):
        valid_y.append(class2num[valid_labels.values[i]]-1)
    valid_y=np.array(valid_y)

    train_x = train_x.astype(np.float64)
    valid_x = valid_x.astype(np.float64)
    train_y = train_y.astype(np.float64)
    valid_y = valid_y.astype(np.float64)
    test_x = test_x.astype(np.float64)
    print('Data has been loaded!')
    return train_x,train_y,valid_x,valid_y,test_x

# Saves predictions(2-d numpy array) into './results/results.csv'
def saveData(predictions,fpath):
    df = pd.DataFrame(predictions) #predictions is a numpy 2d array
    df.index+=1
    headers = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
    df.to_csv(fpath, header=headers,index=True, index_label = 'id')
    print('Predictions has been saved!')

