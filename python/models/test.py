__author__ = 'hujie'
import numpy as np
from sklearn.metrics import log_loss

def test(valid_y,valid_predictions):
    class2num={}
    for label in range(9):
        tmp=[0]*9
        for i in range(9):
            if i == label:
                tmp[i]=1.0
        class2num[label]=tmp

    _valid_y=[]
    for i in range(0,len(valid_y)):
        _valid_y.append(class2num[valid_y[i]])
    _valid_y=np.array(_valid_y)

    print('valid loss:')
    print(log_loss(valid_y,valid_predictions))