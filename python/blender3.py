__author__ = 'hujie'
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import os
import random
from tqdm import tqdm
import re
import gc

# Load validation data set

valid = pd.read_csv("./data/valid_data.csv")

labels = valid['target']

class2num={"Class_1":1,"Class_2":2,"Class_3":3,"Class_4":4,"Class_5":5,"Class_6":6,"Class_7":7,"Class_8":8,"Class_9":9}
for k,v in class2num.items():
    tmp=[0]*9
    for i in range(9):
        if i+1 == v:
            tmp[i]=1.0
    class2num[k]=tmp

valid_y=[]
for i in range(0,len(labels.values)):
    valid_y.append(class2num[labels.values[i]])
valid_y=np.array(valid_y)

def getID(s):
    return int(re.findall(r'[\d]+',s)[0])


def optim(model_values):

    def log_loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, model_values):
            final_prediction += weight*prediction

        return log_loss(valid_y, final_prediction)

    best_score= 10000000000.0
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    bounds = [(0,1)]*len(model_values)
    '''
    for i in range(5):
        weights = [0.0]*len(model_values)
        for i in range(len(weights)):
            weights[i]=random.uniform(0, 1)

        res = minimize(log_loss_func, weights, method='SLSQP', bounds=bounds, constraints=cons)

        if res['fun']<best_score:
            best_score = res['fun']
            best_weights=res['x']
    '''
    best_weights=[1.0/len(model_values)]*len(model_values)
    best_score=log_loss_func(best_weights)
    return best_score,best_weights

def getValue(f):
    data=pd.read_csv("./valid_results/"+f+".csv")
    _data=data[["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]]
    return _data.values

def getRV(f):
    data=pd.read_csv("./results/"+f+".csv")
    _data=data[["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]]
    return _data.values

def blend_rv(models,weights):
    results=[]
    for m in models:
        results.append(getRV("results_"+str(m)))

    final_rv = 0
    for weight,result in zip(weights, results):
        final_rv += weight*result


    df = pd.DataFrame(final_rv) # A is a numpy 2d array
    df.index+=1
    headers = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
    df.to_csv("./final_results/results.csv", header=headers,index=True, index_label = 'id')

    print('Blended!')


def blend(init_models):
    model_values=[]
    models=[]
    models_dict={}
    for m in init_models:
        models_dict[m]=True
        models.append(m)
        model_values.append(getValue("valid_"+str(m)))

    # Step #1 finds all models
    fname=[]
    fname_rv=[]
    for file in os.listdir("./valid_results"):
        fname.append(file)
    for file in os.listdir("./results"):
        fname_rv.append(file)

    fname = sorted(fname, key=getID)
    fname_rv = sorted(fname_rv, key=getID )

    minimum_loss,best_weights=optim(model_values)
    for i in range(len(fname)):
        f=fname[i]
        if not getID(f) in models_dict:
            models.append(getID(f))
            model_values.append(getValue("valid_"+str(getID(f))))
            loss,weights=optim(model_values)
            if minimum_loss-loss>=0.0001:
                minimum_loss=loss
                best_weights=weights
                models_dict[getID(f)]=True
                #print('Found new Model:'+str(getID(f)))
            else:
                models.pop()
                model_values.pop()
        #print('current loss:'),
        #print(minimum_loss)
        gc.collect()
    #print('minimum loss:')
    #print(minimum_loss)
    return models,weights,minimum_loss


#blend([6,13,90001,90007,90014,90016,90018,27,20,7000,36,6300,212,6203,20])

fname=[]
minimum_loss=100.0
for f in os.listdir("./valid_results"):
    fname.append(f)
for i in tqdm(range(len(fname))):
    f=fname[i]
    models,weights,loss= blend([getID(f)])
    if minimum_loss>loss:
        minimum_loss=loss
        best_weights=weights
        best_models=models
        print('new loss:')
        print(minimum_loss)

blend_rv(best_models,best_weights)
