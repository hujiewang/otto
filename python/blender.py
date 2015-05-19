import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import os
import random
from tqdm import tqdm
import re
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


# Load list of model results
predictions = []
results = []
fname = []
fname_rv = []

for file in os.listdir("./valid_results"):
    fname.append(file)
for file in os.listdir("./results"):
    fname_rv.append(file)
def getID(s):
    return int(re.findall(r'[\d]+',s)[0])

fname = sorted(fname, key=getID)
fname_rv = sorted(fname_rv, key=getID )

# Double check every model has two files
#assert(len(fname) == len(fname_rv))
for i in range(len(fname)):
    assert(getID(fname[i])==getID(fname_rv[i]),fname[i]+' '+fname_rv[i])

threshold = 0.480

#exclusion = set([6,13,90001,90007,90014,90016,90018,27,20,7000,36,6300,212,6203,20])

exclusion = set([])

stat=[]
for f in fname:
    data=pd.read_csv("./valid_results/"+f)
    _data=data[["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]]
    loss = log_loss(valid_y, _data.values)
    if  loss < threshold or getID(f) in exclusion:
        predictions.append(_data.values)
        stat.append((getID(f),loss))

#stat = sorted(stat, key=lambda t: t[1])
print('There are {0} models under {1} threshold'.format(len(stat),threshold))
for v in stat:
    print(v)
# finding the optimum weights

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(valid_y, final_prediction)

best_score= 100.0
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
bounds = [(0,1)]*len(predictions)

for i in tqdm(range(2)):
    weights = [0.0]*len(predictions)
    for i in range(len(weights)):
        weights[i]=random.uniform(0, 1)


    res = minimize(log_loss_func, weights, method='SLSQP', bounds=bounds, constraints=cons)

    if res['fun']<best_score:
        best_score = res['fun']
        best_weights=res['x']


print('Best Ensamble Score: {best_score}'.format(best_score=best_score))
print('\nBlending models...')


for f in fname_rv:
    data=pd.read_csv("./results/"+f)
    _data=data[["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]]
    results.append(_data.values)


final_rv = 0
for weight,result in zip(best_weights, results):
    final_rv += weight*result


df = pd.DataFrame(final_rv) # A is a numpy 2d array
df.index+=1
headers = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
df.to_csv("./final_results/results.csv", header=headers,index=True, index_label = 'id') # C is a list of string corresponding to the title of each column of A

print('Blended!')
