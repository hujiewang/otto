__author__ = 'hujie'
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import os
import random
from tqdm import tqdm
import re

def getID(s):
    return int(re.findall(r'[\d]+',s)[0])

'''
# base benchmark #1
# Best Ensamble Score: 0.412838972743
# avg all loss: 0.413709818435
cat=[
    [6,13,14,15,16,27,17,8,20,11,19,22], # NN (pure NN) 0.423969040664
    #[27,17,8,20,11,19,22], # NN (experts)
    [36,37,38,35,39,32,33,34,30,29,28,26,23,25], # XGB 0.434705244415
]
'''

'''
# base benchmark #2
# Best Ensamble Score: 0.412723046753
# avg all loss: 0.413709818435
cat=[
    [6,13,14,15,16], # NN (pure NN) 0.426725660318
    [27,17,8,20,11,19,22], # NN (experts) 0.425725245682
    [36,37,38,35,39,32,33,34,30,29,28,26,23,25], # XGB 0.434705244415
]
'''

'''
# base benchmark #3
# Best Ensamble Score: 0.410233002895
# avg all loss: 0.413709818435
cat=[
    [6],[13],[14],[15],[16],[27],[17],[8],[20],[11],[19],[22],[36],[37],[38],[35],[39],[32],[33],[34],[30],[29],[28],[26],[23],[25],
]
'''

'''
cat=[
    [6,13,14,15,16], # NN (pure NN)
    [27,17,19,8,22,11,218,219], # NN (experts)
    [1018,1101,1203,37,1113,1201,1011,1100,1022,1024,1030,1031,1128,1224,1227], # XGB (random)
    [500,501,502,503,504,505,5000,5001,5002,5100,5101,5102,5300,5301,5302,5400,5401,5402], # XGB (config #1)
]
'''

'''

# Best Ensamble Score: 0.410233002895
# avg all loss: 0.413709818435
cat=[

    #[6],[13],[16],[20],[27],[36],[40],[45],[72],[82],[92],[106],[108],[113],[127],[148],[174],[197],[212],[6004],[6203],[6300],
    #[6203],[6315],[7000],[90000],[90001],[90007],[90014],[90016],[90018],
    #[6],[13],[20],[27],
    #[40],[45],[72],[82],
    #[92],[106],[108],[113],
   # [127],[148],[174],[197],
    #[6004],[6300],[6203],
    #[6315],[7000],[90000],[90001],
   # [90007],[90014],[90016],[90018],
    #[90022],[90023],[90024],[90025],[90026],[90027],[90028],[90029],[90030],[90031],[90032],[90033],[90034],[90035],[90036],[90037],[90038],[90039],
    [90022,90023,90024,90025,90026,90027,90028,90029,90030,90031,90032,90033,90034,90035,90036,90037,90038,90039],
    [6], [13], [16],[90001], [90007], [90014], [90016],[90018], # pure NN #12
 [27],[20],
#[11],
[218],[219],[19], [7000], # NN experts
 [36],
#[502],
[504],[5301],[5103], [6300],[212], [6203], [20],
    [16000,16001,16003],

]
'''

cat=[
[6], [13],[90001],
[90007], [90014], [90016],[90018], # pure NN
[27],[20],
 [7000], [36],
 [6300],[212],
[6203], [20],
]
#[6, 13, 20, 27, 40, 45, 72, 82, 92, 98, 106, 108, 113, 127, 148, 174, 175, 197, 6004, 6203, 6300, 6315, 7000, 90000, 90001, 90007, 90014, 90016, 90018]

'''
# Best Ensamble Score: 0.410233002895
# avg all loss: 0.413709818435
cat=[
    #[0],[1],[2],  # cart glm svm
    [90007],[90011],[90008],[90018],[90002],[90016],[90013],[90003],[90017],[90010], #pure NN #10

    [27],[20],[17],[7001],[19],[219],[218],[7000],[8],[22], #NN experts #10

    [36],[11036],[9051],[5300],[11059],[5102],[501],[502],[504],[5301], #XGB #10
    ]
'''

'''
# Best Ensamble Score: 0.410233002895
# avg all loss: 0.413709818435
cat=[
    #[0],[1],[2],  # cart glm svm
    [90007,90011,90008,90018,90002,90016,90013,90003,90017,90010], #pure NN #10

    [27,20,17,7001,19,219,218,7000,8,22], #NN experts #10

    [36,11036,9051,5300,11059,5102,501,502,504,5301] #XGB #10
    ]
'''
'''
cat=[
    [6,13,14,15,16,7000,7001,7002], # NN (pure NN)
    [27,17,19,8,22,11,218,219], # NN (experts)
    [36,5300,5102,501,502,504,5301,5103,503,6300,212,5000,6203,5001,5101,505,20,1018,500,5002,6020,5401,506,5400,6103], #XGB
]
'''
'''
cat=[
    [0],[1],
]
'''
class2num={"Class_1":1,"Class_2":2,"Class_3":3,"Class_4":4,"Class_5":5,"Class_6":6,"Class_7":7,"Class_8":8,"Class_9":9}
for k,v in class2num.items():
    tmp=[0]*9
    for i in range(9):
        if i+1 == v:
            tmp[i]=1.0
    class2num[k]=tmp

# Load validation data set

valid = pd.read_csv("./data/valid_data.csv")
train = pd.read_csv("./data/train_data.csv")
valid_labels = valid['target']
train_labels = train['target']

valid_y=[]
train_y=[]
for i in range(0,len(valid_labels.values)):
    valid_y.append(class2num[valid_labels.values[i]])
valid_y=np.array(valid_y)

for i in range(0,len(train_labels.values)):
    train_y.append(class2num[train_labels.values[i]])
train_y=np.array(train_y)

def valid_loss(model_id):
    data=pd.read_csv("./valid_results/valid_"+str(model_id)+".csv")
    _data=data[["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]]
    return log_loss(valid_y,_data.values)


'''
# model selections
for file in os.listdir("./valid_results"):
    if getID(file)>=6000 and getID(file)<=7000 and valid_loss(getID(file))<0.438:
        cat[2].append(getID(file))

print(cat)
'''

# step #0 preprocess
# step #1 average each category

avg_all=0
count=0
for c in cat:
    for f in c:
        data=pd.read_csv("./valid_results/valid_"+str(f)+".csv")
        _data=data[["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]]
        avg_all+=_data.values
        count+=1
avg_all/=count
print('avg all loss:'),
print(log_loss(valid_y,avg_all))

def average(c,prefix):
    avg=0
    for f in c:
        data=pd.read_csv(prefix+str(f)+".csv")
        _data=data[["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]]
        avg+=_data.values
    avg/=len(c)
    return avg

av_cat=[]
for c in cat:
    valid_avg=average(c,"./valid_results/valid_")
    results_avg=average(c,"./results/results_")
    av_cat.append([valid_avg,results_avg])


print('There are '+str(len(av_cat))+' categories')

for i in range(len(av_cat)):
    print('cat #'+str(i)+' avg loss: '),
    print(log_loss(valid_y,av_cat[i][0]))

# step #2 find the optimal linear combination of each averaged category
print('==================== Optimization ======================')
# First we try average
def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, results in zip(weights, av_cat):
            final_prediction += weight*results[0] # here we use valid_avg

    return log_loss(valid_y, final_prediction)

best_score= 100.0
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
bounds = [(0,1)]*len(av_cat)

for i in tqdm(range(5)):
    weights = [0.0]*len(av_cat)
    for i in range(len(weights)):
        weights[i]=random.uniform(0, 1)


    res = minimize(log_loss_func, weights, method='SLSQP', bounds=bounds, constraints=cons)

    if res['fun']<best_score:
        best_score = res['fun']
        best_weights=res['x']

print('Best Ensamble Score: {best_score}'.format(best_score=best_score))
print('Best weights: {best_weights}'.format(best_weights=best_weights))
print('\nBlending models...')

final_rv = 0
for weight,result in zip(best_weights, av_cat):
    final_rv += weight*result[1] # here we use results_avg

#avg
final_rv_avg = 0
for weight,result in zip(best_weights, av_cat):
    final_rv_avg += result[1] # here we use results_avg
final_rv_avg/=len(av_cat)

df = pd.DataFrame(final_rv) # A is a numpy 2d array
df.index+=1

df_avg = pd.DataFrame(final_rv_avg) # A is a numpy 2d array
df_avg.index+=1

headers = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
df.to_csv("./final_results/results.csv", header=headers,index=True, index_label = 'id') # C is a list of string corresponding to the title of each column of A
df_avg.to_csv("./final_results/results_avg.csv", header=headers,index=True, index_label = 'id') # C is a list of string corresponding to the title of each column of A
print('Blended!')

'''
# =================== Manual Optimization=======================

print('=================== Manual Optimization=======================')

best_score= 10.0
for a in range(1,10-3,1):
    for b in range(1,10-a-2,1):
        for c in range(1,10-a-b-1,1):
            d=10-a-b-c
            assert(a+b+c+d==10)
            final_prediction = (a/10.0)*av_cat[0][0]+(b/10.0)*av_cat[1][0]+(c/10.0)*av_cat[2][0]+(d/10.0)*av_cat[3][0]
            loss = log_loss(valid_y, final_prediction)
            weights=np.array([a/10.0,b/10.0,c/10.0,d/10.0])
            if loss < best_score:
                best_score=loss
                best_weights= weights
print('Best Ensamble Score: {best_score}'.format(best_score=best_score))
print('Best weights: {best_weights}'.format(best_weights=best_weights))
'''