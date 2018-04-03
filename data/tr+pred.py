import csv
import os
import sys
#from sklearn.svm import SVC
#from sklearn.preprocessing import normalize
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

#print(sys.version_info)
train_data = []
train_label = []
testPrivate_data = []
Private_Y = []
testPublic_data = []
Public_Y = []

with open('Train.csv','rb') as traincsv:
    cnt = 0
    reader = csv.reader(traincsv)
    next(reader)
    for row in reader:                
        tmp = row[1:23]
        train_data.append(tmp)
        train_label.append(row[24])

with open('Test_Public.csv','rb') as testcsv:
    reader = csv.reader(testcsv)
    next(reader)
    cnt = 0
    for row in reader:
        tmp = row[1:23] 
        testPublic_data.append(tmp)

with open('Test_Private.csv','rb') as testcsv:
    reader = csv.reader(testcsv)
    next(reader)
    for row in reader:
        tmp = row[1:23]
        testPrivate_data.append(tmp)



model = CatBoostClassifier()#(iterations=1000, depth=12, learning_rate=0.01, loss_function='Logloss', logging_level='Verbose')
train_pool = Pool(train_data, train_label)#, cat_features=[0,2,5])
model.fit(train_pool)


predsPublic_proba = model.predict_proba(testPublic_data)
Public_Y = sorted(range(len(predsPublic_proba)),key=lambda i: -predsPublic_proba[i][1])

predsPrivate_proba = model.predict_proba(testPrivate_data)
Private_Y = sorted(range(len(predsPrivate_proba)),key=lambda i: -predsPrivate_proba[i][1])
#print("proba = ", predsPrivate_proba)

#print("sort = ", Public_Y)


with open('pub.csv','w') as f:
        f.write('Rank_ID')
        f.write('\n')
        for i in range (5000):
            f.writelines(str(Public_Y[i]+1))
            f.write('\n')


with open('private777.csv','w') as f:
        f.write('Rank_ID')
        f.write('\n')
        for i in range (5000):
            f.writelines(str(Private_Y[i]+1))
            f.write('\n') 
    

            


        