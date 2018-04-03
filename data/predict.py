import csv
import os
import sys
#from sklearn.svm import SVC
#from sklearn.preprocessing import normalize
import numpy as np
from catboost import CatBoostClassifier, Pool

testPrivate_data = []
Private_Y = []
testPublic_data = []
Public_Y = []

PUBLIC_TEST = sys.argv[1]
PRIVATE_TEST = sys.argv[2]

with open(sys.argv[1],'rb') as testcsv:
    reader = csv.reader(testcsv)
    next(reader)
    for row in reader:
        tmp = row[1:23] 
        testPublic_data.append(tmp)

with open(sys.argv[2],'rb') as testcsv:
    reader = csv.reader(testcsv)
    next(reader)
    for row in reader:
        tmp = row[1:23]
        testPrivate_data.append(tmp)

model = CatBoostClassifier()
model.load_model('model')
predsPublic_proba = model.predict_proba(testPublic_data)
Public_Y = sorted(range(len(predsPublic_proba)),key=lambda i: -predsPublic_proba[i][1])

predsPrivate_proba = model.predict_proba(testPrivate_data)
Private_Y = sorted(range(len(predsPrivate_proba)),key=lambda i: -predsPrivate_proba[i][1])


with open('public.csv','w') as f:
        f.write('Rank_ID')
        f.write('\n')
        for i in range (5000):
            f.writelines(str(Public_Y[i]+1))
            f.write('\n')


with open('private.csv','w') as f:
        f.write('Rank_ID')
        f.write('\n')
        for i in range (5000):
            f.writelines(str(Private_Y[i]+5001))
            f.write('\n') 
    

            


        