import csv
import os
import sys
#from sklearn.svm import SVC
#from sklearn.preprocessing import normalize
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

train_data = []
train_label = []


with open(sys.argv[1],'rb') as traincsv:
    reader = csv.reader(traincsv)
    next(reader)
    for row in reader:                
        tmp = row[1:23]
        train_data.append(tmp)
        train_label.append(row[24])

model = CatBoostClassifier()
#model = CatBoostClassifier(iterations=1000, depth=12, learning_rate=0.01, loss_function='Logloss', logging_level='Verbose')
train_pool = Pool(train_data, train_label)#, cat_features=[0,2,5])
model.fit(train_pool)

model.save_model('model')

