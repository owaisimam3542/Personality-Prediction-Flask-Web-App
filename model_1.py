import pandas as pd
from numpy import *
import numpy as np
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
df=pd.read_csv('training_dataset.csv')
array = df.values
for i in range(len(array)):
    if array[i][0]=="Male":
        array[i][0]=1
    else:
        array[i][0]=0
da=pd.DataFrame(array)
maindf =da[[0,1,2,3,4,5,6]]
mainarray=maindf.values
print (mainarray)
temp=da[7]
train_y =temp.values
# print(train_y)
# print(mainarray)
train_y=temp.values
for i in range(len(train_y)):
    train_y[i] =str(train_y[i])
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
mul_lr.fit(mainarray, train_y)
import pickle
pickle.dump(mul_lr, open('a1.pkl','wb'))
model_1 = pickle.load(open('a1.pkl','rb'))