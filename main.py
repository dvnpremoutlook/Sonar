import tensorflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv("D:\Python\PythonProjects\Copy of sonar data.csv",header= None)

print(sonar_data.head())
print(sonar_data.shape)
print(sonar_data.describe())
print(sonar_data[60].value_counts())
print(sonar_data.groupby(60).mean())
x = sonar_data.drop(columns=60, axis= 1)
y = sonar_data[60]

print(x)
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.1,stratify=y,random_state=1)

print(x_train)
print(x_test)

model = LogisticRegression()
model.fit(x_train,y_train)
x_train_prediction = model.predict(x_train)
trainng_data_acc = accuracy_score(x_train_prediction,y_train)


x_test_prediction = model.predict(x_test)
test_data_acc = accuracy_score(x_test_prediction,y_test)


print(trainng_data_acc)
print(test_data_acc)

input_data = (0.0453,0.0523,0.0843,0.0689,0.1183,0.4545,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.0530,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.0140,0.0049,0.0052,0.0044)
input_data_np = np.asarray(input_data)
input_data_reshape = input_data_np.reshape(1,-1)
prediction = model.predict(input_data_reshape)
print(prediction)
