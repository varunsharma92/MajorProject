import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("candidates.csv"). as_matrix()
#print(data)
clf=DecisionTreeClassifier()
#training dataset
xtrain= data[0:21000,0:3]#taking data from 0-21000 but from coulumn 1 onwards because column 0 contains the labels
train_label=data[0:21000,4]#taking 0th column of first 21000 rows(labels)
clf.fit(xtrain,train_label)
#testing data
xtest=data[21000:50000,0:3]#testing data features
actual_label=data[21000:,4]#testing data labels
accuracy= clf.score(xtest, actual_label)
print(accuracy)
d=xtest[8]#taking a test dataset to check our prediction
d.shape=(28,28)#convert into 28x28 matrix
pt.imshow(255-d,cmap='gray')#color
print(clf.predict( [xtest[8]] ))
pt.show()
"""p=clf.predict(xtest)
count = 0
for i in range(0,21000):
    count+=1 if p[i]==actual_label[i] else 0
print("Accuracy=",(count/21000)*100)"""
