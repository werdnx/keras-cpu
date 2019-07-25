import datetime as datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

print "Before read"
print datetime.datetime.now().isoformat()
X_train = pd.read_csv("/Users/werdn/PycharmProjects/frd/resorces/learn/X_train.csv", sep=',')
print "Before encode"
print datetime.datetime.now().isoformat()
for column in X_train.columns:
    if X_train[column].dtype == type(object):
        le = LabelEncoder()
        X_train[column] = le.fit_transform(X_train[column])

print "Before scaler"
print datetime.datetime.now().isoformat()

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
print "Before save"
print datetime.datetime.now().isoformat()
np.savetxt("/Users/werdn/PycharmProjects/frd/resorces/learn/X_train_norm.csv", X_train, delimiter=",")
print "end"
print datetime.datetime.now().isoformat()
