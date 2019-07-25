from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv("/Users/werdn/PycharmProjects/frd/resorces/merged.csv", sep=',')
X = df.drop(['isFraud'], axis=1)
y = np.ravel(df.isFraud)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train.to_csv("/Users/werdn/PycharmProjects/frd/resorces/learn/X_train.csv", index=False)
X_test.to_csv("/Users/werdn/PycharmProjects/frd/resorces/learn/X_test.csv", index=False)
np.savetxt("/Users/werdn/PycharmProjects/frd/resorces/learn/y_train.csv", y_train, delimiter=",", fmt="%d")
np.savetxt("/Users/werdn/PycharmProjects/frd/resorces/learn/y_test.csv", y_test, delimiter=",", fmt="%d")

#from numpy import genfromtxt
#my_data = genfromtxt('my_file.csv', delimiter=',')
