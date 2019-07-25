from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from numpy import genfromtxt
import tensorflow as tf

import sys

path = "/root/"


def merged(index_list):
    merged = pd.read_csv(path + "frd/resources/merged.csv", sep=',')
    result = merged.loc[:, index_list]
    return result


def split_for_traing(merged):
    df = merged
    y = np.ravel(df.isFraud)
    print("split_for_traing DF")
    print(df)
    X = df.drop(['isFraud'], axis=1)
    print("split_for_traing after drop DF")
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print("split_for_traing y_train = 1")
    print(np.array(filter(lambda x:x==1,y_train)))
    print("split_for_traing y_test = 1")
    print(np.array(filter(lambda x:x==1,y_test)))
    return X_train, X_test, y_train, y_test


def prepare(res):
    for column in res.columns:
        if res[column].dtype == type(object):
            le = LabelEncoder()
            res[column] = le.fit_transform(res[column])

    scaler = StandardScaler().fit(res)
    res = scaler.transform(res)
    return res

def create_model(num_of_inputs, X_train_in, y_train_in):
    print("num of neurons")
    print(num_of_inputs)
    X_train = prepare(X_train_in)
    y_train = y_train_in
    model = Sequential()
    layer_neurons = (num_of_inputs + 1) * 2 / 3
    model.add(Dense(num_of_inputs + 1, activation='relu', input_shape=(num_of_inputs,)))
    model.add(Dense(layer_neurons, activation='relu'))
#    model.add(Dense(layer_neurons, activation='relu'))
#    model.add(Dense(layer_neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])
#    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    model.fit(X_train, y_train, epochs=5, verbose=1)
    return model


# args is indexes of columns for training + marked column
def main():
    print sys.argv[1:len(sys.argv)]
#    list = [int(s) for s in sys.argv[1:len(sys.argv)]]
    list = sys.argv[1:len(sys.argv)]
    print("List of args is ")
    print(list)
    merged_df = merged(list)
    print("merged df info")
    print(merged_df.info())
    X_train, X_test, y_train, y_test = split_for_traing(merged_df)
    print("Len of X_train columns")
    print(len(X_train.columns))
    print("Len of X_test columns")
    print(len(X_test.columns))
    print("y_train")
    print(y_train)
    print("y_test")
    print(y_test)
    print("X_train")
    print(X_train)
    model = create_model(len(X_train.columns), X_train, y_train)
    X_test = prepare(X_test)
    score = model.evaluate(X_test, y_test, verbose=1)
    print("model score")
    print(score)
    model.save(path + "frd/xeras_model")




if __name__ == "__main__":
    main()
