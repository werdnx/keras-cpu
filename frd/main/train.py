import datetime as datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from numpy import genfromtxt
import tensorflow as tf

import pandas as pd


def prepare_train():
    return prepare("/root/frd/resources/learn/X_train.csv")


def prepare_test():
    return prepare("/root/frd/resources/learn/X_test.csv")


def get_y_train():
    return genfromtxt("/root/frd/resources/learn/y_train.csv", delimiter=',')


def get_y_test():
    return genfromtxt("/root/frd/resources/learn/y_test.csv", delimiter=',')


def prepare(file_name):
    res = pd.read_csv(file_name, sep=',')
    for column in res.columns:
        if res[column].dtype == type(object):
            le = LabelEncoder()
            res[column] = le.fit_transform(res[column])

    scaler = StandardScaler().fit(res)
    res = scaler.transform(res)
    return res

def create_model():
    model = Sequential()
    model.add(Dense(434, activation='relu', input_shape=(433,)))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(289, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])
    X_train = prepare_train()
    y_train = get_y_train()
    model.fit(X_train, y_train, epochs=2, verbose=1)
    return model


def main():
    model = create_model()
    X_test = prepare_test()
    y_test = get_y_test()
    score = model.evaluate(X_test, y_test, verbose=1)
    print("model score")
    print(score)
    model.save("/root/frd/xeras_model")


if __name__ == "__main__":
    main()
