from keras.models import Sequential
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import sys

path = "/root/"


def prepare(res):
    for column in res.columns:
        if res[column].dtype == type(object):
            le = LabelEncoder()
            print("column data")
            print(res[column])
            res[column] = le.fit_transform(res[column])

    scaler = StandardScaler().fit(res)
    res = scaler.transform(res)
    return res



def main():
    model = load_model("/root/frd/xeras_model")
    index_list = sys.argv[1:len(sys.argv)]
    print(index_list)
    merged = pd.read_csv(path + "frd/resources/merged_predict.csv", sep=',')
    print("MERGE DF")
    print(merged.info())
    result = merged.loc[:, index_list]
    print("BEFORE PREPARE")
    print(result.info())
    df = prepare(result)
    print("test DF")
    print(df)
    y = model.predict(df)
    print("MODEL RESULT")
    print(y)
    print("MODEL HISTOGRAM")
    print(np.histogram(y))
    result = pd.DataFrame(columns=['TransactionID','isFraud'])
    result['TransactionID'] = merged.loc[:,'TransactionID']
    y[y <= 0.5]=0	
    y[y > 0.5]=1
    result['isFraud'] = y
    result.to_csv("/root/frd/resources/learn/result.csv", index=False)




if __name__ == "__main__":
    main()
