import pandas as pd

all = pd.read_csv("/Users/werdn/PycharmProjects/frd/resorces/merged.csv", sep=',')

non_fraud = all[all['isFraud'] == 0]
fraud = all[all['isFraud'] != 0]

non_fraud.to_csv("/Users/werdn/PycharmProjects/frd/resorces/non_fraud.csv", index=False)
fraud.to_csv("/Users/werdn/PycharmProjects/frd/resorces/fraud.csv", index=False)