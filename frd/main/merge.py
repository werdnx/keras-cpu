import pandas as pd

# Read in white wine data
train_transaction = pd.read_csv("/home/werdn/chpoker/frd/resources/train_transaction.csv", sep=',')

# Read in red wine data
train_identity = pd.read_csv("/home/werdn/chpoker/frd/resources/train_identity.csv", sep=',')
merged = pd.merge(train_transaction,train_identity, on='TransactionID',how='left')
print(merged.info())
merged.to_csv("/home/werdn/chpoker/frd/resources/merged.csv", index=False)
