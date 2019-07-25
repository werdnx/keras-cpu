import pandas as pd

# Read in white wine data
test_transaction = pd.read_csv("/root/frd/resources/test_transaction.csv", sep=',')
print(test_transaction)
# Read in red wine data
test_identity = pd.read_csv("/root/frd/resources/test_identity.csv", sep=',')
print(test_identity)
merged = pd.merge(test_transaction,test_identity, on='TransactionID',how='left')
print(merged)
merged.to_csv("/root/frd/resources/merged_predict2.csv", index=False)
