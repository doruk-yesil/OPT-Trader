import pandas as pd

df = pd.read_csv("data/btc_30m_labeled.csv")

train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

df_train.to_csv("data/train.csv", index=False)
df_test.to_csv("data/test.csv", index=False)

print("Train:", len(df_train))
print("Test:", len(df_test))
