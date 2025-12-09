import pandas as pd

def make_labels(df, horizon=5, threshold=0.003):
    df["future_close"] = df["close"].shift(-horizon)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

    conditions = [
        df["future_return"] > threshold,
        df["future_return"] < -threshold
    ]
    choices = [1, -1]  # 1 = BUY, -1 = SELL

    df["label"] = 0
    df.loc[conditions[0], "label"] = 1
    df.loc[conditions[1], "label"] = -1

    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/btc_30m_features.csv")
    df = make_labels(df)
    df.to_csv("data/btc_30m_labeled.csv", index=False)
    print("Saved data/btc_30m_labeled.csv")
    print(df["label"].value_counts())
