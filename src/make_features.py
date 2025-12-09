import pandas as pd
import ta

def add_features(df):
    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # EMA
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_high"] - df["bb_low"]

    # ATR (volatility)
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"])
    df["atr"] = atr.average_true_range()

    # Normalize volume
    df["volume_change"] = df["volume"].pct_change()

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/btc_30m.csv")
    df = add_features(df)
    df.dropna(inplace=True)
    df.to_csv("data/btc_30m_features.csv", index=False)
    print("Saved data/btc_30m_features.csv")
    print(df.head())
