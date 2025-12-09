import torch
import numpy as np
import pandas as pd
from model import TransformerModel
from explain import explain_signal

def load_model(csv_path, model_path="model.pt", seq_len=128):
    df = pd.read_csv(csv_path)

    # Drop non-numeric or useless columns just like in training
    drop_cols = ["open_time", "close_time", "future_close", "future_return"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Drop label if exists
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = df.astype(np.float32).values

    # prepare last sequence
    seq = X[-seq_len:]

    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

    # build model
    num_features = seq.shape[1]
    model = TransformerModel(num_features=num_features, seq_len=seq_len)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(seq_tensor)
        probs_t = torch.softmax(logits, dim=1)[0]  # tensor output

    # Convert tensor to list (Python floats)
    probs = probs_t.tolist()

    classes = ["SELL", "NO-TRADE", "BUY"]
    predicted = classes[int(torch.argmax(probs_t))]

    print("Prediction:", predicted)
    print("Probabilities:", probs)
    print("\n--- Signal Explanation ---")
    print(explain_signal(predicted, probs, seq))



if __name__ == "__main__":
    load_model("data/test.csv")
