import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import BTCDataset
from model import TransformerModel

def train_model():
    seq_len = 128

    train_ds = BTCDataset("data/train.csv", seq_len=seq_len)
    test_ds = BTCDataset("data/test.csv", seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    num_features = train_ds.X.shape[1]

    model = TransformerModel(num_features=num_features, seq_len=seq_len)
    model = model.to("cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(10):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model.pt")
    print("Model saved as model.pt")

if __name__ == "__main__":
    train_model()
