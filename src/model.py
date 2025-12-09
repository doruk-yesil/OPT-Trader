import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, num_features, seq_len=128, d_model=64, nhead=4, num_layers=2, num_classes=3):
        super().__init__()

        self.input_layer = nn.Linear(num_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model * seq_len, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        return out
