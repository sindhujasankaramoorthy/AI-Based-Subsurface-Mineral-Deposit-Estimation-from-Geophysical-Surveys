import torch
import torch.nn as nn
import numpy as np

class MineralCNN(nn.Module):
    """
    Takes 2D surface seismic slice as input,
    outputs 3D probability volume with uncertainty (MC Dropout).
    """
    def __init__(self, input_size=30, depth_out=20, dropout_p=0.3):
        super().__init__()
        self.dropout_p = dropout_p

        # Encoder: extract features from 2D surface data
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Decoder: expand to 3D prediction
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * input_size * input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, input_size * input_size * depth_out),
            nn.Sigmoid()  # output = probability [0,1]
        )

        self.input_size = input_size
        self.depth_out = depth_out

    def forward(self, x):
        # x shape: (batch, 1, NX, NY)
        features = self.encoder(x)
        out = self.decoder(features)
        # Reshape to 3D volume
        return out.view(-1, self.input_size, self.input_size, self.depth_out)


def mc_dropout_predict(model, x, n_samples=50):
    """
    Run model N times with dropout ON to get uncertainty estimate.
    Returns mean prediction and standard deviation (= uncertainty).
    """
    model.train()  # keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x).cpu().numpy())
    preds = np.array(preds)  # shape: (n_samples, batch, NX, NY, NZ)
    mean_pred = preds.mean(axis=0)
    uncertainty = preds.std(axis=0)  # higher std = less confident
    return mean_pred, uncertainty