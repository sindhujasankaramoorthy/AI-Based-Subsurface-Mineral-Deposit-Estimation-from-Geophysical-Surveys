import torch
import torch.nn as nn
import numpy as np
from model import MineralCNN
import matplotlib.pyplot as plt

# Load data
deposit_3d = np.load("deposit_3d.npy").astype(np.float32)  # (30,30,20)
seismic = np.load("seismic_surface.npy").astype(np.float32) # (30,30)

# Create training pairs: seismic surface → 3D deposit
# For hackathon: use augmentation to create more samples
def make_dataset(seismic, deposit, n_augments=200):
    X, Y = [], []
    NX, NY = seismic.shape

    for _ in range(n_augments):
        noisy = seismic + np.random.normal(0, 0.08, seismic.shape)
        noisy_deposit = deposit + np.random.normal(0, 0.02, deposit.shape)

        noisy_deposit = np.clip(noisy_deposit, 0, 1)

        X.append(noisy[np.newaxis, :, :])
        Y.append(noisy_deposit)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


X, Y = make_dataset(seismic, deposit_3d)

split = int(0.85 * len(X))

X_train = torch.tensor(X[:split], dtype=torch.float32)
X_val   = torch.tensor(X[split:], dtype=torch.float32)

Y_train = torch.tensor(Y[:split], dtype=torch.float32)
Y_val   = torch.tensor(Y[split:], dtype=torch.float32)

# Model, optimizer, loss
model = MineralCNN(input_size=30, depth_out=20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
EPOCHS = 50
BATCH = 16
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_train))
    epoch_loss = 0
    for i in range(0, len(X_train), BATCH):
        idx = perm[i:i+BATCH]
        xb, yb = X_train[idx], Y_train[idx]
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        vloss = criterion(val_pred, Y_val).item()

    train_losses.append(epoch_loss / (len(X_train)//BATCH))
    val_losses.append(vloss)

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train loss: {train_losses[-1]:.4f} | Val loss: {vloss:.4f}")

torch.save(model.state_dict(), "mineral_model.pth")
print("Model saved!")

# Plot loss curve
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title("Training curve"); plt.legend()
plt.savefig("training_curve.png")
plt.show()
