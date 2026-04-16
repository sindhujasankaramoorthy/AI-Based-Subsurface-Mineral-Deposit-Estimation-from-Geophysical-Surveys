import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MineralCNN, mc_dropout_predict

# Load model
model = MineralCNN(input_size=30, depth_out=20)
model.load_state_dict(torch.load("mineral_model.pth"))

# Load seismic input
seismic = np.load("seismic_surface.npy").astype(np.float32)
x_input = torch.tensor(seismic[np.newaxis, np.newaxis, :, :])  # (1,1,30,30)

# Get probabilistic prediction with uncertainty
mean_vol, uncertainty_vol = mc_dropout_predict(model, x_input, n_samples=100)
mean_vol = mean_vol[0]         # (30, 30, 20)
uncertainty_vol = uncertainty_vol[0]  # (30, 30, 20)

print("Predicted volume shape:", mean_vol.shape)
print(f"Max probability: {mean_vol.max():.3f}, Mean: {mean_vol.mean():.3f}")

# ── Top-K drill site selection ──
# Score each XY location: mean probability across all depths
# Penalize high uncertainty to prefer reliable predictions
UNCERTAINTY_PENALTY = 0.3
K = 5  # number of drill sites to recommend

scores = mean_vol.mean(axis=2) - UNCERTAINTY_PENALTY * uncertainty_vol.mean(axis=2)

# Get top-K locations
flat_indices = np.argsort(scores.ravel())[::-1][:K]
top_k_sites = [(idx // 30, idx % 30) for idx in flat_indices]

print("\n── Top-K Recommended Drill Sites ──")
for rank, (x, y) in enumerate(top_k_sites):
    prob = mean_vol[x, y, :].mean()
    uncert = uncertainty_vol[x, y, :].mean()
    economic_score = prob * (1 - uncert)  # simple viability score
    print(f"  Rank {rank+1}: Grid ({x},{y}) | Avg grade prob: {prob:.3f} | "
          f"Uncertainty: {uncert:.3f} | Economic score: {economic_score:.3f}")

np.save("mean_volume.npy", mean_vol)
np.save("uncertainty_volume.npy", uncertainty_vol)

# ── Visualization ──
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Surface probability
axes[0,0].imshow(mean_vol[:,:,10], cmap='hot', vmin=0, vmax=1)
axes[0,0].set_title("Mineral prob at depth z=10")
for x,y in top_k_sites:
    axes[0,0].plot(y, x, 'c*', markersize=15)
axes[0,0].legend(["Drill sites"], loc='upper right')

# Uncertainty map
axes[0,1].imshow(uncertainty_vol[:,:,10], cmap='Blues', vmin=0)
axes[0,1].set_title("Uncertainty at z=10")

# Cross-section with confidence envelope
depth_profile = mean_vol[15, :, :]  # row 15, all y and z
uncert_profile = uncertainty_vol[15, :, :]
axes[0,2].imshow(depth_profile.T, cmap='hot', origin='lower', aspect='auto')
axes[0,2].set_title("Cross-section (row 15)")
axes[0,2].set_xlabel("Y"); axes[0,2].set_ylabel("Depth (Z)")

# Max depth probability per site
for rank, (x, y) in enumerate(top_k_sites):
    depth_curve = mean_vol[x, y, :]
    uncert_curve = uncertainty_vol[x, y, :]
    axes[1,rank if rank < 3 else 2].plot(depth_curve, label=f"Site {rank+1} mean")
    axes[1,rank if rank < 3 else 2].fill_between(
        range(20),
        depth_curve - uncert_curve,
        depth_curve + uncert_curve,
        alpha=0.3, label="±1σ"
    )
    axes[1,rank if rank < 3 else 2].set_title(f"Drill site {rank+1} depth profile")
    axes[1,rank if rank < 3 else 2].legend()
    axes[1,rank if rank < 3 else 2].set_xlabel("Depth"); axes[1,rank if rank < 3 else 2].set_ylabel("Grade prob")

plt.tight_layout()
plt.savefig("predictions_output.png", dpi=150)
plt.show()
print("Saved predictions_output.png")