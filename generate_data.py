import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Set grid size (small for speed)
NX, NY, NZ = 30, 30, 20  # 3D voxel grid

# Simulate a mineral deposit: a blob at depth
np.random.seed(42)
deposit = np.zeros((NX, NY, NZ))
cx, cy, cz = 15, 15, 12   # center of deposit
for i in range(NX):
    for j in range(NY):
        for k in range(NZ):
            dist = ((i-cx)**2 + (j-cy)**2 + (k-cz)**2)**0.5
            deposit[i,j,k] = max(0, 1 - dist/8)  # grade falls off with distance

deposit = gaussian_filter(deposit, sigma=2)
deposit += np.random.normal(0, 0.05, deposit.shape)  # add noise
deposit = np.clip(deposit, 0, 1)

# Simulate 2D seismic profiles (surface slices + noise)
seismic_profiles = deposit[:, :, 0] + np.random.normal(0, 0.1, (NX, NY))

# Drill-core assay: take vertical samples at random XY locations
drill_cores = []
for _ in range(50):
    x, y = np.random.randint(0, NX), np.random.randint(0, NY)
    core = deposit[x, y, :]  # full depth column
    drill_cores.append(((x, y), core))

np.save("deposit_3d.npy", deposit)
np.save("seismic_surface.npy", seismic_profiles)
print("Data generated! Shape:", deposit.shape)

# Quick visualization
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Surface seismic")
plt.imshow(seismic_profiles, cmap='seismic')
plt.colorbar()
plt.subplot(1,3,2)
plt.title("Deposit depth slice (z=12)")
plt.imshow(deposit[:,:,12], cmap='hot')
plt.colorbar()
plt.subplot(1,3,3)
plt.title("Vertical cross-section")
plt.imshow(deposit[15,:,:].T, cmap='hot', origin='lower')
plt.colorbar()
plt.tight_layout()
plt.savefig("data_preview.png")
plt.show()
print("Saved data_preview.png")