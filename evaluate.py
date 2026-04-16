import numpy as np
from sklearn.metrics import mean_squared_error

# Load ground truth and predictions
true_vol = np.load("deposit_3d.npy")
pred_vol = np.load("mean_volume.npy")
uncert_vol = np.load("uncertainty_volume.npy")

# ── Metric 1: RMSE on held-out drill-core data ──
np.random.seed(99)
held_out = [(np.random.randint(0,30), np.random.randint(0,30)) for _ in range(10)]

all_true, all_pred = [], []
for (x, y) in held_out:
    all_true.extend(true_vol[x, y, :])
    all_pred.extend(pred_vol[x, y, :])

rmse = np.sqrt(mean_squared_error(all_true, all_pred))
print(f"RMSE on held-out drill cores: {rmse:.4f}")

# ── Metric 2: Uncertainty calibration ──
# Check if true values fall within predicted ± 1.96σ (95% interval)
lower = pred_vol - 1.96 * uncert_vol
upper = pred_vol + 1.96 * uncert_vol
coverage = np.mean((true_vol >= lower) & (true_vol <= upper))
print(f"Uncertainty interval coverage (target ~0.95): {coverage:.3f}")

# ── Metric 3: IoU — deposit boundary ──
# Threshold at 0.3 to define "deposit present"
THRESHOLD = 0.3
true_binary = (true_vol > THRESHOLD).astype(int)
pred_binary = (pred_vol > THRESHOLD).astype(int)
intersection = (true_binary & pred_binary).sum()
union = (true_binary | pred_binary).sum()
iou = intersection / (union + 1e-8)
print(f"Spatial IoU (deposit boundary): {iou:.4f}")

# ── Summary ──
print("\n── Evaluation Summary ──")
print(f"  RMSE:              {rmse:.4f}  (lower is better)")
print(f"  Uncertainty cov.:  {coverage:.3f}  (closer to 0.95 is better)")
print(f"  Boundary IoU:      {iou:.4f}  (higher is better, max=1.0)")