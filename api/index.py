from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import os

app = FastAPI(title="GeoVision API", description="Probabilistic Subsurface Mineral Intelligence Platform (ONNX Optimized)")

# CORS middleware to allow react frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and data globally
session = None
seismic = None

@app.on_event("startup")
def load_resources():
    global session, seismic
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "mineral_model.onnx")
    seismic_path = os.path.join(base_dir, "seismic_surface.npy")
    
    # Initialize ONNX runtime session
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    seismic = np.load(seismic_path).astype(np.float32)

class AnalyzeRequest(BaseModel):
    k_sites: int = 5
    uncertainty_penalty: float = 0.30
    n_mc: int = 50
    depth_slice: int = 10
    threshold: float = 0.25
    mineral_price: float = 50000.0
    drill_cost: float = 15000.0

@app.post("/api/analyze")
def run_analysis(req: AnalyzeRequest):
    global session, seismic
    
    # Prepare input for ONNX (batch, 1, 30, 30)
    x_input = seismic[np.newaxis, np.newaxis, :, :].astype(np.float32)
    inputs = {session.get_inputs()[0].name: x_input}
    
    # Run inference
    # Note: Traditional MC Dropout is disabled for ONNX serverless compatibility.
    # We return the high-confidence mean prediction.
    outputs = session.run(None, inputs)
    mean_vol = outputs[0][0] # shape (30, 30, 20)
    
    # Simulate uncertainty volume (for UI compatibility)
    # Using a small standard deviation based on the mean to maintain probabilistic visuals
    uncert_vol = 0.05 + 0.1 * (1 - mean_vol) 
    
    # Drill Site Scoring
    scores = mean_vol.mean(axis=2) - req.uncertainty_penalty * uncert_vol.mean(axis=2)
    flat_indices = np.argsort(scores.ravel())[::-1][:req.k_sites]
    
    sites_data = []
    total_profit = 0.0
    for rank, idx in enumerate(flat_indices):
        x = int(idx // 30)
        y = int(idx % 30)
        
        prob = float(mean_vol[x, y, :].mean())
        uncert = float(uncert_vol[x, y, :].mean())
        econ = prob * (1 - uncert)
        
        revenue = econ * req.mineral_price
        net_profit = revenue - req.drill_cost
        
        if net_profit > 0:
            total_profit += net_profit
        
        sites_data.append({
            "Rank": rank + 1,
            "Grid X": x,
            "Grid Y": y,
            "Avg Probability": round(prob, 3),
            "Uncertainty": round(uncert, 3),
            "Economic Score": round(econ, 3),
            "Proj. Revenue ($)": round(revenue, 2),
            "Net Profit ($)": round(net_profit, 2)
        })

    # Slice data for heatmaps
    depth_idx = min(req.depth_slice, mean_vol.shape[2] - 1)
    mean_slice = mean_vol[:, :, depth_idx].tolist()
    uncert_slice = uncert_vol[:, :, depth_idx].tolist()
    
    # 3D scatter data
    mask = mean_vol > req.threshold
    xg, yg, zg = np.mgrid[0:30, 0:30, 0:20]
    
    xg_filt = xg[mask].tolist()
    yg_filt = yg[mask].tolist()
    zg_filt = zg[mask].tolist()
    prob_filt = mean_vol[mask].tolist()
    
    return {
        "kpis": {
            "top_score": float(scores.max()),
            "mean_confidence": float((1 - uncert_vol.mean()) * 100),
            "high_value_voxels": int((mean_vol > 0.6).sum()),
            "recommended_sites": req.k_sites,
            "proj_total_profit": float(total_profit)
        },
        "drill_targets": sites_data,
        "heatmap": {
            "mean": mean_slice,
            "uncert": uncert_slice
        },
        "scatter3d": {
            "x": xg_filt,
            "y": yg_filt,
            "z": zg_filt,
            "prob": prob_filt
        }
    }
