from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import torch
import sys
import os

# Load model from local api directory
from .model import MineralCNN, mc_dropout_predict


app = FastAPI(title="GeoVision API", description="Probabilistic Subsurface Mineral Intelligence Platform")

# CORS middleware to allow react frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and data globally
model = None
seismic = None
x_input = None

@app.on_event("startup")
def load_resources():
    global model, seismic, x_input
    model = MineralCNN(input_size=30, depth_out=20)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "mineral_model.pth")
    seismic_path = os.path.join(base_dir, "seismic_surface.npy")

    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    seismic = np.load(seismic_path).astype(np.float32)
    x_input = torch.tensor(seismic[np.newaxis, np.newaxis, :, :], dtype=torch.float32)

class AnalyzeRequest(BaseModel):
    k_sites: int = 5
    uncertainty_penalty: float = 0.30
    n_mc: int = 50
    depth_slice: int = 10
    threshold: float = 0.25
    mineral_price: float = 50000.0
    drill_cost: float = 15000.0

# Simple session cache
analysis_cache = {}

@app.post("/api/analyze")
def run_analysis(req: AnalyzeRequest):
    global model, x_input
    
    # Check cache
    cache_key = f"{req.k_sites}_{req.uncertainty_penalty}_{req.n_mc}_{req.depth_slice}_{req.threshold}_{req.mineral_price}_{req.drill_cost}"
    if cache_key in analysis_cache:
        return analysis_cache[cache_key]
    
    # Run prediction
    mean_vol, uncert_vol = mc_dropout_predict(model, x_input, n_samples=req.n_mc)
    
    mean_vol = mean_vol[0]
    uncert_vol = uncert_vol[0]
    
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
    mean_slice = mean_vol[:, :, req.depth_slice].tolist()
    uncert_slice = uncert_vol[:, :, req.depth_slice].tolist()
    
    # 3D scatter data using threshold
    mask = mean_vol > req.threshold
    xg, yg, zg = np.mgrid[0:30, 0:30, 0:20]
    
    xg_filt = xg[mask].tolist()
    yg_filt = yg[mask].tolist()
    zg_filt = zg[mask].tolist()
    prob_filt = mean_vol[mask].tolist()
    
    resp = {
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
    
    analysis_cache[cache_key] = resp
    return resp
