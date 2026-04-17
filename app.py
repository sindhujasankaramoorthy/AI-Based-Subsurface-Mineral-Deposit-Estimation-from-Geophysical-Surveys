import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import folium
from streamlit_folium import st_folium
from fpdf import FPDF
from scipy.ndimage import gaussian_filter
from model import MineralCNN, mc_dropout_predict

# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="GeoVision AI",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════
# PREMIUM CSS
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.main { background-color: #0E1117; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 95%; }

h1 { font-size: 2rem !important; font-weight: 700 !important; }
h2 { font-size: 1.4rem !important; font-weight: 600 !important; color: #00d2ff !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(0,210,255,0.08), rgba(0,210,255,0.02));
    border: 1px solid rgba(0,210,255,0.15);
    padding: 16px;
    border-radius: 12px;
}
section[data-testid="stSidebar"] { background-color: #161A23; }
.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
    background: linear-gradient(135deg, #00d2ff, #0090ff);
    color: white;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00e5ff, #00a0ff);
    transform: translateY(-1px);
}
.stButton > button:disabled {
    background: #333;
    color: #666;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ═══════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    model = MineralCNN(input_size=30, depth_out=20)
    model.load_state_dict(torch.load("mineral_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ═══════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════
if "target_coords" not in st.session_state:
    st.session_state.target_coords = None

# ═══════════════════════════════════════════════════════
# SIDEBAR — All Controls Organized
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# ⛏️ GeoVision AI")
    st.caption("Mineral Intelligence Platform")
    st.markdown("---")

    with st.expander("🔬 Model Parameters", expanded=True):
        n_mc = st.slider("MC Dropout Samples", 10, 100, 50)
        uncertainty_penalty = st.slider("Uncertainty Penalty", 0.0, 1.0, 0.30)

    with st.expander("🎯 Targeting", expanded=True):
        k_sites = st.slider("Top-K Drill Sites", 1, 10, 5)
        depth_slice = st.slider("Depth Slice View", 0, 19, 10)
        threshold = st.slider("3D Ore Threshold", 0.10, 0.90, 0.25)

    with st.expander("💰 Financial Inputs", expanded=False):
        mineral_price = st.number_input("Commodity Spot Price (₹)", 10000, 1000000, 50000, step=5000)
        drill_cost = st.number_input("Drill Mobilization Cost (₹)", 5000, 200000, 15000, step=1000)

    with st.expander("📂 Data Source", expanded=False):
        uploaded_file = st.file_uploader("Upload Seismic .npy", type=["npy"])
        use_demo = st.checkbox("Use Demo Data", value=True)

# ═══════════════════════════════════════════════════════
# LOAD INPUT DATA
# ═══════════════════════════════════════════════════════
if uploaded_file is not None and not use_demo:
    seismic = np.load(uploaded_file).astype(np.float32)
else:
    seismic = np.load("seismic_surface.npy").astype(np.float32)

x_input = torch.tensor(seismic[np.newaxis, np.newaxis, :, :], dtype=torch.float32)

# ═══════════════════════════════════════════════════════
# MAIN CONTENT — HEADER
# ═══════════════════════════════════════════════════════
st.markdown("# ⛏️ GeoVision AI")
st.caption("AI-powered seismic inversion • uncertainty estimation • optimal drill targeting")

# ═══════════════════════════════════════════════════════
# STEP 1 — SELECT LOCATION
# ═══════════════════════════════════════════════════════
st.markdown("## 🌍 Step 1: Select Exploration Region")
col_map, col_info = st.columns([3, 1])

with col_map:
    m = folium.Map(
        location=[-23.5, 120.5],
        zoom_start=4,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri"
    )
    if st.session_state.target_coords:
        c = st.session_state.target_coords
        folium.Marker(
            [c["lat"], c["lng"]],
            popup="🎯 Target",
            icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
        ).add_to(m)

    map_data = st_folium(m, height=350, use_container_width=True, returned_objects=["last_clicked"])

if map_data and map_data.get("last_clicked"):
    st.session_state.target_coords = {"lat": map_data["last_clicked"]["lat"], "lng": map_data["last_clicked"]["lng"]}
    st.rerun()

with col_info:
    st.markdown("#### 📍 Target")
    if st.session_state.target_coords:
        c = st.session_state.target_coords
        st.metric("Latitude", f"{c['lat']:.4f}")
        st.metric("Longitude", f"{c['lng']:.4f}")
        st.success("Region locked. Ready to analyze.")
    else:
        st.warning("Click on the map to select a target region.")

st.markdown("---")

# ═══════════════════════════════════════════════════════
# STEP 2 — RUN ANALYSIS
# ═══════════════════════════════════════════════════════
st.markdown("## 🚀 Step 2: Run AI Exploration")
button_disabled = st.session_state.target_coords is None
if st.button("⚡ Launch Probabilistic Analysis", disabled=button_disabled, use_container_width=True):

    progress = st.progress(0)
    with st.spinner("Running Monte Carlo dropout inference..."):
        for i in range(100):
            progress.progress(i + 1)
        mean_vol, uncert_vol = mc_dropout_predict(model, x_input, n_samples=n_mc)

        # Saliency extraction
        model.eval()
        x_input.requires_grad = True
        out = model(x_input)
        out.sum().backward()
        raw_sal = x_input.grad.abs().squeeze().numpy()
        smoothed = gaussian_filter(raw_sal, sigma=1.5)
        saliency_map = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
        x_input.requires_grad = False

    progress.empty()
    mean_vol = mean_vol[0]
    uncert_vol = uncert_vol[0]

    # ───────────────────────────────────────────────
    # SCORING & FINANCIALS
    # ───────────────────────────────────────────────
    scores = mean_vol.mean(axis=2) - uncertainty_penalty * uncert_vol.mean(axis=2)
    flat_indices = np.argsort(scores.ravel())[::-1][:k_sites]

    sites_data = []
    total_net_profit = 0.0
    for rank, idx in enumerate(flat_indices):
        x, y = idx // 30, idx % 30
        prob = mean_vol[x, y, :].mean()
        uncert = uncert_vol[x, y, :].mean()
        econ = prob * (1 - uncert)
        revenue = econ * mineral_price
        net_profit = revenue - drill_cost
        if net_profit > 0:
            total_net_profit += net_profit
        sites_data.append({
            "Rank": rank + 1, "X": x, "Y": y,
            "Probability": round(float(prob), 3),
            "Uncertainty": round(float(uncert), 3),
            "Econ Score": round(float(econ), 3),
            "Revenue (₹)": f"₹{revenue:,.0f}",
            "Net Profit (₹)": f"₹{net_profit:,.0f}"
        })
    df = pd.DataFrame(sites_data)

    # ═══════════════════════════════════════════════
    # STEP 3 — RESULTS
    # ═══════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 📈 Step 3: Results")

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Top Score", f"{scores.max():.3f}")
    k2.metric("Confidence", f"{(1 - uncert_vol.mean()) * 100:.1f}%")
    k3.metric("Projected ROI", f"₹{total_net_profit:,.0f}")
    k4.metric("Sites Found", k_sites)
    c = st.session_state.target_coords
    k5.metric("Region", f"{c['lat']:.1f}°, {c['lng']:.1f}°")

    st.markdown("")

    # ───────────────────────────────────────────────
    # ORGANIZED TABS
    # ───────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Probability",
        "🌍 3D Ore Body",
        "🎯 Drill Targets",
        "🧠 AI Reasoning",
        "📄 Reports"
    ])

    # TAB 1 — Probability & Uncertainty Maps
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = go.Figure(data=go.Heatmap(z=mean_vol[:, :, depth_slice], colorscale="Hot"))
            fig1.update_layout(title=f"Mineral Probability · Depth {depth_slice}", height=420,
                               margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = go.Figure(data=go.Heatmap(z=uncert_vol[:, :, depth_slice], colorscale="Blues"))
            fig2.update_layout(title=f"Uncertainty · Depth {depth_slice}", height=420,
                               margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    # TAB 2 — 3D Ore Body
    with tab2:
        xg, yg, zg = np.mgrid[0:30, 0:30, 0:20]
        mask = mean_vol > threshold
        fig3d = go.Figure(data=go.Scatter3d(
            x=xg[mask], y=yg[mask], z=zg[mask], mode="markers",
            marker=dict(size=3, color=mean_vol[mask], colorscale="Hot",
                        opacity=0.65, colorbar=dict(title="Prob"))
        ))
        fig3d.update_layout(height=600, title="3D Predicted Ore Body",
                            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Depth"))
        st.plotly_chart(fig3d, use_container_width=True)

    # TAB 3 — Drill Targets Table
    with tab3:
        st.dataframe(df, use_container_width=True, hide_index=True)

    # TAB 4 — AI Explainability
    with tab4:
        st.caption("Gradient-based feature attribution showing which seismic anomalies influenced the CNN's predictions.")
        e1, e2, e3 = st.columns(3)
        with e1:
            fig_r = go.Figure(data=go.Heatmap(z=seismic, colorscale="Greys"))
            fig_r.update_layout(title="Raw Seismic", height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_r, use_container_width=True)
        with e2:
            fig_s = go.Figure(data=go.Heatmap(z=saliency_map, colorscale="Inferno"))
            fig_s.update_layout(title="AI Attention", height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_s, use_container_width=True)
        with e3:
            overlay = seismic.copy()
            overlay[saliency_map < 0.35] = overlay.min()
            fig_o = go.Figure(data=go.Heatmap(z=overlay, colorscale="Viridis"))
            fig_o.update_layout(title="Structural Overlay", height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_o, use_container_width=True)

    # TAB 5 — Download Reports
    with tab5:
        st.markdown("#### Export Your Analysis")
        st.caption("Download drill targets as CSV or a formatted executive PDF report.")

        def create_pdf(dataframe, total_profit, lat, lng):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(200, 15, "GeoVision AI - Executive Report", ln=True, align="C")
            pdf.set_font("Arial", '', 12)
            pdf.cell(200, 8, f"Target: Lat {lat:.4f}, Lng {lng:.4f}", ln=True)
            pdf.cell(200, 8, f"Projected NPV: INR {total_profit:,.2f}", ln=True)
            pdf.cell(200, 8, f"Optimal Sites: {len(dataframe)}", ln=True)
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 10)
            cols = [15, 15, 15, 28, 28, 40, 40]
            heads = ["Rank", "X", "Y", "Uncert", "Score", "Revenue", "Profit"]
            for h, w in zip(heads, cols):
                pdf.cell(w, 10, h, border=1, align="C")
            pdf.ln()
            pdf.set_font("Arial", '', 9)
            for _, row in dataframe.iterrows():
                vals = [str(row["Rank"]), str(row["X"]), str(row["Y"]),
                        str(row["Uncertainty"]), str(row["Econ Score"]),
                        str(row["Revenue (₹)"]).replace("₹", "INR "),
                        str(row["Net Profit (₹)"]).replace("₹", "INR ")]
                for v, w in zip(vals, cols):
                    pdf.cell(w, 10, v, border=1, align="C")
                pdf.ln()
            return bytes(pdf.output())

        b1, b2 = st.columns(2)
        with b1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📄 Download CSV", data=csv, file_name="drill_targets.csv", mime="text/csv")
        with b2:
            pdf_bytes = create_pdf(df, total_net_profit, c['lat'], c['lng'])
            st.download_button("🖨️ Download PDF Report", data=pdf_bytes, file_name="executive_report.pdf", mime="application/pdf")

    st.success("✅ Analysis complete. High-confidence targets identified.")