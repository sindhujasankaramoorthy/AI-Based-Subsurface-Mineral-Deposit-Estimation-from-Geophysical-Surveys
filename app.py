import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from model import MineralCNN, mc_dropout_predict

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="GeoVision AI",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS (Premium UI)
# ---------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 95%;
}

h1, h2, h3 {
    letter-spacing: 0.3px;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 14px;
}

section[data-testid="stSidebar"] {
    background-color: #161A23;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("# ⛏️ GeoVision AI")
st.caption("Probabilistic Subsurface Mineral Intelligence Platform")
st.info("AI-powered seismic inversion, uncertainty estimation, and optimal drill targeting.")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = MineralCNN(input_size=30, depth_out=20)
    model.load_state_dict(torch.load("mineral_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("⚙️ Exploration Controls")
st.sidebar.caption("Tune model and drilling strategy")

k_sites = st.sidebar.slider("Top-K Drill Sites", 1, 10, 5)
uncertainty_penalty = st.sidebar.slider("Uncertainty Penalty", 0.0, 1.0, 0.30)
n_mc = st.sidebar.slider("MC Dropout Samples", 10, 100, 50)
depth_slice = st.sidebar.slider("Depth Slice", 0, 19, 10)
threshold = st.sidebar.slider("3D Ore Threshold", 0.10, 0.90, 0.25)

# ---------------------------------------------------
# LOAD INPUT DATA
# ---------------------------------------------------
seismic = np.load("seismic_surface.npy").astype(np.float32)
x_input = torch.tensor(seismic[np.newaxis, np.newaxis, :, :], dtype=torch.float32)

# ---------------------------------------------------
# RUN ANALYSIS
# ---------------------------------------------------
if st.button("🚀 Run AI Exploration Analysis"):

    progress = st.progress(0)

    with st.spinner("Running probabilistic inversion..."):
        for i in range(100):
            progress.progress(i + 1)

        mean_vol, uncert_vol = mc_dropout_predict(model, x_input, n_samples=n_mc)

    progress.empty()

    mean_vol = mean_vol[0]
    uncert_vol = uncert_vol[0]

    # ---------------------------------------------------
    # DRILL SITE SCORING
    # ---------------------------------------------------
    scores = mean_vol.mean(axis=2) - uncertainty_penalty * uncert_vol.mean(axis=2)

    flat_indices = np.argsort(scores.ravel())[::-1][:k_sites]

    sites_data = []
    for rank, idx in enumerate(flat_indices):
        x = idx // 30
        y = idx % 30

        prob = mean_vol[x, y, :].mean()
        uncert = uncert_vol[x, y, :].mean()
        econ = prob * (1 - uncert)

        sites_data.append({
            "Rank": rank + 1,
            "Grid X": x,
            "Grid Y": y,
            "Avg Probability": round(float(prob), 3),
            "Uncertainty": round(float(uncert), 3),
            "Economic Score": round(float(econ), 3)
        })

    df = pd.DataFrame(sites_data)

    # ---------------------------------------------------
    # KPI METRICS
    # ---------------------------------------------------
    st.markdown("## 📈 Executive Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Top Score", f"{scores.max():.3f}")
    c2.metric("Mean Confidence", f"{(1 - uncert_vol.mean()) * 100:.1f}%")
    c3.metric("High Value Voxels", int((mean_vol > 0.6).sum()))
    c4.metric("Recommended Sites", k_sites)

    # ---------------------------------------------------
    # TABS
    # ---------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "📊 Probability Maps",
        "🌍 3D Ore Body",
        "🎯 Drill Targets"
    ])

    # ---------------------------------------------------
    # TAB 1 - MAPS
    # ---------------------------------------------------
    with tab1:

        col1, col2 = st.columns(2)

        with col1:
            fig1 = go.Figure(
                data=go.Heatmap(
                    z=mean_vol[:, :, depth_slice],
                    colorscale="Hot"
                )
            )

            fig1.update_layout(
                title=f"Mineral Probability at Depth {depth_slice}",
                height=450
            )

            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure(
                data=go.Heatmap(
                    z=uncert_vol[:, :, depth_slice],
                    colorscale="Blues"
                )
            )

            fig2.update_layout(
                title=f"Prediction Uncertainty at Depth {depth_slice}",
                height=450
            )

            st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------------------
    # TAB 2 - 3D ORE BODY
    # ---------------------------------------------------
    with tab2:

        xg, yg, zg = np.mgrid[0:30, 0:30, 0:20]
        mask = mean_vol > threshold

        fig3d = go.Figure(data=go.Scatter3d(
            x=xg[mask],
            y=yg[mask],
            z=zg[mask],
            mode="markers",
            marker=dict(
                size=3,
                color=mean_vol[mask],
                colorscale="Hot",
                opacity=0.65,
                colorbar=dict(title="Probability")
            )
        ))

        fig3d.update_layout(
            height=650,
            title="3D Predicted Ore Body",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Depth"
            )
        )

        st.plotly_chart(fig3d, use_container_width=True)

    # ---------------------------------------------------
    # TAB 3 - DRILL TARGETS
    # ---------------------------------------------------
    with tab3:

        st.subheader("Recommended Drill Coordinates")

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📄 Download Drill Report",
            data=csv,
            file_name="drill_targets.csv",
            mime="text/csv"
        )

    st.success("✅ Analysis Complete. High-confidence targets identified.")