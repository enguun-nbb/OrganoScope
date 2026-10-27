"""
OrganoScope — web app (Streamlit).
Upload an organoid microscopy image → per-organoid live/dead + morphology.

Run locally:   streamlit run app.py
Deploy free:   push this repo to GitHub, then share it on https://streamlit.io/cloud
"""
import io, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import streamlit as st
from skimage.io import imread
from skimage.segmentation import find_boundaries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from organoscope import analyze_array

st.set_page_config(page_title="OrganoScope", page_icon="🔬", layout="wide")

GRADE = [(78, "#2E9E5B", "Alive / resistant"),
         (60, "#E0A82E", "Partial"),
         (0,  "#D9533B", "Responding / killed")]

def grade_color(live):
    for thr, col, _ in GRADE:
        if live >= thr:
            return col
    return "#D9533B"

def overlay_figure(img, labels, df):
    """Original image with each organoid outlined and labeled by live %."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    for _, o in df.iterrows():
        col = grade_color(o["live_pct"])
        b = find_boundaries(labels == o["organoid_id"], mode="inner")
        ys, xs = np.where(b)
        ax.scatter(xs, ys, s=1, c=col, marker=".", linewidths=0)
        ax.text(o["centroid_x"], o["centroid_y"], f"{o['live_pct']:.0f}%",
                color="black", fontsize=9, ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="white"))
    ax.axis("off")
    fig.tight_layout()
    return fig

# ---------------- UI ----------------
st.title("🔬 OrganoScope")
st.caption("Per-organoid analysis of organoid microscopy. Each organoid is measured "
           "individually — size, shape, and live/dead — instead of one average per image.")

with st.sidebar:
    st.header("Image")
    up = st.file_uploader("Upload a microscopy image", type=["tif", "tiff", "png", "jpg", "jpeg"])
    st.caption("Fluorescence (green = live, red = dead) works best. "
               "A bundled sample loads if you don't upload one.")
    st.divider()
    st.caption("Note: live/dead from a merged RGB image is approximate. "
               "This is a research prototype, not a clinical tool.")

# load image: upload, else bundled sample, else prompt
import os
if up is not None:
    img = imread(up)
    name = up.name
else:
    sample = next((p for p in ["data/tic_field_01.png", "data/fields/field_2241.tif",
                               "sample.png"] if os.path.exists(p)), None)
    if sample is None:
        st.info("👈 Upload an organoid microscopy image in the sidebar to begin.")
        st.stop()
    img = imread(sample)
    name = "bundled sample"
    st.info("Showing a bundled sample. Upload your own image in the sidebar.")

df, labels = analyze_array(img, source=name)

if len(df) == 0:
    st.warning("No organoids detected — try a clearer fluorescence image.")
    st.stop()

# summary metrics
lives = df["live_pct"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Organoids", len(df))
c2.metric("Mean live %", f"{lives.mean():.0f}%")
c3.metric("Field Ri (dead)", f"{1 - lives.mean()/100:.2f}")
c4.metric("Live % range", f"{lives.min():.0f}–{lives.max():.0f}%")

left, right = st.columns([3, 2])
with left:
    st.subheader("Per-organoid map")
    st.pyplot(overlay_figure(img[:, :, :3] if img.ndim == 3 else img, labels, df))
with right:
    st.subheader("Viability distribution")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(lives, bins=np.arange(0, 101, 10), color="#5B2A86", edgecolor="white")
    ax.set_xlabel("live %"); ax.set_ylabel("organoids")
    fig.tight_layout(); st.pyplot(fig)
    st.markdown(
        f"**{(lives < 60).sum()}** organoid(s) below 60% live — the responders. "
        f"The field average ({lives.mean():.0f}%) hides a "
        f"{lives.max()-lives.min():.0f}-point spread.")

st.subheader("Per-organoid table")
show = df.drop(columns=["centroid_x", "centroid_y"]).sort_values("live_pct", ascending=False)
st.dataframe(show, use_container_width=True, hide_index=True)
st.download_button("Download table (CSV)", show.to_csv(index=False).encode(),
                   file_name="organoid_features.csv", mime="text/csv")
