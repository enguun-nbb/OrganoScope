"""
OrganoScope — web app (Streamlit).  Navy editorial ("data-journal") theme.
Run locally:  streamlit run app.py   |   Deploy: https://share.streamlit.io
"""
import os
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

# ---- edit to put your own name on it ----
AUTHOR = "Enguun Erdenesugar"
INSTITUTION = "Tsinghua University"

# ---- palette ----
CREAM, PANEL, RULE = "#F7F4EC", "#EFEBDE", "#DCD5C2"
NAVY, INK, MUTED, GOLD = "#0A2342", "#14233A", "#6B7685", "#B8902C"
LIVE, DEAD, PARTIAL = "#1F9E5A", "#D6452F", "#D99A1F"          # data colors on cream
LIVE_I, DEAD_I, PARTIAL_I = "#3DE07E", "#FF5247", "#F3B23E"    # brighter, for the black plate
GRADE = [(78, LIVE_I, "alive / resistant"), (60, PARTIAL_I, "partial"), (0, DEAD_I, "responding / killed")]
LEG = [(LIVE, "alive / resistant"), (PARTIAL, "partial"), (DEAD, "responding / killed")]

st.set_page_config(page_title="OrganoScope", page_icon="◆", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Libre+Franklin:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, div, span, label {{ font-family:'Libre Franklin',system-ui,sans-serif; color:{INK}; }}
.stApp {{ background:{CREAM}; }}
#MainMenu, footer, header {{ visibility:hidden; }}
.block-container {{ padding-top:1.6rem; max-width:1080px; }}

.wordmark {{ font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:clamp(2.6rem,7vw,4rem);
  color:{NAVY}; line-height:.95; letter-spacing:-.015em; }}
.tagline {{ text-transform:uppercase; letter-spacing:.24em; font-size:.72rem; color:{NAVY};
  font-weight:600; margin-top:12px; }}
.blurb {{ font-size:1.08rem; color:{INK}; max-width:60ch; margin-top:12px; line-height:1.55; }}
.hr {{ border:0; border-top:1px solid {RULE}; margin:22px 0; }}
.goldline {{ height:3px; width:54px; background:{GOLD}; margin:14px 0 0; }}

.stats {{ display:flex; gap:40px; flex-wrap:wrap; align-items:baseline; }}
.stat .n {{ font-family:'Fraunces',serif; font-weight:600; font-size:2.1rem; color:{NAVY}; font-variant-numeric:tabular-nums; }}
.stat .n small {{ font-size:1rem; color:{MUTED}; }}
.stat .k {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.14em; color:{MUTED}; margin-top:2px; }}

.sec {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.2rem; color:{NAVY}; margin:30px 0 4px; }}
.cap {{ font-size:.82rem; color:{MUTED}; font-style:italic; margin:6px 0 10px; }}
.legend {{ display:flex; gap:20px; flex-wrap:wrap; font-size:.84rem; color:{INK}; margin:6px 0 10px; }}
.legend span{{ display:inline-flex; align-items:center; gap:7px; }}
.dot{{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
.note {{ color:{INK}; font-size:.92rem; line-height:1.5; }}
.k-em {{ font-family:'Fraunces',serif; font-weight:600; color:{NAVY}; }}

.stDownloadButton>button, .stButton>button {{
  background:{NAVY}; border:none; border-radius:3px; font-weight:600; letter-spacing:.01em; }}
.stDownloadButton>button p, .stButton>button p,
.stDownloadButton>button div, .stButton>button div {{ color:{CREAM}!important; }}
.stDownloadButton>button:hover, .stButton>button:hover {{ background:{GOLD}; }}
.stDownloadButton>button:hover p, .stButton>button:hover p {{ color:{NAVY}!important; }}
[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {RULE}; }}
.foot {{ color:{MUTED}; font-size:.78rem; margin-top:34px; border-top:1px solid {RULE}; padding-top:12px; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="wordmark">organoscope</div>
<div class="tagline">per-organoid read-out of the whole field</div>
<div class="goldline"></div>
<div class="blurb">Every organoid in a microscopy field, measured on its own — live/dead and
morphology — instead of a single average per image.</div>
<hr class="hr">
""", unsafe_allow_html=True)

def grade_color(live):
    for thr, col, _ in GRADE:
        if live >= thr:
            return col
    return DEAD_I

def overlay_figure(img, labels, df):
    fig, ax = plt.subplots(figsize=(7, 7)); fig.patch.set_facecolor(CREAM)
    ax.imshow(img)
    for _, o in df.iterrows():
        col = grade_color(o["live_pct"])
        ys, xs = np.where(find_boundaries(labels == o["organoid_id"], mode="inner"))
        ax.scatter(xs, ys, s=1.3, c=col, marker=".", linewidths=0)
        ax.text(o["centroid_x"], o["centroid_y"], f"{o['live_pct']:.0f}", color="#0A0C0E",
                fontsize=9, ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc=col, ec="white", lw=0.6))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():                      # navy frame -> journal plate
        s.set_color(NAVY); s.set_linewidth(2)
    fig.tight_layout()
    return fig

with st.sidebar:
    st.markdown(f"<div class='sec' style='margin-top:0'>Specimen</div>", unsafe_allow_html=True)
    up = st.file_uploader("Load image", type=["tif", "tiff", "png", "jpg", "jpeg"])
    st.markdown("<div class='note'>Fluorescence (green = live, red = dead) works best. "
                "A sample loads if nothing is uploaded.</div>", unsafe_allow_html=True)
    st.markdown("<div class='note' style='margin-top:14px;color:#6B7685'>Live/dead from a merged RGB "
                "image is approximate. Research prototype, not a clinical tool.</div>", unsafe_allow_html=True)

if up is not None:
    img, name = imread(up), up.name
else:
    sample = next((p for p in ["data/tic_field_01.png", "data/fields/field_2241.tif"] if os.path.exists(p)), None)
    if sample is None:
        st.markdown("<div class='note'>Load a microscopy image in the sidebar to begin.</div>", unsafe_allow_html=True)
        st.stop()
    img, name = imread(sample), "sample"

df, labels = analyze_array(img, source=name)
if len(df) == 0:
    st.markdown("<div class='note'>No organoids detected — try a clearer fluorescence image.</div>", unsafe_allow_html=True)
    st.stop()

lives = df["live_pct"]
st.markdown(f"""
<div class="stats">
  <div class="stat"><div class="n">{len(df)}</div><div class="k">organoids</div></div>
  <div class="stat"><div class="n">{lives.mean():.0f}<small>%</small></div><div class="k">mean live</div></div>
  <div class="stat"><div class="n">{1-lives.mean()/100:.2f}</div><div class="k">field Ri</div></div>
  <div class="stat"><div class="n">{lives.min():.0f}–{lives.max():.0f}<small>%</small></div><div class="k">live range</div></div>
</div>""", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown("<div class='sec'>Per-organoid map</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='legend'>"
                + "".join(f"<span><i class='dot' style='background:{c}'></i>{t}</span>" for c, t in LEG)
                + "</div>", unsafe_allow_html=True)
    st.pyplot(overlay_figure(img[:, :, :3] if img.ndim == 3 else img, labels, df))
    st.markdown("<div class='cap'>Fig. 1 — each organoid labelled with its live %. "
                "Numbers are per-organoid, not a field average.</div>", unsafe_allow_html=True)
with right:
    st.markdown("<div class='sec'>Viability distribution</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 3.5)); fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.hist(lives, bins=np.arange(0, 101, 10), color=NAVY, edgecolor=CREAM)
    ax.set_xlabel("live %", color=MUTED); ax.set_ylabel("organoids", color=MUTED)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]: ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED)
    fig.tight_layout(); st.pyplot(fig)
    st.markdown(f"<div class='note'><span class='k-em'>{(lives<60).sum()}</span> organoid(s) below 60% "
                f"live — the responders. The field average (<span class='k-em'>{lives.mean():.0f}%</span>) "
                f"hides a <span class='k-em'>{lives.max()-lives.min():.0f}-point</span> spread.</div>",
                unsafe_allow_html=True)

st.markdown("<div class='sec'>Per-organoid table</div>", unsafe_allow_html=True)
show = df.drop(columns=["centroid_x", "centroid_y"]).sort_values("live_pct", ascending=False)
st.dataframe(show, use_container_width=True, hide_index=True)
st.download_button("Export table (CSV)", show.to_csv(index=False).encode(),
                   file_name="organoid_features.csv", mime="text/csv")

st.markdown(f"<div class='foot'>OrganoScope · {AUTHOR}, {INSTITUTION}</div>", unsafe_allow_html=True)
