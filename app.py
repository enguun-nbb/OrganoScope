"""
OrganoScope — web app (Streamlit).  Dark theme · Switzer · fluorescence-glow header.
Modes: single field, or compare groups (the Prism step).
Run:  streamlit run app.py   |   Deploy: https://share.streamlit.io
"""
import os, re, glob, base64
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import streamlit as st
from skimage.io import imread
from skimage.segmentation import find_boundaries
from scipy.stats import kruskal, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from organoscope import analyze_array

AUTHOR = "Enguun Erdenesugar (银姑)"
INSTITUTION = "Tsinghua University"

# ---- palette (dark) ----
BG, BG2, PANEL, LINE = "#07080A", "#0E1013", "#121419", "#262B33"
FG, MUTED, GOLD = "#F3F5F7", "#8B939C", "#D4A62E"
LIVE, DEAD, PARTIAL = "#3DE07E", "#FF5247", "#F3B23E"
GRADE = [(78, LIVE, "alive / resistant"), (60, PARTIAL, "partial"), (0, DEAD, "responding / killed")]


st.set_page_config(page_title="OrganoScope", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://api.fontshare.com/v2/css?f%5B%5D=switzer@400,500,600,700,800&display=swap');
html, body, [class*="css"], .stMarkdown, p, div, span, label, input, button {{
  font-family:'Switzer',system-ui,-apple-system,Arial,sans-serif; color:{FG}; }}
.stApp {{ background:{BG}; }}
#MainMenu, footer, header {{ visibility:hidden; }}
header {{ background:transparent!important; }}
/* keep the open-sidebar arrow visible after the sidebar is collapsed */
[data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] *, [data-testid="stExpandSidebarButton"] * {{ visibility:visible!important; color:{FG}!important; }}
.block-container {{ padding-top:0; max-width:1120px; }}

/* ---- glow hero ---- */
.hero {{ position:relative; margin:0 0 20px; padding:40px 0 26px; border-bottom:1px solid {LINE}; }}
.wm {{ font-weight:800; font-size:clamp(2.4rem,6vw,3.6rem); line-height:1; letter-spacing:-.02em; color:#fff; }}
.tag {{ font-weight:600; letter-spacing:.2em; text-transform:uppercase; font-size:.72rem; color:{LIVE}; margin-top:12px; }}
.blurb {{ color:#C9CED4; font-size:1.02rem; max-width:54ch; margin-top:12px; line-height:1.55; }}

.stats {{ display:flex; gap:40px; flex-wrap:wrap; align-items:baseline; margin:4px 0 2px; }}
.stat .n {{ font-weight:800; font-size:2rem; color:#fff; font-variant-numeric:tabular-nums; }}
.stat .n small {{ font-size:1rem; color:{MUTED}; }}
.stat .k {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.13em; color:{MUTED}; font-weight:600; margin-top:2px; }}
.sec {{ font-weight:700; font-size:1.18rem; color:#fff; margin:26px 0 6px; padding-top:16px; border-top:1px solid {LINE}; }}
.cap {{ font-size:.82rem; color:{MUTED}; margin:6px 0 10px; }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; font-size:.84rem; color:#C9CED4; margin:6px 0 10px; }}
.legend span{{ display:inline-flex; align-items:center; gap:7px; }}
.dot{{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
.note {{ color:#C9CED4; font-size:.92rem; line-height:1.5; }}
.k-em {{ font-weight:800; color:#fff; }}

.stDownloadButton>button, .stButton>button {{ background:transparent; border:1px solid {LINE}; border-radius:5px; font-weight:600; }}
.stDownloadButton>button p, .stButton>button p {{ color:{FG}!important; }}
.stDownloadButton>button:hover, .stButton>button:hover {{ border-color:{LIVE}; }}
.stDownloadButton>button:hover p, .stButton>button:hover p {{ color:{LIVE}!important; }}
[data-testid="stSidebar"] {{ background:{BG2}; border-right:1px solid {LINE}; }}
[data-testid="stFileUploaderDropzone"] {{ background:{PANEL}!important; border:1px dashed {LINE}!important; }}
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small {{ color:{FG}!important; }}
[data-testid="stFileUploaderDropzone"] button {{ background:{LINE}!important; border:none!important; }}
[data-testid="stFileUploaderDropzone"] button p {{ color:{FG}!important; }}
[data-testid="stElementToolbar"] {{ background:transparent; }}
[data-testid="stElementToolbar"] button {{ color:{FG}!important; background:{PANEL}!important; border:1px solid {LINE}!important; border-radius:4px; }}
.foot {{ color:{MUTED}; font-size:.78rem; margin-top:30px; border-top:1px solid {LINE}; padding-top:12px; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <div class="wm">Organoscope</div>
  <div class="tag">Per-organoid read-out of the whole field</div>
  <div class="blurb">Every organoid in a microscopy field, measured on its own — live/dead and
  morphology — instead of a single average per image.</div>
</div>
""", unsafe_allow_html=True)

# ---------------- helpers ----------------
def grade_color(live):
    for thr, col, _ in GRADE:
        if live >= thr:
            return col
    return DEAD

def overlay_figure(img, labels, df):
    fig, ax = plt.subplots(figsize=(7, 7)); fig.patch.set_facecolor(BG)
    ax.imshow(img)
    for _, o in df.iterrows():
        col = grade_color(o["live_pct"])
        ys, xs = np.where(find_boundaries(labels == o["organoid_id"], mode="inner"))
        ax.scatter(xs, ys, s=1.3, c=col, marker=".", linewidths=0)
        ax.text(o["centroid_x"], o["centroid_y"], f"{o['live_pct']:.0f}", color=col,
                fontsize=9.5, ha="center", va="center", fontweight="bold", family="monospace",
                bbox=dict(boxstyle="square,pad=0.18", fc="#0A0C0E", ec=col, lw=0.8))
    ax.axis("off"); fig.tight_layout()
    return fig

def dark_ax(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]: ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED); ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)

def guess_group(fname):
    stem = re.sub(r"\.(png|tif|tiff|jpg|jpeg)$", "", str(fname), flags=re.I)
    toks = re.split(r"[_\-\s]+", stem)
    for t in toks:
        if t.upper() == "CTRL": return "CTRL"
        if re.fullmatch(r"g?\d+", t, flags=re.I): return t.lower()
    return toks[-1] if toks else stem

def group_key(g):
    g = str(g)
    if g.upper() == "CTRL": return (0, 0)
    m = re.search(r"(\d+)", g)
    return (1, int(m.group(1))) if m else (2, g)

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("<div class='sec' style='border:none;margin-top:0;padding-top:0'>Mode</div>", unsafe_allow_html=True)
    mode = st.radio("Mode", ["Single field", "Compare groups"], label_visibility="collapsed")
    st.markdown("<div class='sec'>Settings</div>", unsafe_allow_html=True)
    sep = st.slider("Organoid separation", 10, 35, 18,
                    help="Lower separates touching organoids more; raise it if one splits in two.")
    st.markdown("<div class='note' style='color:#8B939C;margin-top:10px'>Live/dead from a merged RGB "
                "image is approximate. Research prototype, not a clinical tool.</div>", unsafe_allow_html=True)

# ============================================================ SINGLE FIELD
if mode == "Single field":
    with st.sidebar:
        up = st.file_uploader("Load image", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if up is not None:
        img, name = imread(up), up.name
    else:
        sample = next((p for p in ["data/tic_field_01.png", "data/fields/field_2241.tif"] if os.path.exists(p)), None)
        if sample is None:
            st.markdown("<div class='note'>Load a microscopy image in the sidebar to begin.</div>", unsafe_allow_html=True)
            st.stop()
        img, name = imread(sample), "sample"

    df, labels = analyze_array(img, source=name, split_min_distance=sep)
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
                    + "".join(f"<span><i class='dot' style='background:{c}'></i>{t}</span>" for _, c, t in GRADE)
                    + "</div>", unsafe_allow_html=True)
        st.pyplot(overlay_figure(img[:, :, :3] if img.ndim == 3 else img, labels, df))
        st.markdown("<div class='cap'>Each organoid labelled with its live %. Numbers are per-organoid, not a field average.</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='sec'>Per-organoid viability</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.3, 4.3)); fig.patch.set_alpha(0); ax.set_facecolor("none")
        m, sd = lives.mean(), lives.std()
        ax.bar(0, m, width=0.52, color="#FFFFFF", alpha=.10, edgecolor="#FFFFFF", linewidth=1.2, zorder=1)
        ax.errorbar(0, m, sd, color="#C9CED4", capsize=7, elinewidth=1.3, capthick=1.3, zorder=2)
        xj = np.random.default_rng(0).uniform(-0.13, 0.13, len(lives))
        ax.scatter(xj, lives, c=[grade_color(v) for v in lives], s=75, edgecolor=BG, linewidth=.9, zorder=3)
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, 100)
        ax.set_xticks([0]); ax.set_xticklabels([str(name)[:16]]); ax.set_ylabel("live %")
        dark_ax(fig.tight_layout() or ax); st.pyplot(fig)
        st.markdown(f"<div class='note'><span class='k-em'>{(lives<60).sum()}</span> organoid(s) below 60% "
                    f"live — the responders. The field average (<span class='k-em'>{lives.mean():.0f}%</span>) "
                    f"hides a <span class='k-em'>{lives.max()-lives.min():.0f}-point</span> spread.</div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='sec'>Per-organoid table</div>", unsafe_allow_html=True)
    show = df.drop(columns=["centroid_x", "centroid_y"]).sort_values("live_pct", ascending=False)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("Export table (CSV)", show.to_csv(index=False).encode(),
                       file_name="organoid_features.csv", mime="text/csv")

# ============================================================ COMPARE GROUPS
else:
    with st.sidebar:
        ups = st.file_uploader("Load images (several)", accept_multiple_files=True,
                               type=["tif", "tiff", "png", "jpg", "jpeg"])
        use_ex = st.checkbox("Use bundled example (24 images)",
                             value=not ups and len(glob.glob("data/groups/*")) > 0)
    sources = []
    if ups:
        for f in ups: sources.append((f.name, imread(f)))
    elif use_ex:
        for p in sorted(glob.glob("data/groups/*")): sources.append((os.path.basename(p), imread(p)))

    if not sources:
        st.markdown("<div class='note'>Upload several images — one condition per image. Put the condition in the "
                    "filename (<b>CTRL_a.png</b>, <b>g1_a.png</b>, <b>g7_b.png</b>) and OrganoScope groups them "
                    "automatically.</div>", unsafe_allow_html=True)
        st.stop()

    guess = pd.DataFrame({"image": [n for n, _ in sources], "group": [guess_group(n) for n, _ in sources]})
    st.markdown("<div class='sec'>Assign groups</div>", unsafe_allow_html=True)
    edited = st.data_editor(guess, hide_index=True, use_container_width=True, disabled=["image"], key="grp")
    gmap = dict(zip(edited["image"], edited["group"]))

    rows = []; prog = st.progress(0.0, text="Analysing organoids…")
    for i, (n, im) in enumerate(sources):
        d, _ = analyze_array(im, source=n, split_min_distance=sep)
        d["group"] = str(gmap.get(n, guess_group(n))); rows.append(d)
        prog.progress((i + 1) / len(sources), text=f"Analysing… {i+1}/{len(sources)}")
    prog.empty()
    D = pd.concat(rows, ignore_index=True)
    if len(D) == 0:
        st.markdown("<div class='note'>No organoids detected.</div>", unsafe_allow_html=True); st.stop()
    groups = sorted(D["group"].unique(), key=group_key)

    st.markdown(f"""
    <div class="stats">
      <div class="stat"><div class="n">{len(groups)}</div><div class="k">groups</div></div>
      <div class="stat"><div class="n">{len(D)}</div><div class="k">organoids</div></div>
      <div class="stat"><div class="n">{len(sources)}</div><div class="k">images</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec'>Viability by group</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='legend'>"
                + "".join(f"<span><i class='dot' style='background:{c}'></i>{t}</span>" for _, c, t in GRADE)
                + "</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(min(10, 1.1 * len(groups) + 2), 4.6)); fig.patch.set_alpha(0); ax.set_facecolor("none")
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        v = D[D["group"] == g]["live_pct"].values
        ax.bar(i, v.mean(), 0.62, color="#FFFFFF", alpha=.10, edgecolor="#FFFFFF", lw=1.1, zorder=1)
        ax.errorbar(i, v.mean(), v.std(), color="#C9CED4", capsize=5, elinewidth=1.2, capthick=1.2, zorder=2)
        ax.scatter(i + rng.uniform(-.14, .14, len(v)), v, c=[grade_color(x) for x in v], s=45, edgecolor=BG, lw=.7, zorder=3)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups); ax.set_ylim(0, 105); ax.set_ylabel("live %  (viability)")
    dark_ax(ax); fig.tight_layout(); st.pyplot(fig)

    if len(groups) >= 2:
        try:
            if len(groups) == 2:
                a, b = [D[D["group"] == g]["live_pct"].values for g in groups]
                stat, p = mannwhitneyu(a, b); test = "Mann–Whitney"
            else:
                stat, p = kruskal(*[D[D["group"] == g]["live_pct"].values for g in groups]); test = "Kruskal–Wallis"
            verdict = ("a <b>significant</b> difference between groups" if p < 0.05
                       else "<b>no significant</b> difference between groups")
            st.markdown(f"<div class='note'>{test} p = <span class='k-em'>{p:.3g}</span> — {verdict}. "
                        f"<span style='color:{MUTED}'>(p &lt; 0.05 = real difference.)</span></div>", unsafe_allow_html=True)
        except Exception:
            pass

    st.markdown("<div class='sec'>Group summary</div>", unsafe_allow_html=True)
    summ = (D.groupby("group")["live_pct"].agg(organoids="count", mean_live="mean", sd="std")
            .reindex(groups).round(1).reset_index())
    st.dataframe(summ, use_container_width=True, hide_index=True)
    st.download_button("Export per-organoid data (CSV)", D.to_csv(index=False).encode(),
                       file_name="organoscope_groups.csv", mime="text/csv")

st.markdown(f"<div class='foot'>OrganoScope · {AUTHOR}, {INSTITUTION}</div>", unsafe_allow_html=True)
