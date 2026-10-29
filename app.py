"""
OrganoScope — web app (Streamlit).  Navy editorial theme.
Two modes: analyse a single field, or compare several groups (the Prism step).
Run locally:  streamlit run app.py   |   Deploy: https://share.streamlit.io
"""
import os, re, glob
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

# ---- edit to put your own name on it ----
AUTHOR = "Enguun Erdenesugar"
INSTITUTION = "Tsinghua University"

# ---- palette ----
CREAM, PANEL, RULE = "#F7F4EC", "#EFEBDE", "#DCD5C2"
NAVY, INK, MUTED, GOLD = "#0A2342", "#14233A", "#6B7685", "#B8902C"
LIVE, DEAD, PARTIAL = "#1F9E5A", "#D6452F", "#D99A1F"
LIVE_I, DEAD_I, PARTIAL_I = "#3DE07E", "#FF5247", "#F3B23E"
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
.wordmark {{ font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:clamp(2.4rem,7vw,3.6rem);
  color:{NAVY}; line-height:.95; letter-spacing:-.015em; }}
.tagline {{ text-transform:uppercase; letter-spacing:.24em; font-size:.72rem; color:{NAVY}; font-weight:600; margin-top:12px; }}
.blurb {{ font-size:1.05rem; color:{INK}; max-width:60ch; margin-top:12px; line-height:1.55; }}
.hr {{ border:0; border-top:1px solid {RULE}; margin:20px 0; }}
.goldline {{ height:3px; width:54px; background:{GOLD}; margin:14px 0 0; }}
.stats {{ display:flex; gap:40px; flex-wrap:wrap; align-items:baseline; }}
.stat .n {{ font-family:'Fraunces',serif; font-weight:600; font-size:2.1rem; color:{NAVY}; font-variant-numeric:tabular-nums; }}
.stat .n small {{ font-size:1rem; color:{MUTED}; }}
.stat .k {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.14em; color:{MUTED}; margin-top:2px; }}
.sec {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.2rem; color:{NAVY}; margin:28px 0 4px; }}
.cap {{ font-size:.82rem; color:{MUTED}; font-style:italic; margin:6px 0 10px; }}
.legend {{ display:flex; gap:20px; flex-wrap:wrap; font-size:.84rem; color:{INK}; margin:6px 0 10px; }}
.legend span{{ display:inline-flex; align-items:center; gap:7px; }}
.dot{{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
.note {{ color:{INK}; font-size:.92rem; line-height:1.5; }}
.k-em {{ font-family:'Fraunces',serif; font-weight:600; color:{NAVY}; }}
.stDownloadButton>button, .stButton>button {{ background:{NAVY}; border:none; border-radius:3px; font-weight:600; letter-spacing:.01em; }}
.stDownloadButton>button p, .stButton>button p, .stDownloadButton>button div, .stButton>button div {{ color:{CREAM}!important; }}
.stDownloadButton>button:hover, .stButton>button:hover {{ background:{GOLD}; }}
.stDownloadButton>button:hover p, .stButton>button:hover p {{ color:{NAVY}!important; }}
[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {RULE}; }}
/* file-uploader: keep it on-theme and readable (cream dropzone, navy button) */
[data-testid="stFileUploaderDropzone"] {{ background:{CREAM}!important; border:1px dashed {RULE}!important; }}
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small {{ color:{INK}!important; }}
[data-testid="stFileUploaderDropzone"] button {{ background:{NAVY}!important; border:none!important; }}
[data-testid="stFileUploaderDropzone"] button p, [data-testid="stFileUploaderDropzone"] button div,
[data-testid="stFileUploaderDropzone"] button span {{ color:{CREAM}!important; }}
[data-testid="stElementToolbar"] {{ background:transparent; }}
[data-testid="stElementToolbarButton"], [data-testid="StyledFullScreenButton"], [data-testid="stElementToolbar"] button {{
  color:{NAVY}!important; background:{PANEL}!important; border:1px solid {RULE}!important; border-radius:3px; }}
[data-testid="stElementToolbar"] button:hover {{ color:{GOLD}!important; }}
.foot {{ color:{MUTED}; font-size:.78rem; margin-top:34px; border-top:1px solid {RULE}; padding-top:12px; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="wordmark">Organoscope</div>
<div class="tagline">per-organoid read-out of the whole field</div>
<div class="goldline"></div>
<div class="blurb">Every organoid in a microscopy field, measured on its own — live/dead and
morphology — instead of a single average per image.</div>
<hr class="hr">
""", unsafe_allow_html=True)

# ---------------- helpers ----------------
def grade_color(live):
    for thr, col, _ in GRADE:
        if live >= thr:
            return col
    return DEAD_I

def grade_cream(v):
    return LIVE if v >= 78 else PARTIAL if v >= 60 else DEAD

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
    for s in ax.spines.values(): s.set_color(NAVY); s.set_linewidth(2)
    fig.tight_layout()
    return fig

def guess_group(fname):
    """Pull a condition label out of a filename, e.g. 'YST_CTRL.png' -> 'CTRL'."""
    stem = re.sub(r"\.(png|tif|tiff|jpg|jpeg)$", "", str(fname), flags=re.I)
    toks = re.split(r"[_\-\s]+", stem)
    for t in toks:
        if t.upper() == "CTRL":
            return "CTRL"
        if re.fullmatch(r"g?\d+", t, flags=re.I):
            return t.lower()
    return toks[-1] if toks else stem

def group_key(g):
    g = str(g)
    if g.upper() == "CTRL":
        return (0, 0)
    m = re.search(r"(\d+)", g)
    return (1, int(m.group(1))) if m else (2, g)

# ---------------- sidebar: mode ----------------
with st.sidebar:
    st.markdown(f"<div class='sec' style='margin-top:0'>Mode</div>", unsafe_allow_html=True)
    mode = st.radio("Mode", ["Single field", "Compare groups"], label_visibility="collapsed")
    st.markdown(f"<div class='sec'>Settings</div>", unsafe_allow_html=True)
    sep = st.slider("Organoid separation", 10, 35, 18,
                    help="Lower separates touching organoids more; raise it if one splits in two.")
    st.markdown("<div class='note' style='color:#6B7685;margin-top:10px'>Live/dead from a merged RGB "
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
                    + "".join(f"<span><i class='dot' style='background:{c}'></i>{t}</span>" for c, t in LEG)
                    + "</div>", unsafe_allow_html=True)
        st.pyplot(overlay_figure(img[:, :, :3] if img.ndim == 3 else img, labels, df))
        st.markdown("<div class='cap'>Fig. 1 — each organoid labelled with its live %. "
                    "Numbers are per-organoid, not a field average.</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='sec'>Per-organoid viability</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.3, 4.3)); fig.patch.set_alpha(0); ax.set_facecolor("none")
        m, sd = lives.mean(), lives.std()
        ax.bar(0, m, width=0.52, color=NAVY, alpha=.16, edgecolor=NAVY, linewidth=1.3, zorder=1)
        ax.errorbar(0, m, sd, color=NAVY, capsize=7, elinewidth=1.4, capthick=1.4, zorder=2)
        xj = np.random.default_rng(0).uniform(-0.13, 0.13, len(lives))
        ax.scatter(xj, lives, c=[grade_cream(v) for v in lives], s=75, edgecolor="white", linewidth=.8, zorder=3)
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(0, 100)
        ax.set_xticks([0]); ax.set_xticklabels([str(name)[:16]], color=MUTED)
        ax.set_ylabel("live %", color=MUTED)
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

# ============================================================ COMPARE GROUPS
else:
    with st.sidebar:
        ups = st.file_uploader("Load images (several)", accept_multiple_files=True,
                               type=["tif", "tiff", "png", "jpg", "jpeg"])
        use_ex = st.checkbox("Use bundled example (24 images)",
                             value=not ups and len(glob.glob("data/groups/*")) > 0)

    # gather sources: (name, image_array)
    sources = []
    if ups:
        for f in ups:
            sources.append((f.name, imread(f)))
    elif use_ex:
        for p in sorted(glob.glob("data/groups/*")):
            sources.append((os.path.basename(p), imread(p)))

    if not sources:
        st.markdown("<div class='note'>Upload several images in the sidebar — one condition per image "
                    "(or several per condition). Name them so the condition is in the filename "
                    "(e.g. <b>CTRL_a.png</b>, <b>g1_a.png</b>, <b>g7_b.png</b>) and OrganoScope groups them "
                    "automatically. Tick “Use bundled example” to try it on 24 sample images.</div>",
                    unsafe_allow_html=True)
        st.stop()

    # group assignment table (editable)
    guess = pd.DataFrame({"image": [n for n, _ in sources],
                          "group": [guess_group(n) for n, _ in sources]})
    st.markdown("<div class='sec'>Assign groups</div>", unsafe_allow_html=True)
    st.markdown("<div class='cap'>Edit the group column if the auto-grouping is wrong.</div>", unsafe_allow_html=True)
    edited = st.data_editor(guess, hide_index=True, use_container_width=True,
                            disabled=["image"], key="grp")
    gmap = dict(zip(edited["image"], edited["group"]))

    # run OrganoScope on every image
    rows = []
    prog = st.progress(0.0, text="Analysing organoids…")
    for i, (n, im) in enumerate(sources):
        d, _ = analyze_array(im, source=n, split_min_distance=sep)
        d["group"] = str(gmap.get(n, guess_group(n)))
        rows.append(d)
        prog.progress((i + 1) / len(sources), text=f"Analysing… {i+1}/{len(sources)}")
    prog.empty()
    D = pd.concat(rows, ignore_index=True)
    if len(D) == 0:
        st.markdown("<div class='note'>No organoids detected across the images.</div>", unsafe_allow_html=True)
        st.stop()

    groups = sorted(D["group"].unique(), key=group_key)

    # headline stats
    st.markdown(f"""
    <div class="stats">
      <div class="stat"><div class="n">{len(groups)}</div><div class="k">groups</div></div>
      <div class="stat"><div class="n">{len(D)}</div><div class="k">organoids</div></div>
      <div class="stat"><div class="n">{len(sources)}</div><div class="k">images</div></div>
    </div>""", unsafe_allow_html=True)

    # Prism-style multi-group chart
    st.markdown("<div class='sec'>Viability by group</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='legend'>"
                + "".join(f"<span><i class='dot' style='background:{c}'></i>{t}</span>" for c, t in LEG)
                + "</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(min(10, 1.1 * len(groups) + 2), 4.6)); fig.patch.set_alpha(0)
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        v = D[D["group"] == g]["live_pct"].values
        ax.bar(i, v.mean(), 0.62, color=NAVY, alpha=.16, edgecolor=NAVY, lw=1.2, zorder=1)
        ax.errorbar(i, v.mean(), v.std(), color=NAVY, capsize=5, elinewidth=1.3, capthick=1.3, zorder=2)
        ax.scatter(i + rng.uniform(-.14, .14, len(v)), v, c=[grade_cream(x) for x in v],
                   s=45, edgecolor="white", lw=.6, zorder=3)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, color=MUTED)
    ax.set_ylim(0, 105); ax.set_ylabel("live %  (viability)", color=MUTED)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]: ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED)
    fig.tight_layout(); st.pyplot(fig)

    # statistics
    if len(groups) >= 2 and all((D["group"] == g).sum() >= 1 for g in groups):
        try:
            if len(groups) == 2:
                a, b = [D[D["group"] == g]["live_pct"].values for g in groups]
                stat, p = mannwhitneyu(a, b); test = "Mann–Whitney"
            else:
                stat, p = kruskal(*[D[D["group"] == g]["live_pct"].values for g in groups]); test = "Kruskal–Wallis"
            verdict = ("a <b>significant</b> difference between groups" if p < 0.05
                       else "<b>no significant</b> difference between groups")
            st.markdown(f"<div class='note'>{test} p = <span class='k-em'>{p:.3g}</span> — {verdict}. "
                        f"<span style='color:{MUTED}'>(p &lt; 0.05 = real difference; higher = statistically similar.)</span></div>",
                        unsafe_allow_html=True)
        except Exception:
            pass

    st.markdown("<div class='sec'>Group summary</div>", unsafe_allow_html=True)
    summ = (D.groupby("group")["live_pct"].agg(organoids="count", mean_live="mean", sd="std")
            .reindex(groups).round(1).reset_index())
    st.dataframe(summ, use_container_width=True, hide_index=True)
    st.download_button("Export per-organoid data (CSV)", D.to_csv(index=False).encode(),
                       file_name="organoscope_groups.csv", mime="text/csv")

st.markdown(f"<div class='foot'>OrganoScope · {AUTHOR}, {INSTITUTION}</div>", unsafe_allow_html=True)
