# OrganoScope

**Per-organoid analysis of organoid microscopy images.**

OrganoScope finds every organoid in a microscopy field and measures each one separately
(size, shape and live/dead fraction) instead of reducing the whole image to a single
average.

**Live demo:** [organoscope.streamlit.app](https://organoscope.streamlit.app/)

---

## Why

Tumor–immune co-culture assays are often read out with live/dead fluorescence staining
and summarized as one response value per imaging field. A field average can hide
organoids that respond very differently from their neighbours, such as a surviving
organoid in a field where most were killed. Measuring each organoid separately keeps
that variation visible.

---

## What it does

**Single field**
- Segments each organoid in a fluorescence (or brightfield) image
- Splits touching organoids with a distance-transform watershed (adjustable in the sidebar)
- Measures per-organoid area, equivalent diameter, circularity, solidity, eccentricity
  and live % (green vs. red signal)
- Shows a labelled per-organoid map, the field mean, the field response index
  (Ri = 1 − mean live fraction) and the live % range
- Exports the per-organoid table as CSV

**Compare groups**
- Analyses several images at once, one treatment condition per image
- Reads the group from the filename (`CTRL_a.png`, `g1_a.png`, ...), editable in the app
- Plots per-organoid viability by group and tests for differences
  (Mann–Whitney for 2 groups, Kruskal–Wallis for 3+)
- Exports a group summary and all per-organoid data as CSV

---

## How it works

```
image → remove scale bar / overlays → threshold signal (Otsu) → clean mask
      → split touching organoids (watershed) → per-organoid features → map, stats, CSV
```

The analysis engine is in `organoscope.py`. The web app is in `app.py`.

---

## Run locally

```bash
git clone https://github.com/enguun-nbb/OrganoScope.git
cd OrganoScope
pip install -r requirements.txt
streamlit run app.py
```

---

## Limitations

- Live % is calculated from the green and red channels of an RGB image. On merged or
  brightness-adjusted images it is an approximation, not a calibrated viability measure.
- The colour grades in the app (≥78% alive, 60–78% partial, <60% responding) are for
  display only and have not been validated as biological cut-offs.
- Segmentation is tuned for fluorescence organoid images and may need adjustment for
  other imaging setups.
- This is a research prototype, not a clinical tool.

---

## Planned

- Unsupervised grouping of organoids into response phenotypes
- Automatic flagging of outlier (possibly treatment-resistant) organoids
- Validation of per-organoid results against manually scored images
- Immune context: density of immune cells around each organoid
- Response trajectories from time-lapse imaging

---

## Tech

Python · scikit-image · SciPy · NumPy · pandas · Matplotlib · Streamlit

---

## Author

**Enguun Erdenesugar**, Biomedical Engineering, Tsinghua University

## License

MIT. See [LICENSE](LICENSE).
