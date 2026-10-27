# OrganoScope

**Per-organoid analysis of tumor–immune co-culture microscopy.**
A computer-vision tool that measures *each organoid individually* — size, shape,
and live/dead — instead of collapsing a whole image into one average number, and
flags organoids whose response is unusual (e.g. candidate treatment-resistant clones).

---

## Why

In patient-derived lung-cancer organoid + PBMC co-culture assays (the GLI / InSMAR-chip
platform), immune killing is typically read out with Calcein-AM/PI live-dead staining and
summarized as **one Response index (Ri) per imaging field**. That field average hides the
biology that matters: within a single field, organoids can range from ~50% to ~86% live.
A resistant organoid sitting in a field that mostly died is exactly the cell that drives
relapse — and the field average erases it.

**OrganoScope measures every organoid separately, then discovers response patterns and
flags the outliers — with no manual labels.**

---

## Scope — what this project *is* (and is not)

This is deliberately scoped so it can actually be finished and defended.

**V1 — Per-organoid response profiling (the build).**
- Input: any organoid microscopy field (brightfield *or* two-channel fluorescence).
- Segment each organoid; measure per-organoid morphology (area, circularity, solidity,
  edge roughness) and, for fluorescence, live/dead fraction.
- Validation: per-group averages should reproduce the lab's published Ri ranking.
- Unsupervised: group organoids into response phenotypes; flag anomalies (resistant outliers).
- A web app to upload an image and see the per-organoid result.

**Future work (vision, NOT part of V1 — honest about data we don't yet have):**
- *V2 — Immune context:* per-organoid density of surrounding PBMCs (needs cleaner images).
- *V3 — Response trajectories:* multi-timepoint tracking (needs time-lapse data; our TIC
  data is single-endpoint, so this is developed on a public time-lapse first).
- *Parked — genomics / multimodal / "digital twin":* predicting molecular subtype from
  morphology, drug-response prediction, patient digital profiles. These need paired
  imaging + genomic data we do not have. Interesting; not buildable now.

---

## Data

- **Public (development & validation):** OrganoID 12-hour brightfield time-lapse (tracked,
  CC-licensed). Gives volume and a time-series.
- **Own research data (showcase):** one+ fluorescence fields from a lung-cancer organoid +
  PBMC co-culture (Calcein-AM/PI). *Limitation, stated honestly:* these are merged,
  brightness-adjusted composites, not raw channels — so live/dead separation is approximate.
  Single patient, single timepoint. This is a proof of concept, not a clinical claim.

---

## Pipeline

```
image → split channels → threshold (find cells) → label & split touching organoids
      → per-organoid features → live/dead + phenotype → anomaly flag → report / web
```

## Tech

Python · scikit-image · scipy · numpy · pandas · matplotlib. Web demo: self-contained HTML.

## Status

🟢 Engine prototype working on real images · 🟡 Rebuilding from scratch, step by step, for
full understanding · ⚪ Web app: preview built, live version pending engine completion.
