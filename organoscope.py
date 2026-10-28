"""
organoscope.py — the reusable engine (Steps 1-6 folded into functions).

    analyze_image(path)  ->  DataFrame, one row per organoid.

This is the tidy version of everything we built script by script.
"""
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, remove_small_objects, dilation
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.measure import label, regionprops


def segment(img, blue_thresh=100, min_organoid_px=1400, split=True,
            split_min_distance=18, split_sigma=2.5):
    """Find organoids. Returns (labels, green, red). Removes the white scale bar
    using the blue channel (organoids emit no blue), then splits touching
    organoids with a watershed on the distance transform.

    split_min_distance: smaller -> separates touching organoids more aggressively.
    """
    red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    graphics = dilation(blue > blue_thresh, disk(4))          # scale bar / overlays
    signal = np.maximum(green, red).astype(float)
    signal[graphics] = 0
    if (~graphics).sum() == 0:
        return np.zeros_like(red, dtype=int), green, red
    mask = signal > threshold_otsu(signal[~graphics])
    clean = remove_small_objects(
        ndi.binary_fill_holes(closing(mask, disk(10))), min_organoid_px)

    if split:
        # Distance transform: each organoid is a "hill"; peaks = centers.
        dist = gaussian(ndi.distance_transform_edt(clean), split_sigma)
        coords = peak_local_max(dist, min_distance=split_min_distance, labels=clean,
                                threshold_rel=0.3, exclude_border=False)
        markers = np.zeros(dist.shape, dtype=int)
        for i, (r, c) in enumerate(coords, 1):
            markers[r, c] = i
        markers, _ = ndi.label(markers > 0)
        labels = watershed(-dist, markers, mask=clean)
        labels = remove_small_objects(labels, min_organoid_px)
    else:
        labels = label(clean)
    return labels, green, red


def analyze_array(img, source="upload", split_min_distance=18):
    """Per-organoid feature table for an in-memory image.
    Returns (DataFrame, labels) so a UI can also draw the overlay.
    split_min_distance: smaller separates touching organoids more aggressively."""
    if img.ndim == 2:                                  # grayscale -> fake 3 channels
        img = np.stack([img, img, np.zeros_like(img)], axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    labels, green, red = segment(img, split_min_distance=split_min_distance)
    recs = []
    for r in regionprops(labels):
        m = labels == r.label
        g, d = float(green[m].sum()), float(red[m].sum())
        perim = r.perimeter if r.perimeter > 0 else 1
        recs.append({
            "source": source,
            "organoid_id": int(r.label),
            "centroid_x": round(r.centroid[1], 1),
            "centroid_y": round(r.centroid[0], 1),
            "area": int(r.area),
            "diameter": round(r.equivalent_diameter_area, 1),
            "circularity": round(min(4 * np.pi * r.area / perim ** 2, 1.0), 3),
            "solidity": round(r.solidity, 3),
            "eccentricity": round(r.eccentricity, 3),
            "live_pct": round(100 * g / (g + d + 1e-9), 1),
        })
    return pd.DataFrame(recs), labels


def analyze_image(path, source=None):
    """Full per-organoid feature table for one image file."""
    df, _ = analyze_array(imread(path), source=source or path.split("/")[-1])
    return df.drop(columns=["centroid_x", "centroid_y"])
