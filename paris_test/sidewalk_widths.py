import argparse
import os
import time

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, nearest_points
from centerline.geometry import Centerline

from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
PROJ = "EPSG:2154"          # Lambert-93 (metres)
MIN_AREA_M2 = 10.0           # skip slivers that crash the centerline library
SIMPLIFY_TOL = 0.5          # metres, applied to centerlines before segmenting

# label columns to carry through from the input to the output segments
LABEL_COLS = ["sidewalk_id", "num_pave", "pvp_tile", "arrondissement_id", "arrondissement"]


# ── helpers ───────────────────────────────────────────────────────────────────

def get_centerline(geom):
    """Extract skeleton centerline(s) from a sidewalk polygon. Returns None on failure."""
    try:
        return MultiLineString(list(Centerline(geom).geometry.geoms))
    except Exception:
        return None


def clean_centerline(ml):
    """Merge collinear segments and drop short dead-end stubs."""
    if ml is None:
        return None
    ml = linemerge(ml)
    if ml is None:
        return None
    # wrap a plain LineString so the rest of the code handles one type
    if ml.geom_type == "LineString":
        ml = MultiLineString([ml])
    passing = []
    for i, ls in enumerate(ml.geoms):
        others = MultiLineString([x for j, x in enumerate(ml.geoms) if j != i])
        p0, p1 = Point(ls.coords[0]), Point(ls.coords[-1])
        is_deadend = p0.disjoint(others) or p1.disjoint(others)
        if not is_deadend or ls.length > 5:
            passing.append(ls)
    return MultiLineString(passing) if passing else None


def to_segments(ml):
    """Split a MultiLineString into individual 2-point LineString segments."""
    if ml is None:
        return []
    segs = []
    for ls in ml.geoms:
        try:
            ls = ls.simplify(SIMPLIFY_TOL, preserve_topology=True)
        except Exception:
            pass
        coords = list(ls.coords)
        segs += [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]
    return segs


def polygon_to_multilinestring(poly):
    """Convert a polygon's boundary (exterior + holes) to a MultiLineString."""
    return MultiLineString([poly.exterior] + list(poly.interiors))


def avg_half_width(segment, boundary_ml):
    """
    Average distance from points along the centerline segment to the polygon
    boundary, which equals half the local sidewalk width.
    """
    # sample one point per metre along the segment
    n = max(2, round(segment.length))
    points = [segment.interpolate(segment.length / (n - 1) * i) for i in range(n)]
    dists = [nearest_points(boundary_ml, pt)[0].distance(pt) for pt in points]
    return float(np.mean(dists))


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    """Compute centerline-based sidewalk widths for Paris and write segment GeoJSON."""
    start_time = time.time()
    root = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(root, "data", "processed", "sidewalks_paris.geojson")
    out_dir = os.path.join(root, "data", "processed")
    out_path = os.path.join(out_dir, "sidewalk_widths_paris.geojson")

    print(f"Loading processed sidewalks from {in_path} ...", flush=True)
    gdf = gpd.read_file(in_path).to_crs(PROJ)
    print(f"Loaded {len(gdf)} sidewalk polygons", flush=True)

    # drop tiny slivers
    n_before = len(gdf)
    gdf = gdf[gdf.geometry.area >= MIN_AREA_M2].copy().reset_index(drop=True)
    print(
        f"Dropped {n_before - len(gdf)} slivers < {MIN_AREA_M2} m² -> {len(gdf)} polygons remain",
        flush=True,
    )

    # ── step 1: centerlines ───────────────────────────────────────────────────
    print("Step 1/4: computing centerlines ...", flush=True)
    gdf["centerline"] = [
        get_centerline(geom)
        for geom in tqdm(gdf["geometry"], total=len(gdf), desc="Centerlines", mininterval=1.0)
    ]

    n_failed = gdf["centerline"].isna().sum()
    print(f"Centerline failures: {n_failed} / {len(gdf)}", flush=True)
    gdf = gdf[gdf["centerline"].notna()].copy()
    print(f"Centerlines kept: {len(gdf)}", flush=True)

    # ── step 2: clean centerlines ─────────────────────────────────────────────
    print("Step 2/4: cleaning centerlines ...", flush=True)
    gdf["centerline"] = [
        clean_centerline(ml)
        for ml in tqdm(gdf["centerline"], total=len(gdf), desc="Clean centerlines", mininterval=1.0)
    ]
    gdf = gdf[gdf["centerline"].notna()].copy()
    print(f"Clean centerlines kept: {len(gdf)}", flush=True)

    # ── step 3: split into segments ───────────────────────────────────────────
    print("Step 3/4: splitting centerlines into segments ...", flush=True)
    gdf["segments"] = [
        to_segments(ml)
        for ml in tqdm(gdf["centerline"], total=len(gdf), desc="Segments", mininterval=1.0)
    ]
    gdf["n_segments"] = [len(segs) for segs in gdf["segments"]]
    print(f"Total raw segments: {int(gdf['n_segments'].sum())}", flush=True)

    # ── step 4: compute widths ────────────────────────────────────────────────
    print("Step 4/4: computing widths (slow, one sample point per metre) ...", flush=True)

    rows = []
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Widths", mininterval=1.0):
        boundary_ml = polygon_to_multilinestring(row.geometry)
        labels = {col: row[col] for col in LABEL_COLS if col in row.index}
        for seg in row["segments"]:
            hw = avg_half_width(seg, boundary_ml)
            entry = {"geometry": seg, "width_m": hw * 2}
            entry.update(labels)
            rows.append(entry)

    df_segs = GeoDataFrame(rows, geometry="geometry", crs=PROJ)
    df_segs["segment_id"] = df_segs.index

    print(f"Total final segments: {len(df_segs)}", flush=True)
    print(f"Width stats (m):\n{df_segs['width_m'].describe().to_string()}", flush=True)
    print(f"Writing {out_path} ...", flush=True)
    df_segs.to_file(out_path, driver="GeoJSON")
    elapsed_s = time.time() - start_time
    print(f"Done in {elapsed_s / 60:.1f} minutes.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris sidewalk widths from polygon centerlines."
    )
    args = parser.parse_args()
    main(args)
