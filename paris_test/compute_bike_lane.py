import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from compute_pedestrian_density import PROJ, load_features

# lib_classe values that count as proper bike infrastructure for bike-lane availability.
PISTE_CLASSES = {"Piste cyclable"}           # score 1.0
PISTE_CYCLABLE_CLASSES = {"Piste"}           # score 0.8  (lib_niveau "PISTE - PISTE CYCLABLE")

# Width buffer: sidewalk_width × 1.5, capped at 8 m.
WIDTH_MULTIPLIER = 1.5
MAX_BUFFER_M = 8.0
DEFAULT_BUFFER_M = 3.0   # fallback if width_m missing


def flatten_z_geometries(gdf):
    """Drop Z coordinates so spatial joins stay 2D."""
    if gdf.geometry.has_z.any():
        gdf = gdf.copy()
        gdf["geometry"] = gpd.GeoSeries(
            shapely.force_2d(gdf.geometry.values),
            index=gdf.index,
            crs=gdf.crs,
        )
    return gdf


def build_width_buffers(points):
    """Buffer each sample point by min(width_m * 1.5, 8 m)."""
    if "width_m" in points.columns:
        widths = pd.to_numeric(points["width_m"], errors="coerce").fillna(DEFAULT_BUFFER_M)
    else:
        widths = pd.Series(DEFAULT_BUFFER_M, index=points.index)
    radii = (widths * WIDTH_MULTIPLIER).clip(upper=MAX_BUFFER_M)
    buffers = points.geometry.buffer(radii)
    return gpd.GeoDataFrame({"point_index": points["point_index"].values, "geometry": buffers}, crs=points.crs)


def flag_bike_lane_intersection(point_buffers, pistes_piste, pistes_piste_cyclable):
    """Return per-point score: 0 (none), 0.8 (piste), 1.0 (piste cyclable)."""
    n = len(point_buffers)
    scores = np.zeros(n, dtype=float)

    if not pistes_piste.empty:
        pistes_tree = pistes_piste.sindex
        for i, buf in enumerate(point_buffers.geometry):
            candidates = list(pistes_tree.intersection(buf.bounds))
            if candidates and pistes_piste.iloc[candidates].intersects(buf).any():
                scores[i] = 0.8

    if not pistes_piste_cyclable.empty:
        pc_tree = pistes_piste_cyclable.sindex
        for i, buf in enumerate(point_buffers.geometry):
            candidates = list(pc_tree.intersection(buf.bounds))
            if candidates and pistes_piste_cyclable.iloc[candidates].intersects(buf).any():
                scores[i] = 1.0

    return scores


def main(args):
    """Score bike-lane availability: 0 base, 0.8 if adjacent piste, 1.0 if adjacent piste cyclable."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    pistes_path = os.path.join(raw_dir, "pistes_cyclables_raw.geojson")
    output_path = os.path.join(processed_dir, "bike_lane_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run segmentize_sidewalks.py first.")
    if not os.path.exists(pistes_path):
        raise FileNotFoundError(f"Missing {pistes_path}. Run download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    print("Loading pistes cyclables ...")
    pistes = flatten_z_geometries(load_features(pistes_path))
    print(f"Loaded {len(pistes)} piste features")
    if "lib_classe" in pistes.columns:
        print("lib_classe counts:\n" + pistes["lib_classe"].value_counts(dropna=False).to_string())

    # Separate the two tiers (only piste/piste cyclable, not couloir mixte or bus).
    if "lib_classe" in pistes.columns:
        pistes_piste_cyclable = pistes[pistes["lib_classe"].astype(str).str.strip() == "Piste cyclable"].copy()
        pistes_piste = pistes[pistes["lib_classe"].astype(str).str.strip() == "Piste"].copy()
    else:
        pistes_piste_cyclable = pistes.iloc[0:0].copy()
        pistes_piste = pistes.iloc[0:0].copy()

    print(f"Piste cyclable (score 1.0): {len(pistes_piste_cyclable)} features")
    print(f"Piste (score 0.8): {len(pistes_piste)} features")

    print("Building width-based buffers for each sample point ...")
    point_buffers = build_width_buffers(points)

    print("Flagging intersections with bike lanes ...")
    points["bike_lane_score"] = flag_bike_lane_intersection(point_buffers, pistes_piste, pistes_piste_cyclable)

    n_piste = (points["bike_lane_score"] == 0.8).sum()
    n_pc = (points["bike_lane_score"] == 1.0).sum()
    print(f"Points adjacent to piste: {n_piste}, piste cyclable: {n_pc}, none: {len(points) - n_piste - n_pc}")

    keep_columns = [
        "point_index", "segment_index", "sidewalk_id", "pvp_tile",
        "arrondissement_id", "arrondissement",
        "qa_n_sq_qu", "qa_c_qu", "qa_c_quinsee", "qa_l_qu", "qa_c_ar", "qa_n_sq_ar",
        "width_m",
        "bike_lane_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(f"bike_lane_score stats:\n{points['bike_lane_score'].describe().to_string()}")
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris bike-lane availability score from pistes cyclables adjacency."
    )
    args = parser.parse_args()
    main(args)
