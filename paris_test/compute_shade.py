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

from compute_pedestrian_density import (
    PROJ,
    buffer_sample_points,
    load_features,
    normalize_with_quantile_clamping,
)

FEET_TO_METERS = 0.3048
SHADE_BUFFER_DISTANCE_M = 50.0 * FEET_TO_METERS   # 50 ft
MIN_HEIGHT_M = 3.0
YOUNG_TREE_WEIGHT = 0.5


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


def weighted_tree_count_in_buffers(buffers, trees):
    """Count trees in each buffer, weighting young trees at 0.5 and mature at 1.0."""
    # Assign weight per tree.
    if "stadedeveloppement" in trees.columns:
        is_young = trees["stadedeveloppement"].astype(str).str.strip().str.lower() == "jeune (arbre)"
        trees = trees.copy()
        trees["_weight"] = np.where(is_young, YOUNG_TREE_WEIGHT, 1.0)
    else:
        trees = trees.copy()
        trees["_weight"] = 1.0

    joined = gpd.sjoin(buffers[["point_index", "geometry"]], trees[["_weight", "geometry"]], how="left", predicate="intersects")
    counts = (
        joined.groupby("point_index")["_weight"]
        .sum()
        .reindex(buffers["point_index"])
        .fillna(0.0)
    )
    return counts.values


def main(args):
    """Count weighted trees (>3 m) within 50 ft of each point, normalize 2.5-99.5% to 0-1."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    arbres_path = os.path.join(raw_dir, "arbres_raw.geojson")
    output_path = os.path.join(processed_dir, "shade_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run segmentize_sidewalks.py first.")
    if not os.path.exists(arbres_path):
        raise FileNotFoundError(f"Missing {arbres_path}. Run download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    print(f"Buffering sample points by {args.buffer_distance_m:.1f} m (50 ft) ...")
    shade_buffers = buffer_sample_points(points, args.buffer_distance_m)

    print("Loading arbres ...")
    arbres = gpd.read_file(arbres_path, columns=["hauteurenm", "stadedeveloppement", "geometry"]).to_crs(PROJ)
    arbres = flatten_z_geometries(arbres)
    arbres = arbres[arbres.geometry.notna() & ~arbres.geometry.is_empty].copy()
    arbres["hauteurenm"] = pd.to_numeric(arbres["hauteurenm"], errors="coerce")
    arbres = arbres[arbres["hauteurenm"] >= args.min_height_m].copy()
    print(f"Loaded {len(arbres)} arbres taller than {args.min_height_m} m")
    if "stadedeveloppement" in arbres.columns:
        print("Stade de développement counts:\n" + arbres["stadedeveloppement"].value_counts(dropna=False).head(10).to_string())

    print("Computing weighted tree counts in buffers ...")
    points["shade_tree_count_raw"] = weighted_tree_count_in_buffers(shade_buffers, arbres)

    points["shade_score"], lo, hi = normalize_with_quantile_clamping(points["shade_tree_count_raw"])
    print(f"Weighted tree count quantiles: {lo:.3f} -> {hi:.3f}")

    keep_columns = [
        "point_index", "segment_index", "sidewalk_id", "pvp_tile",
        "arrondissement_id", "arrondissement",
        "qa_n_sq_qu", "qa_c_qu", "qa_c_quinsee", "qa_l_qu", "qa_c_ar", "qa_n_sq_ar",
        "shade_tree_count_raw",
        "shade_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(f"shade_score stats:\n{points['shade_score'].describe().to_string()}")
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris shade score from nearby trees taller than 3 m."
    )
    parser.add_argument(
        "--buffer_distance_m",
        type=float,
        default=SHADE_BUFFER_DISTANCE_M,
        help="Buffer radius for tree counting (50 ft default)",
    )
    parser.add_argument(
        "--min_height_m",
        type=float,
        default=MIN_HEIGHT_M,
        help="Minimum tree height in metres to count",
    )
    args = parser.parse_args()
    main(args)
