import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from compute_pedestrian_density import (
    NEAR_BUFFER_DISTANCE_M,
    PROJ,
    load_features,
    normalize_with_quantile_clamping,
)

# Neighbor search radius for slope computation (50 ft, matching NYC paper).
FEET_TO_METERS = 0.3048
SLOPE_NEIGHBOR_RADIUS_M = 50.0 * FEET_TO_METERS


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


def assign_elevation_to_points(points, nivellement, buffer_m):
    """Average altitude_en_metre of nivellement points within buffer_m of each sample point."""
    # Build a KD-tree on the nivellement point cloud for fast radius queries.
    niv_coords = np.array([[geom.x, geom.y] for geom in nivellement.geometry])
    niv_alts = nivellement["altitude_en_metre"].to_numpy(dtype=float)
    pt_coords = np.array([[geom.x, geom.y] for geom in points.geometry])

    print(f"  KD-tree built on {len(niv_coords)} points. Querying {len(pt_coords)} sample points ...")
    tree = cKDTree(niv_coords)
    elevations = np.full(len(pt_coords), np.nan)

    indices_list = tree.query_ball_point(pt_coords, r=buffer_m, workers=-1)
    print(f"  Radius queries done. Averaging altitudes ...")
    for i, idxs in enumerate(indices_list):
        if idxs:
            elevations[i] = float(np.mean(niv_alts[idxs]))

    # For points with no coverage, interpolate from the nearest single point.
    missing = np.where(np.isnan(elevations))[0]
    if len(missing) > 0:
        dists, nearest = tree.query(pt_coords[missing], k=1)
        elevations[missing] = niv_alts[nearest]

    return elevations


def compute_mean_slope(pt_coords, elevations, neighbor_radius_m, k=10):
    """For each point find up to k neighbors within radius and return the mean |rise/run| slope."""
    print(f"  Building sample-point KD-tree ({len(pt_coords)} points) ...")
    tree = cKDTree(pt_coords)
    mean_slopes = np.zeros(len(pt_coords))

    print(f"  Querying up to k={k} neighbors within {neighbor_radius_m:.1f} m ...")
    distances, indices = tree.query(pt_coords, k=k + 1, distance_upper_bound=neighbor_radius_m)
    print("  Computing mean slopes ...")
    # query returns k+1 results; index 0 is always the point itself (distance 0).
    for i in range(len(pt_coords)):
        dists = distances[i, 1:]   # exclude self
        idxs = indices[i, 1:]
        valid = (idxs < len(pt_coords)) & (dists > 0)
        if valid.any():
            h_diffs = np.abs(elevations[idxs[valid]] - elevations[i])
            slopes = h_diffs / dists[valid]
            mean_slopes[i] = float(slopes.mean())

    return mean_slopes


def main(args):
    """Assign elevation from nearby nivellement points, then compute mean slope to neighboring sample points."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    sidewalks_path = os.path.join(processed_dir, "sidewalks_paris.geojson")
    nivellement_path = os.path.join(raw_dir, "points_de_nivellement_raw.geojson")
    output_path = os.path.join(processed_dir, "slope_gradient_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run segmentize_sidewalks.py first.")
    if not os.path.exists(sidewalks_path):
        raise FileNotFoundError(f"Missing {sidewalks_path}. Run process_sidewalks.py first.")
    if not os.path.exists(nivellement_path):
        raise FileNotFoundError(f"Missing {nivellement_path}. Run download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    print("Loading sidewalk polygons for spatial filtering ...")
    sidewalks = gpd.read_file(sidewalks_path, columns=["geometry"]).to_crs(PROJ)
    sidewalks = flatten_z_geometries(sidewalks)
    print(f"Loaded {len(sidewalks)} sidewalk polygons")

    print("Loading points de nivellement ...")
    nivellement = gpd.read_file(nivellement_path, columns=["altitude_en_metre", "geometry"]).to_crs(PROJ)
    nivellement = flatten_z_geometries(nivellement)
    nivellement = nivellement[
        nivellement["altitude_en_metre"].notna()
        & nivellement.geometry.notna()
        & ~nivellement.geometry.is_empty
    ].copy()
    nivellement["altitude_en_metre"] = pd.to_numeric(nivellement["altitude_en_metre"], errors="coerce")
    nivellement = nivellement[nivellement["altitude_en_metre"].notna()].copy()
    print(f"Loaded {len(nivellement)} nivellement labels with valid altitude")

    print("Filtering nivellement to points inside sidewalk polygons (sjoin) ...")
    sidewalk_hits = gpd.sjoin(
        nivellement[["altitude_en_metre", "geometry"]],
        sidewalks[["geometry"]],
        how="inner",
        predicate="within",
    )
    nivellement = nivellement.loc[sidewalk_hits.index.unique()].copy()
    print(f"Retained {len(nivellement)} nivellement labels within sidewalks")

    print(f"Assigning elevation from nivellement within {args.elevation_buffer_m:.1f} m (25 ft) ...")
    print("  (building KD-tree on filtered nivellement ...)")
    pt_coords = np.array([[geom.x, geom.y] for geom in points.geometry])
    elevations = assign_elevation_to_points(points, nivellement, args.elevation_buffer_m)
    points["slope_gradient_elevation_m"] = elevations
    n_missing = int(np.isnan(elevations).sum())
    print(f"Elevation assigned to all points (fallback nearest used for {n_missing})")
    print(f"Elevation range: {np.nanmin(elevations):.2f} m – {np.nanmax(elevations):.2f} m")

    print(f"Computing mean slope to neighbors within {args.slope_radius_m:.1f} m (50 ft) ...")
    mean_slopes = compute_mean_slope(pt_coords, elevations, args.slope_radius_m, k=args.k_neighbors)
    points["slope_gradient_mean_raw"] = mean_slopes
    print(f"Mean slope stats: min={mean_slopes.min():.4f} mean={mean_slopes.mean():.4f} max={mean_slopes.max():.4f}")

    points["slope_gradient_score"], lo, hi = normalize_with_quantile_clamping(points["slope_gradient_mean_raw"])
    print(f"Slope score (clamped 2.5-99.5%): {lo:.4f} -> {hi:.4f}")
    # Invert: 1 = flat (good for robots), 0 = steep.
    points["slope_gradient_score"] = 1.0 - points["slope_gradient_score"]

    keep_columns = [
        "point_index",
        "segment_index",
        "sidewalk_id",
        "pvp_tile",
        "arrondissement_id",
        "arrondissement",
        "qa_n_sq_qu",
        "qa_c_qu",
        "qa_c_quinsee",
        "qa_l_qu",
        "qa_c_ar",
        "qa_n_sq_ar",
        "slope_gradient_elevation_m",
        "slope_gradient_mean_raw",
        "slope_gradient_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(
        "slope_gradient_score stats:\n"
        f"{points['slope_gradient_score'].describe().to_string()}"
    )
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris slope-gradient score from points de nivellement and segmentized sidewalk points."
    )
    parser.add_argument(
        "--elevation_buffer_m",
        type=float,
        default=NEAR_BUFFER_DISTANCE_M,
        help="Radius for averaging nearby nivellement altitudes (25 ft default)",
    )
    parser.add_argument(
        "--slope_radius_m",
        type=float,
        default=SLOPE_NEIGHBOR_RADIUS_M,
        help="Radius for neighbor search when computing slope (50 ft default)",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=10,
        help="Max number of neighbors to consider per point",
    )
    args = parser.parse_args()
    main(args)
