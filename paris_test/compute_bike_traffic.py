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
    AMENITY_BUFFER_DISTANCE_M,
    PROJ,
    aggregate_features_in_buffers,
    buffer_sample_points,
    load_features,
    normalize_with_quantile_clamping,
)
from compute_bike_lane import build_width_buffers, flatten_z_geometries

# Piste classes that count for bicycle traffic (includes couloir mixte).
BIKE_TRAFFIC_CLASSES = {"Piste cyclable", "Piste", "Couloir mixte"}
PISTE_SCORE = 0.8
VELIB_WEIGHT = 0.2


def flag_bike_traffic_intersection(point_buffers, pistes_bike):
    """Return 1.0 if buffer intersects any bike-traffic-compatible lane, else 0.0."""
    flags = np.zeros(len(point_buffers), dtype=float)
    if pistes_bike.empty:
        return flags
    tree = pistes_bike.sindex
    for i, buf in enumerate(point_buffers.geometry):
        candidates = list(tree.intersection(buf.bounds))
        if candidates and pistes_bike.iloc[candidates].intersects(buf).any():
            flags[i] = 1.0
    return flags


def main(args):
    """Score bicycle traffic: 0.8 if adjacent to piste/couloir mixte, +0.2 * clamped Vélib count."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    pistes_path = os.path.join(raw_dir, "pistes_cyclables_raw.geojson")
    velib_path = os.path.join(raw_dir, "velib_stations_raw.geojson")
    output_path = os.path.join(processed_dir, "bike_traffic_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run segmentize_sidewalks.py first.")
    for path in [pistes_path, velib_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    print("Loading pistes cyclables ...")
    pistes = flatten_z_geometries(load_features(pistes_path))
    if "lib_classe" in pistes.columns:
        pistes_bike = pistes[
            pistes["lib_classe"].astype(str).str.strip().isin(BIKE_TRAFFIC_CLASSES)
        ].copy()
    else:
        pistes_bike = pistes.copy()
    print(f"Bike-traffic pistes (piste + couloir mixte): {len(pistes_bike)}")

    print("Building width-based buffers ...")
    point_buffers = build_width_buffers(points)

    print("Flagging adjacency to bike lanes ...")
    piste_flags = flag_bike_traffic_intersection(point_buffers, pistes_bike)
    points["bike_traffic_piste_flag"] = piste_flags
    points["bike_traffic_score"] = PISTE_SCORE * piste_flags

    print(f"Buffering sample points by {args.velib_buffer_m:.1f} m (200 ft) for Vélib ...")
    far_buffers = buffer_sample_points(points, args.velib_buffer_m)

    print("Loading Vélib stations ...")
    velib = flatten_z_geometries(load_features(velib_path))
    print(f"Loaded {len(velib)} Vélib stations")

    print("Counting Vélib stations in 200 ft buffers ...")
    points["bike_traffic_velib_count_raw"] = aggregate_features_in_buffers(
        far_buffers, velib, output_column="bike_traffic_velib_count_raw"
    ).to_numpy()

    points["bike_traffic_velib_count_score"], lo, hi = normalize_with_quantile_clamping(
        points["bike_traffic_velib_count_raw"]
    )
    print(f"Vélib count quantiles: {lo:.3f} -> {hi:.3f}")

    points["bike_traffic_score"] = (
        points["bike_traffic_score"] + VELIB_WEIGHT * points["bike_traffic_velib_count_score"].fillna(0.0)
    ).clip(0.0, 1.0)

    keep_columns = [
        "point_index", "segment_index", "sidewalk_id", "pvp_tile",
        "arrondissement_id", "arrondissement",
        "qa_n_sq_qu", "qa_c_qu", "qa_c_quinsee", "qa_l_qu", "qa_c_ar", "qa_n_sq_ar",
        "width_m",
        "bike_traffic_piste_flag",
        "bike_traffic_velib_count_raw",
        "bike_traffic_velib_count_score",
        "bike_traffic_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(f"bike_traffic_score stats:\n{points['bike_traffic_score'].describe().to_string()}")
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris bicycle-traffic score from adjacent pistes and nearby Vélib stations."
    )
    parser.add_argument(
        "--velib_buffer_m",
        type=float,
        default=AMENITY_BUFFER_DISTANCE_M,
        help="Buffer radius for Vélib station counting (200 ft default)",
    )
    args = parser.parse_args()
    main(args)
