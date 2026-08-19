import argparse
import os
import sys

import geopandas as gpd
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
from compute_street_furniture_density import flatten_z_geometries, load_tagged_soustypes

BASE_SCORE = 0.5
FEUX_WEIGHT = 0.5
ANOMALY_MALUS_WEIGHT = 0.25


def main(args):
    """Score traffic management: 0.5 base + 0.5 * clamped feu count - 0.25 * clamped anomaly count."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    feux_path = os.path.join(raw_dir, "feux_tricolores_raw.geojson")
    dans_ma_rue_path = os.path.join(raw_dir, "dans_ma_rue_raw.geojson")
    soustypes_path = os.path.join(root, "dans_ma_rue_soustypes.csv")
    output_path = os.path.join(processed_dir, "traffic_management_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run segmentize_sidewalks.py first.")
    for path in [feux_path, dans_ma_rue_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    print(f"Buffering sample points by {args.buffer_distance_m:.3f} m (200 ft) ...")
    far_buffers = buffer_sample_points(points, args.buffer_distance_m)

    print("Counting feux tricolores in 200 ft buffers ...")
    feux = flatten_z_geometries(load_features(feux_path))
    print(f"Loaded {len(feux)} feux tricolores")
    points["traffic_management_feux_count_raw"] = aggregate_features_in_buffers(
        far_buffers,
        feux,
        output_column="traffic_management_feux_count_raw",
    ).to_numpy()

    print("Loading Dans Ma Rue reports tagged as traffic_management ...")
    tagged_soustypes = load_tagged_soustypes(soustypes_path, "traffic_management")
    anomalies = gpd.read_file(dans_ma_rue_path, columns=["soustype", "geometry"]).to_crs(PROJ)
    anomalies = anomalies[anomalies.geometry.notna() & ~anomalies.geometry.is_empty].copy()
    anomalies = flatten_z_geometries(anomalies)
    anomalies["soustype"] = anomalies["soustype"].astype(str).str.strip()
    anomalies = anomalies[anomalies["soustype"].isin(tagged_soustypes)].copy()
    print(f"Dans Ma Rue traffic-management reports: {len(anomalies)}")
    if not anomalies.empty:
        print(anomalies["soustype"].value_counts().to_string())

    points["traffic_management_anomaly_count_raw"] = aggregate_features_in_buffers(
        far_buffers,
        anomalies,
        output_column="traffic_management_anomaly_count_raw",
    ).to_numpy()

    points["traffic_management_feux_count_score"], lo, hi = normalize_with_quantile_clamping(
        points["traffic_management_feux_count_raw"]
    )
    print(f"Feux count quantiles: {lo:.3f} -> {hi:.3f}")

    points["traffic_management_anomaly_count_score"], lo, hi = normalize_with_quantile_clamping(
        points["traffic_management_anomaly_count_raw"]
    )
    print(f"Anomaly count quantiles: {lo:.3f} -> {hi:.3f}")

    points["traffic_management_score"] = (
        BASE_SCORE
        + FEUX_WEIGHT * points["traffic_management_feux_count_score"].fillna(0.0)
        - ANOMALY_MALUS_WEIGHT * points["traffic_management_anomaly_count_score"].fillna(0.0)
    )
    points["traffic_management_score"] = points["traffic_management_score"].clip(0.0, 1.0)

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
        "traffic_management_feux_count_raw",
        "traffic_management_feux_count_score",
        "traffic_management_anomaly_count_raw",
        "traffic_management_anomaly_count_score",
        "traffic_management_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(
        "traffic_management_score stats:\n"
        f"{points['traffic_management_score'].describe().to_string()}"
    )
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris traffic-management score from feux tricolores and Dans Ma Rue anomalies."
    )
    parser.add_argument(
        "--buffer_distance_m",
        type=float,
        default=AMENITY_BUFFER_DISTANCE_M,
        help="Buffer radius for feature counting, in metres (200 ft)",
    )
    args = parser.parse_args()
    main(args)
