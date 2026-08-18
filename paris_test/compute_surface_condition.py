import argparse
import os
import sys

import geopandas as gpd
import pandas as pd
import shapely

root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from compute_pedestrian_density import (
    NEAR_BUFFER_DISTANCE_M,
    PROJ,
    aggregate_features_in_buffers,
    buffer_sample_points,
    load_features,
    normalize_with_quantile_clamping,
)

CHANTIER_PENALTY = 0.25
ANOMALY_WEIGHT = 0.75
CHANTIER_OCCUPANCY_TOKENS = ("EMPRISE_TROTTOIR", "EMPRISE_PISTE_CYCLABLE")


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


def load_surface_condition_soustypes(csv_path):
    """Return Dans Ma Rue soustypes tagged as surface_condition."""
    soustypes = pd.read_csv(csv_path)
    tagged = soustypes[soustypes["variable"].astype(str).str.strip() == "surface_condition"].copy()
    tagged["soustype"] = tagged["soustype"].astype(str).str.strip()
    tagged = tagged[tagged["soustype"].notna() & (tagged["soustype"] != "") & (tagged["soustype"] != "nan")]
    print(f"Surface-condition Dans Ma Rue soustypes ({len(tagged)}):")
    for row in tagged.itertuples(index=False):
        print(f"  - {row.type} / {row.soustype}")
    return tagged["soustype"].tolist()


def localisation_affects_sidewalk_or_bike_lane(value):
    """True if chantier occupancy includes a sidewalk or bike lane."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, (list, tuple, set)):
        text = "/".join(str(item) for item in value)
    else:
        text = str(value)
    text = text.upper()
    return any(token in text for token in CHANTIER_OCCUPANCY_TOKENS)


def filter_chantiers_on_sidewalk_or_bike_lane(chantiers):
    """Keep chantier emprises that occupy a trottoir or piste cyclable."""
    if "localisation_detail" not in chantiers.columns:
        raise KeyError(
            "chantiers-a-paris is missing localisation_detail (Encombrement espace public)."
        )
    mask = chantiers["localisation_detail"].map(localisation_affects_sidewalk_or_bike_lane)
    filtered = chantiers.loc[mask].copy()
    print(f"Chantiers occupying trottoir or piste cyclable: {len(filtered)} / {len(chantiers)}")
    return filtered


def flag_points_intersecting_chantiers(buffered_points, chantiers):
    """Return a 0/1 series: 1 if the 25 ft buffer intersects a qualifying chantier."""
    if chantiers.empty:
        return pd.Series(0.0, index=buffered_points["point_index"], name="surface_condition_chantier_flag")

    joined = gpd.sjoin(
        buffered_points[["point_index", "geometry"]],
        chantiers[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    flagged_ids = joined["point_index"].unique() if not joined.empty else []
    flag = pd.Series(0.0, index=buffered_points["point_index"], name="surface_condition_chantier_flag")
    flag.loc[flag.index.isin(flagged_ids)] = 1.0
    return flag


def main(args):
    """Score sidewalk surface condition from chantiers and Dans Ma Rue anomalies."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    chantiers_path = os.path.join(raw_dir, "chantiers_a_paris_raw.geojson")
    dans_ma_rue_path = os.path.join(raw_dir, "dans_ma_rue_raw.geojson")
    soustypes_path = os.path.join(root, "dans_ma_rue_soustypes.csv")
    output_path = os.path.join(processed_dir, "surface_condition_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run paris_test/segmentize_sidewalks.py first.")
    if not os.path.exists(chantiers_path) or not os.path.exists(dans_ma_rue_path):
        raise FileNotFoundError("Missing chantiers or Dans Ma Rue data. Run paris_test/download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    print(f"Buffering sample points by {args.near_buffer_distance_m:.3f} m (25 ft) ...")
    near_buffers = buffer_sample_points(points, args.near_buffer_distance_m)

    print("Loading chantiers and keeping sidewalk / bike-lane occupancy ...")
    chantiers = flatten_z_geometries(load_features(chantiers_path))
    chantiers = filter_chantiers_on_sidewalk_or_bike_lane(chantiers)
    points["surface_condition_chantier_flag"] = flag_points_intersecting_chantiers(
        near_buffers, chantiers
    ).to_numpy()
    n_flagged = int((points["surface_condition_chantier_flag"] > 0).sum())
    print(f"Sample points intersecting a qualifying chantier: {n_flagged} / {len(points)}")

    print("Loading Dans Ma Rue reports tagged as surface_condition ...")
    tagged_soustypes = load_surface_condition_soustypes(soustypes_path)
    anomalies = gpd.read_file(dans_ma_rue_path, columns=["soustype", "geometry"]).to_crs(PROJ)
    anomalies = anomalies[anomalies.geometry.notna() & ~anomalies.geometry.is_empty].copy()
    anomalies = flatten_z_geometries(anomalies)
    anomalies["soustype"] = anomalies["soustype"].astype(str).str.strip()
    anomalies = anomalies[anomalies["soustype"].isin(tagged_soustypes)].copy()
    print(f"Dans Ma Rue surface-condition reports: {len(anomalies)}")
    if not anomalies.empty:
        print(anomalies["soustype"].value_counts().to_string())

    points["surface_condition_anomaly_count_raw"] = aggregate_features_in_buffers(
        near_buffers,
        anomalies,
        output_column="surface_condition_anomaly_count_raw",
    ).to_numpy()
    points["surface_condition_anomaly_count_score"], lower_value, upper_value = normalize_with_quantile_clamping(
        points["surface_condition_anomaly_count_raw"]
    )
    print(f"Anomaly count quantiles: {lower_value:.3f} -> {upper_value:.3f}")
    print(
        "Anomaly count stats:\n"
        f"{points['surface_condition_anomaly_count_raw'].describe().to_string()}"
    )

    points["surface_condition_score"] = (
        1.0
        - CHANTIER_PENALTY * points["surface_condition_chantier_flag"].fillna(0.0)
        - ANOMALY_WEIGHT * points["surface_condition_anomaly_count_score"].fillna(0.0)
    )
    points["surface_condition_score"] = points["surface_condition_score"].clip(0.0, 1.0)

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
        "surface_condition_chantier_flag",
        "surface_condition_anomaly_count_raw",
        "surface_condition_anomaly_count_score",
        "surface_condition_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(f"surface_condition_score stats:\n{points['surface_condition_score'].describe().to_string()}")
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris sidewalk surface condition from chantiers and Dans Ma Rue anomalies."
    )
    parser.add_argument(
        "--near_buffer_distance_m",
        type=float,
        default=NEAR_BUFFER_DISTANCE_M,
        help="Buffer radius for chantiers and anomalies, in metres (25 ft)",
    )
    args = parser.parse_args()
    main(args)
