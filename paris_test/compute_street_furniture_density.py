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

PVP_JARDINIERES_BANCS_CORBEILLES_WEIGHT = 0.1
PVP_BORNES_BARRIERES_POTELETS_WEIGHT = 0.2
PVP_KIOSQUES_TOILETTES_PANNEAUX_WEIGHT = 0.3
TRILIB_WEIGHT = 0.10
FONTAINES_WEIGHT = 0.05
COMPOSTEURS_WEIGHT = 0.05
ANOMALIES_WEIGHT = 0.2


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


def load_tagged_soustypes(csv_path, variable_name):
    """Return Dans Ma Rue soustypes tagged with the requested variable."""
    soustypes = pd.read_csv(csv_path)
    tagged = soustypes[soustypes["variable"].astype(str).str.strip() == variable_name].copy()
    tagged["soustype"] = tagged["soustype"].astype(str).str.strip()
    tagged = tagged[tagged["soustype"].notna() & (tagged["soustype"] != "") & (tagged["soustype"] != "nan")]
    print(f"{variable_name} Dans Ma Rue soustypes ({len(tagged)}):")
    for row in tagged.itertuples(index=False):
        print(f"  - {row.type} / {row.soustype}")
    return tagged["soustype"].tolist()


def filter_trilib(features):
    """Keep Trilib stations that are currently in service."""
    if "emplacement_statut" not in features.columns:
        return features
    filtered = features[features["emplacement_statut"].astype(str).str.contains("service", case=False, na=False)].copy()
    print(f"Active Trilib stations: {len(filtered)} / {len(features)}")
    return filtered


def filter_fontaines(features):
    """Keep available drinking fountains located in Paris arrondissements."""
    filtered = features.copy()
    if "commune" in filtered.columns:
        filtered = filtered[filtered["commune"].astype(str).str.startswith("PARIS", na=False)].copy()
    if "dispo" in filtered.columns:
        filtered = filtered[filtered["dispo"].astype(str).str.upper().eq("OUI")].copy()
    print(f"Available Paris drinking fountains: {len(filtered)} / {len(features)}")
    return filtered


def normalize_clamped_or_sparse(series):
    """Clamp dense counts by quantile, but preserve sparse binary-ish layers with max scaling."""
    normalized, lower_value, upper_value = normalize_with_quantile_clamping(series)
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if upper_value <= lower_value and values.max() > 0:
        normalized = (values / values.max()).astype(float)
        lower_value = float(values.min())
        upper_value = float(values.max())
    return normalized, lower_value, upper_value


def main(args):
    """Score street-furniture density from nearby urban furniture and Dans Ma Rue clutter (0 = none, 1 = dense)."""
    processed_dir = os.path.join(root, "data", "processed")
    raw_dir = os.path.join(root, "data", "raw")
    os.makedirs(processed_dir, exist_ok=True)

    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    output_path = os.path.join(processed_dir, "street_furniture_density_paris.geojson")
    soustypes_path = os.path.join(root, "dans_ma_rue_soustypes.csv")
    raw_paths = {
        "street_furniture_jardinieres_bancs_corbeilles_count_raw": os.path.join(
            raw_dir, "street_furniture_jardinieres_bancs_corbeilles_raw.geojson"
        ),
        "street_furniture_bornes_barrieres_potelets_count_raw": os.path.join(
            raw_dir, "street_furniture_bornes_barrieres_potelets_raw.geojson"
        ),
        "street_furniture_kiosques_toilettes_panneaux_count_raw": os.path.join(
            raw_dir, "street_furniture_kiosques_toilettes_panneaux_raw.geojson"
        ),
        "street_furniture_composteurs_count_raw": os.path.join(raw_dir, "street_furniture_composteurs_raw.geojson"),
        "street_furniture_trilib_count_raw": os.path.join(raw_dir, "street_furniture_trilib_raw.geojson"),
        "street_furniture_fontaines_count_raw": os.path.join(raw_dir, "street_furniture_fontaines_raw.geojson"),
    }
    dans_ma_rue_path = os.path.join(raw_dir, "dans_ma_rue_raw.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run paris_test/segmentize_sidewalks.py first.")
    missing_paths = [path for path in list(raw_paths.values()) + [dans_ma_rue_path] if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(f"Missing source files: {missing_paths}. Run paris_test/download_data.py first.")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path).to_crs(PROJ)
    print(f"Loaded {len(points)} sample points")

    print(f"Buffering sample points by {args.near_buffer_distance_m:.3f} m (25 ft) ...")
    near_buffers = buffer_sample_points(points, args.near_buffer_distance_m)

    raw_component_columns = list(raw_paths.keys()) + ["street_furniture_anomaly_count_raw"]
    for column_name, path in raw_paths.items():
        print(f"Counting {column_name} in 25 ft buffers ...")
        features = flatten_z_geometries(load_features(path))
        if column_name == "street_furniture_trilib_count_raw":
            features = filter_trilib(features)
        elif column_name == "street_furniture_fontaines_count_raw":
            features = filter_fontaines(features)
        points[column_name] = aggregate_features_in_buffers(
            near_buffers,
            features,
            output_column=column_name,
        ).to_numpy()

    print("Loading Dans Ma Rue reports tagged as street_furniture ...")
    tagged_soustypes = load_tagged_soustypes(soustypes_path, "street_furniture")
    anomalies = gpd.read_file(dans_ma_rue_path, columns=["soustype", "geometry"]).to_crs(PROJ)
    anomalies = anomalies[anomalies.geometry.notna() & ~anomalies.geometry.is_empty].copy()
    anomalies = flatten_z_geometries(anomalies)
    anomalies["soustype"] = anomalies["soustype"].astype(str).str.strip()
    anomalies = anomalies[anomalies["soustype"].isin(tagged_soustypes)].copy()
    print(f"Dans Ma Rue street-furniture reports: {len(anomalies)}")
    if not anomalies.empty:
        print(anomalies["soustype"].value_counts().to_string())

    points["street_furniture_anomaly_count_raw"] = aggregate_features_in_buffers(
        near_buffers,
        anomalies,
        output_column="street_furniture_anomaly_count_raw",
    ).to_numpy()

    for raw_column in raw_component_columns:
        score_column = raw_column.replace("_raw", "_score")
        points[score_column], lower_value, upper_value = normalize_clamped_or_sparse(points[raw_column])
        print(f"{score_column} quantiles: {lower_value:.3f} -> {upper_value:.3f}")

    weights = {
        "street_furniture_jardinieres_bancs_corbeilles_count_score": PVP_JARDINIERES_BANCS_CORBEILLES_WEIGHT,
        "street_furniture_bornes_barrieres_potelets_count_score": PVP_BORNES_BARRIERES_POTELETS_WEIGHT,
        "street_furniture_kiosques_toilettes_panneaux_count_score": PVP_KIOSQUES_TOILETTES_PANNEAUX_WEIGHT,
        "street_furniture_trilib_count_score": TRILIB_WEIGHT,
        "street_furniture_fontaines_count_score": FONTAINES_WEIGHT,
        "street_furniture_composteurs_count_score": COMPOSTEURS_WEIGHT,
        "street_furniture_anomaly_count_score": ANOMALIES_WEIGHT,
    }

    points["street_furniture_density_score"] = 0.0
    for column_name, weight in weights.items():
        points["street_furniture_density_score"] += points[column_name].fillna(0.0) * weight
    points["street_furniture_density_score"] = points["street_furniture_density_score"].clip(0.0, 1.0)

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
        "street_furniture_jardinieres_bancs_corbeilles_count_raw",
        "street_furniture_jardinieres_bancs_corbeilles_count_score",
        "street_furniture_bornes_barrieres_potelets_count_raw",
        "street_furniture_bornes_barrieres_potelets_count_score",
        "street_furniture_kiosques_toilettes_panneaux_count_raw",
        "street_furniture_kiosques_toilettes_panneaux_count_score",
        "street_furniture_composteurs_count_raw",
        "street_furniture_composteurs_count_score",
        "street_furniture_trilib_count_raw",
        "street_furniture_trilib_count_score",
        "street_furniture_fontaines_count_raw",
        "street_furniture_fontaines_count_score",
        "street_furniture_anomaly_count_raw",
        "street_furniture_anomaly_count_score",
        "street_furniture_density_score",
        "geometry",
    ]
    keep_columns = [col for col in keep_columns if col in points.columns]
    points = points[keep_columns].copy()

    print(
        "street_furniture_density_score stats:\n"
        f"{points['street_furniture_density_score'].describe().to_string()}"
    )
    print(f"Writing {output_path} ...")
    points.to_file(output_path, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Paris street-furniture density from nearby urban furniture and Dans Ma Rue clutter."
    )
    parser.add_argument(
        "--near_buffer_distance_m",
        type=float,
        default=NEAR_BUFFER_DISTANCE_M,
        help="Buffer radius for nearby clutter sources, in metres (25 ft)",
    )
    args = parser.parse_args()
    main(args)
