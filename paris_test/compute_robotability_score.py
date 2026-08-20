import argparse
import os

import geopandas as gpd
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(root)

# NYC polarities from feature_processing/score.ipynb.
# Positive: higher feature value helps robotability.
# Negative: higher feature value hurts robotability.
POLARITIES = {
    "sidewalk_width": 1,
    "pedestrian_density": -1,
    "street_furniture_density": -1,
    "sidewalk_roughness": -1,
    "surface_condition": 1,
    "communication_infrastructure": 1,
    "slope_gradient": -1,
    "charging_station_proximity": 1,
    "curb_ramp_availability": 1,
    "crowd_dynamics": 1,
    "traffic_management": 1,
    "surveillance_coverage": 1,
    "zoning_laws": 1,
    "bike_lane_availability": 1,
    "gps_signal_strength": 1,
    "bicycle_traffic": -1,
    "vehicle_traffic": -1,
    "digital_map_existence": 1,
    "intersection_safety": -1,
}

# Paris already stores these as high=good, while NYC polarity assumes high=bad raw.
# Convert to NYC semantic direction before applying NYC polarities.
PARIS_ALREADY_GOOD_ORIENTED = {
    "intersection_safety",  # Paris score is safety (1 = safe)
}

METADATA_COLUMNS = [
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
    "width_m",
    "lon",
    "lat",
]

# Canonical feature -> Paris GeoJSON file + score column + optional raw columns.
FEATURE_SOURCES = {
    "pedestrian_density": {
        "file": "pedestrian_density_paris.geojson",
        "score": "pedestrian_density_score",
        "raw": [
            "pedestrian_density_base_density_raw",
            "pedestrian_density_tourist_sites_count_raw",
            "pedestrian_density_lieux_municipaux_count_raw",
            "pedestrian_density_colleges_count_raw",
            "pedestrian_density_ecoles_elementaires_count_raw",
            "pedestrian_density_ecoles_maternelles_count_raw",
            "pedestrian_density_kiosques_de_presse_count_raw",
            "pedestrian_density_points_arrets_count_raw",
            "pedestrian_density_terrasses_surface_raw",
            "pedestrian_density_activities_2025_count_raw",
        ],
    },
    "crowd_dynamics": {
        "file": "crowd_dynamics_paris.geojson",
        "score": "crowd_dynamics_score",
        "raw": [
            "crowd_dynamics_zti_score",
            "crowd_dynamics_tourist_sites_count_raw",
            "crowd_dynamics_tourism_score",
        ],
    },
    "surface_condition": {
        "file": "surface_condition_paris.geojson",
        "score": "surface_condition_score",
        "raw": [
            "surface_condition_chantier_flag",
            "surface_condition_anomaly_count_raw",
        ],
    },
    "sidewalk_width": {
        "file": "sidewalk_width_paris.geojson",
        "score": "sidewalk_width_score",
        "raw": ["sidewalk_width_raw"],
    },
    "street_furniture_density": {
        "file": "street_furniture_density_paris.geojson",
        "score": "street_furniture_density_score",
        "raw": [
            "street_furniture_jardinieres_bancs_corbeilles_count_raw",
            "street_furniture_bornes_barrieres_potelets_count_raw",
            "street_furniture_kiosques_toilettes_panneaux_count_raw",
            "street_furniture_composteurs_count_raw",
            "street_furniture_trilib_count_raw",
            "street_furniture_fontaines_count_raw",
            "street_furniture_anomaly_count_raw",
        ],
    },
    "intersection_safety": {
        "file": "intersection_safety_paris.geojson",
        "score": "intersection_safety_score",
        "raw": [
            "intersection_safety_aire_pietonne_flag",
            "intersection_safety_zone_rencontre_flag",
            "intersection_safety_zone_score",
            "intersection_safety_accident_count_raw",
            "intersection_safety_accident_rate_raw",
            "intersection_safety_anomaly_count_raw",
        ],
    },
    "curb_ramp_availability": {
        "file": "curb_ramp_availability_paris.geojson",
        "score": "curb_ramp_availability_score",
        "raw": [
            "curb_ramp_availability_escalier_flag",
            "curb_ramp_availability_accessibilite_flag",
        ],
    },
    "communication_infrastructure": {
        "file": "communication_infrastructure_paris.geojson",
        "score": "communication_infrastructure_score",
        "raw": [],
    },
    "digital_map_existence": {
        "file": "digital_map_existence_paris.geojson",
        "score": "digital_map_existence_score",
        "raw": [],
    },
    "gps_signal_strength": {
        "file": "gps_signal_strength_paris.geojson",
        "score": "gps_signal_strength_score",
        "raw": [],
    },
    "vehicle_traffic": {
        "file": "vehicle_traffic_paris.geojson",
        "score": "vehicle_traffic_score",
        "raw": [
            "vehicle_traffic_occupation_raw",
            "vehicle_traffic_distance_m",
            "iu_ac",
        ],
    },
    "sidewalk_roughness": {
        "file": "sidewalk_roughness_paris.geojson",
        "score": "sidewalk_roughness_score",
        "raw": [],
    },
    "slope_gradient": {
        "file": "slope_gradient_paris.geojson",
        "score": "slope_gradient_score",
        "raw": [
            "slope_gradient_elevation_m",
            "slope_gradient_mean_raw",
        ],
    },
    "traffic_management": {
        "file": "traffic_management_paris.geojson",
        "score": "traffic_management_score",
        "raw": [
            "traffic_management_feux_count_raw",
            "traffic_management_anomaly_count_raw",
        ],
    },
    "zoning_laws": {
        "file": "zoning_laws_paris.geojson",
        "score": "zoning_laws_score",
        "raw": [
            "zoning_laws_aire_pietonne_flag",
            "zoning_laws_zone_rencontre_flag",
            "zoning_laws_ztl_flag",
            "zoning_laws_paris_respire_flag",
        ],
    },
    "bicycle_traffic": {
        "file": "bicycle_traffic_paris.geojson",
        "score": "bicycle_traffic_score",
        "raw": [
            "bicycle_traffic_piste_flag",
            "bicycle_traffic_velib_count_raw",
        ],
    },
    "charging_station_proximity": {
        "file": "charging_station_proximity_paris.geojson",
        "score": "charging_station_proximity_score",
        "raw": ["charging_station_proximity_count_raw"],
    },
    "surveillance_coverage": {
        "file": "surveillance_coverage_paris.geojson",
        "score": "surveillance_coverage_score",
        "raw": [],
    },
    "bike_lane_availability": {
        "file": "bike_lane_availability_paris.geojson",
        "score": "bike_lane_availability_score",
        "raw": [],
    },
}

# Extra Paris-only features kept in the CSV but not in the AHP weights.
EXTRA_FEATURES = {
    "street_lighting": {
        "file": "street_lighting_paris.geojson",
        "score": "street_lighting_score",
        "raw": ["street_lighting_lamp_count_raw"],
    },
    "shade": {
        "file": "shade_paris.geojson",
        "score": "shade_score",
        "raw": ["shade_tree_count_raw"],
    },
}


def load_weights(weights_path):
    """Load survey AHP weights indexed by feature name."""
    weights = pd.read_csv(weights_path)
    return weights.set_index("Feature")["Weight"].astype(float).to_dict()


def load_feature_table(processed_dir, feature_name, spec):
    """Load one feature GeoJSON and keep only join key + score/raw columns."""
    path = os.path.join(processed_dir, spec["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} for feature {feature_name}")

    wanted = ["point_index", spec["score"]] + list(spec.get("raw", []))
    gdf = gpd.read_file(path, columns=wanted)
    keep = [col for col in wanted if col in gdf.columns]
    table = pd.DataFrame(gdf[keep]).copy()

    # Canonical score column name without the redundant suffix when already named.
    score_col = spec["score"]
    if score_col != feature_name and score_col in table.columns:
        table = table.rename(columns={score_col: feature_name})
    return table


def main(args):
    """Merge Paris feature layers, apply NYC weights/polarities, and write CSV + GeoJSON."""
    processed_dir = os.path.join(root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    weights_path = args.weights_path
    points_path = os.path.join(processed_dir, "sidewalks_paris_segmentized.geojson")
    csv_path = os.path.join(processed_dir, "robotability_features_paris.csv")
    geojson_path = os.path.join(processed_dir, "robotability_score_paris.geojson")

    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Missing {points_path}. Run segmentize_sidewalks.py first.")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing {weights_path}.")

    weights = load_weights(weights_path)
    print(f"Loaded {len(weights)} feature weights")

    print("Loading segmentized sidewalk sample points ...")
    points = gpd.read_file(points_path)
    points = points.to_crs("EPSG:4326")
    points["lon"] = points.geometry.x
    points["lat"] = points.geometry.y
    print(f"Loaded {len(points)} sample points")

    meta_keep = [col for col in METADATA_COLUMNS if col in points.columns] + ["geometry"]
    merged = points[meta_keep].copy()

    print("Merging weighted survey features ...")
    for feature_name, spec in FEATURE_SOURCES.items():
        print(f"  - {feature_name}")
        table = load_feature_table(processed_dir, feature_name, spec)
        merged = merged.merge(table, on="point_index", how="left", suffixes=("", f"_{feature_name}"))

    print("Merging extra non-weighted features ...")
    for feature_name, spec in EXTRA_FEATURES.items():
        print(f"  - {feature_name}")
        table = load_feature_table(processed_dir, feature_name, spec)
        merged = merged.merge(table, on="point_index", how="left")

    # Build the NYC-compatible score: weight * polarity * feature_value.
    # Flip Paris features that are already high=good while NYC polarity assumes high=bad.
    print("Computing robotability score ...")
    score = pd.Series(0.0, index=merged.index, dtype=float)
    for feature_name, weight in weights.items():
        if feature_name not in merged.columns:
            raise KeyError(f"Weighted feature missing from merged table: {feature_name}")
        if feature_name not in POLARITIES:
            raise KeyError(f"No polarity defined for weighted feature: {feature_name}")

        values = pd.to_numeric(merged[feature_name], errors="coerce").fillna(0.0)
        if feature_name in PARIS_ALREADY_GOOD_ORIENTED:
            values = 1.0 - values
        contribution = weight * POLARITIES[feature_name] * values
        merged[f"{feature_name}_contribution"] = contribution
        score = score + contribution

    merged["robotability_score"] = score

    # Also store a 0-1 display version for maps (same idea as NYC viz assuming unit-ish scores).
    lo = merged["robotability_score"].min()
    hi = merged["robotability_score"].max()
    if hi > lo:
        merged["robotability_score_01"] = (merged["robotability_score"] - lo) / (hi - lo)
    else:
        merged["robotability_score_01"] = 0.0

    # CSV: all attributes, no geometry column (lon/lat kept).
    csv_columns = [col for col in merged.columns if col != "geometry"]
    print(f"Writing {csv_path} ...")
    merged[csv_columns].to_csv(csv_path, index=False)

    # Individual GeoJSON: one Feature per sample point, matching the NYC per-point score export.
    score_gdf = gpd.GeoDataFrame(
        merged[
            [
                "point_index",
                "segment_index",
                "sidewalk_id",
                "pvp_tile",
                "arrondissement_id",
                "arrondissement",
                "qa_c_qu",
                "qa_l_qu",
                "qa_c_ar",
                "robotability_score",
                "robotability_score_01",
                "geometry",
            ]
        ].copy(),
        geometry="geometry",
        crs="EPSG:4326",
    )
    print(f"Writing {geojson_path} ...")
    score_gdf.to_file(geojson_path, driver="GeoJSON")

    print(
        "robotability_score stats:\n"
        f"{merged['robotability_score'].describe().to_string()}"
    )
    print(
        "robotability_score_01 stats:\n"
        f"{merged['robotability_score_01'].describe().to_string()}"
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge Paris features and compute the final robotability score."
    )
    parser.add_argument(
        "--weights_path",
        default=os.path.join(repo_root, "survey_processing", "feature_weights.csv"),
        help="Path to survey feature_weights.csv",
    )
    args = parser.parse_args()
    main(args)
